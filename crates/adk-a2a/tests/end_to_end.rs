//! End-to-end: a real A2A client talking to a real ADK agent.
//!
//! The bridge is only worth anything if a peer that knows nothing about ADK
//! can drive it, so these tests go over an actual socket with `rusty_a2a`'s
//! own client — the closest stand-in available for the Python or Go agent that
//! would be on the other end in practice.

use std::sync::Arc;
use std::time::Duration;

use adk_a2a::{card_for_agent, AdkAgentExecutor};
use adk_agents::LlmAgent;
use adk_core::{RunConfig, Schema, Services};
use adk_graph::{chain, FunctionNode, Graph, NodeConfig, NodeOutcome};
use adk_models::MockModel;
use adk_runner::Runner;
use adk_sessions::{InMemoryArtifactService, InMemorySessionService};
use adk_tools::{FunctionTool, ToolSource};
use futures::StreamExt;
use rusty_a2a::client::A2aClient;
use rusty_a2a::server::AgentServer;
use rusty_a2a::types::{
    AgentInterface, Message, Part, SendMessageResult, StreamResponse, TaskState,
};
use serde_json::json;
use tokio::net::TcpListener;

/// Serves `card` + `executor` on an ephemeral port, returning a ready client.
async fn spawn(
    card: rusty_a2a::types::AgentCard,
    executor: Arc<AdkAgentExecutor>,
    url: String,
    listener: TcpListener,
) -> A2aClient {
    let router = AgentServer::new(card, executor).into_router();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    // Poll discovery rather than sleeping a fixed amount: the test should not
    // be a race on how fast this machine binds a socket.
    for _ in 0..100 {
        if A2aClient::fetch_agent_card(&url).await.is_ok() {
            return A2aClient::new(&url);
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("server at {url} never became ready");
}

/// Serves an ADK runner and returns a client pointed at it.
async fn serve(runner: Runner) -> A2aClient {
    serve_executor(AdkAgentExecutor::new(Arc::new(runner))).await
}

async fn serve_executor(executor: AdkAgentExecutor) -> A2aClient {
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let card = rusty_a2a::types::AgentCard::new(
        "test-agent",
        "An ADK agent behind A2A.",
        "0.1.0",
        AgentInterface::json_rpc(&url),
    )
    .with_streaming(true);
    spawn(card, Arc::new(executor), url, listener).await
}

fn services() -> Services {
    Services::new(Arc::new(InMemorySessionService::new()))
}

fn echo_runner(reply: &str) -> Runner {
    let agent = LlmAgent::builder("echo")
        .model(Arc::new(MockModel::new().push_text(reply)))
        .description("Says one thing.")
        .build()
        .unwrap()
        .shared();
    Runner::new("app", agent, services())
}

#[tokio::test]
async fn a_client_gets_the_agents_answer_as_a_completed_task() {
    let client = serve(echo_runner("It is sunny in Paris.")).await;

    let result = client
        .send_message(Message::user_text("Weather in Paris?"), None)
        .await
        .unwrap();

    let task = result.as_task().expect("expected a task");
    assert_eq!(task.status.state, TaskState::Completed);
    assert_eq!(
        task.status.message.as_ref().unwrap().text(),
        "It is sunny in Paris."
    );
}

#[tokio::test]
async fn one_context_is_one_adk_session_across_turns() {
    let agent = LlmAgent::builder("counter")
        .model(Arc::new(
            MockModel::new().push_text("first").push_text("second"),
        ))
        .build()
        .unwrap()
        .shared();
    let runner = Runner::new("app", agent, services());
    let client = serve(runner).await;

    let first = client
        .send_message(Message::user_text("one"), None)
        .await
        .unwrap();
    let context_id = first
        .as_task()
        .unwrap()
        .context_id
        .clone()
        .expect("a task carries its context");

    let second = client
        .send_message(Message::user_text("two").with_context_id(&context_id), None)
        .await
        .unwrap();

    // Same conversation, so the second turn saw the first: the mock model is
    // scripted in order, and it only reaches "second" if both turns landed in
    // one session rather than each starting fresh.
    assert_eq!(
        second
            .as_task()
            .unwrap()
            .status
            .message
            .as_ref()
            .unwrap()
            .text(),
        "second"
    );
    assert_eq!(
        second.as_task().unwrap().context_id.as_deref(),
        Some(context_id.as_str())
    );
}

#[tokio::test]
async fn a_streaming_client_sees_working_then_completed() {
    let client = serve(echo_runner("done")).await;

    let mut stream = client
        .send_streaming_message(Message::user_text("go"), None)
        .await
        .unwrap();

    let mut states = Vec::new();
    while let Some(event) = stream.next().await {
        if let StreamResponse::StatusUpdate { status_update } = event.unwrap() {
            states.push(status_update.status.state);
        }
    }
    assert_eq!(states, vec![TaskState::Working, TaskState::Completed]);
}

#[tokio::test]
async fn an_adk_artifact_reaches_the_peer_as_an_a2a_artifact() {
    let tool = FunctionTool::new(
        "write_report",
        "Writes a report artifact.",
        Schema::object(),
        |_args, ctx| {
            let ctx = ctx.clone();
            Box::pin(async move {
                ctx.save_artifact("report.txt", adk_core::Part::text("the report"))
                    .await?;
                Ok(adk_tools::success(json!({"written": true})))
            })
        },
    );
    let agent = LlmAgent::builder("writer")
        .model(Arc::new(
            MockModel::new()
                .push_call_json("write_report", json!({}))
                .push_text("Report written."),
        ))
        .tool(ToolSource::Tool(tool.shared()))
        .build()
        .unwrap()
        .shared();

    let runner = Runner::new(
        "app",
        agent,
        services().with_artifact(Arc::new(InMemoryArtifactService::new())),
    );
    let client = serve(runner).await;

    let result = client
        .send_message(Message::user_text("write it"), None)
        .await
        .unwrap();
    let task = result.as_task().unwrap();

    assert_eq!(
        task.artifacts.len(),
        1,
        "expected the artifact to cross over"
    );
    assert_eq!(task.artifacts[0].artifact_id, "report.txt");
    assert_eq!(task.artifacts[0].parts[0].as_text(), Some("the report"));
}

/// The mapping the bridge exists for: an ADK graph suspension is an A2A
/// `InputRequired` task, and the client's reply on that task resumes the graph
/// at the node that suspended.
#[tokio::test]
async fn a_graph_suspension_becomes_input_required_and_resumes() {
    let approve = FunctionNode::new("approve", NodeConfig::default(), |ctx| {
        let ctx = ctx.clone();
        Box::pin(async move {
            let answer = ctx.resume_or_request_input("Approve this refund?", None)?;
            Ok(NodeOutcome::output(answer))
        })
    })
    .shared();
    let graph = Arc::new(Graph::new(vec![approve], chain(["approve"])).unwrap());
    let client = serve(Runner::new("app", graph, services())).await;

    // First turn: the node suspends, so the task parks on InputRequired with
    // the node's prompt attached.
    let first = client
        .send_message(Message::user_text("refund please"), None)
        .await
        .unwrap();
    let task = first.as_task().expect("expected a task");
    assert_eq!(task.status.state, TaskState::InputRequired);
    assert_eq!(
        task.status.message.as_ref().unwrap().text(),
        "Approve this refund?"
    );

    // Second turn: answering on the same task resumes the suspended node.
    let reply = Message::user_text("approved")
        .with_task_id(&task.id)
        .with_context_id(task.context_id.clone().unwrap());
    let second = client.send_message(reply, None).await.unwrap();

    let resumed = second.as_task().expect("expected a task");
    assert_eq!(resumed.id, task.id, "the same task continued");
    assert_eq!(resumed.status.state, TaskState::Completed);
}

#[tokio::test]
async fn a_failing_run_surfaces_as_a_failed_task() {
    // A zero model-call budget makes the run fail with `LimitExceeded` on the
    // agent's very first turn — a real ADK error, raised by the framework
    // rather than staged, which is what the bridge has to translate.
    let executor =
        AdkAgentExecutor::new(Arc::new(echo_runner("never reached"))).with_run_config(RunConfig {
            max_llm_calls: 0,
            ..RunConfig::default()
        });
    let client = serve_executor(executor).await;

    let result = client
        .send_message(Message::user_text("go"), None)
        .await
        .unwrap();
    match result {
        SendMessageResult::Task { task } => {
            assert_eq!(task.status.state, TaskState::Failed);
            let reason = task.status.message.as_ref().unwrap().text();
            assert!(
                reason.contains("max_llm_calls"),
                "the failure should name the limit it hit: {reason}"
            );
        }
        other => panic!("expected a failed task, got {other:?}"),
    }
}

#[tokio::test]
async fn structured_data_from_a_peer_reaches_the_agent() {
    let agent = LlmAgent::builder("reader")
        .model(Arc::new(MockModel::new().push_text("read it")))
        .build()
        .unwrap()
        .shared();
    let runner = Runner::new("app", agent, services());
    let executor = AdkAgentExecutor::new(Arc::new(runner));
    let runner_handle = Arc::clone(executor.runner());
    let client = serve_executor(executor).await;

    let message = Message::new(
        rusty_a2a::types::Role::User,
        vec![Part::data(json!({"city": "Kyoto"}))],
    );
    let result = client.send_message(message, None).await.unwrap();
    let context_id = result.as_task().unwrap().context_id.clone().unwrap();

    // The agent's history holds the data part as its JSON text.
    let session = runner_handle
        .session(&context_id, &context_id)
        .await
        .unwrap()
        .unwrap();
    assert!(session
        .events
        .iter()
        .any(|e| e.text().contains(r#""city":"Kyoto""#)));
}

#[tokio::test]
async fn the_generated_card_describes_the_adk_agent() {
    let agent = LlmAgent::builder("support")
        .model(Arc::new(MockModel::new()))
        .description("Front desk.")
        .build()
        .unwrap()
        .shared();
    let card = card_for_agent(
        &agent,
        "2.0.0",
        AgentInterface::json_rpc("http://localhost:9999"),
    );
    assert_eq!(card.name, "support");
    assert_eq!(card.description, "Front desk.");
    assert_eq!(card.version, "2.0.0");
}
