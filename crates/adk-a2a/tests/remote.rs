//! End-to-end: an ADK run delegating out to a remote A2A agent.
//!
//! The remote is a real `rusty_a2a` `AgentServer` on a real socket, so these
//! exercise the wire rather than a stub — and the last test closes the loop by
//! making the remote itself an ADK agent served through this crate's other
//! direction.

use std::sync::Arc;
use std::time::Duration;

use adk_a2a::{AdkAgentExecutor, RemoteA2aAgent};
use adk_agents::{Agent, LlmAgent};
use adk_core::{Content, Event, InvocationContext, RunConfig, Services};
use adk_graph::{chain, FunctionNode, Graph, NodeConfig, NodeOutcome};
use adk_models::MockModel;
use adk_runner::Runner;
use adk_sessions::{InMemoryArtifactService, InMemorySessionService};
use async_trait::async_trait;
use futures::StreamExt;
use rusty_a2a::client::A2aClient;
use rusty_a2a::error::Result as A2aResult;
use rusty_a2a::server::{AgentExecutor, AgentServer, EventSink, RequestContext};
use rusty_a2a::types::{AgentCard, AgentInterface, Artifact, Message, Part, TaskState};
use tokio::net::TcpListener;

/// Serves any A2A executor on an ephemeral port; returns its base URL.
async fn serve(executor: Arc<dyn AgentExecutor>, streaming: bool) -> String {
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let card = AgentCard::new(
        "remote-agent",
        "A remote agent that answers questions.",
        "0.1.0",
        AgentInterface::json_rpc(&url),
    )
    .with_streaming(streaming);

    let router = AgentServer::new(card, executor).into_router();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });

    for _ in 0..100 {
        if A2aClient::fetch_agent_card(&url).await.is_ok() {
            return url;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("remote at {url} never became ready");
}

/// A scripted remote: replies, asks for input, fails, or emits an artifact,
/// depending on what it is sent.
struct ScriptedRemote;

#[async_trait]
impl AgentExecutor for ScriptedRemote {
    async fn execute(&self, ctx: RequestContext, events: EventSink) -> A2aResult<()> {
        let text = ctx.message.text();
        events.status(TaskState::Working);

        if text.contains("approve") {
            // A second turn on an existing task is the answer to the question.
            if ctx.task.is_some() && !text.contains("please approve") {
                events.status_with_message(
                    TaskState::Completed,
                    Some(Message::agent_text(format!("acted on: {text}"))),
                );
                return Ok(());
            }
            events.status_with_message(
                TaskState::InputRequired,
                Some(Message::agent_text("Do you approve?")),
            );
            return Ok(());
        }

        if text.contains("fail") {
            events.status_with_message(
                TaskState::Failed,
                Some(Message::agent_text("the remote could not do that")),
            );
            return Ok(());
        }

        if text.contains("report") {
            events.artifact(Artifact::new(
                "remote-report.txt",
                vec![Part::text("findings from the remote")],
            ));
        }

        events.status_with_message(
            TaskState::Completed,
            Some(Message::agent_text("the remote answer")),
        );
        Ok(())
    }
}

fn services() -> Services {
    Services::new(Arc::new(InMemorySessionService::new()))
        .with_artifact(Arc::new(InMemoryArtifactService::new()))
}

/// Drives one turn through a `Runner` whose root is `agent`, returning events.
async fn run_turn(runner: &Runner, session_id: &str, text: &str) -> Vec<Event> {
    runner
        .run("u1", session_id, Content::user_text(text), None)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|e| e.expect("run failed"))
        .collect()
}

#[tokio::test]
async fn a_remote_answer_arrives_as_an_adk_event() {
    let url = serve(Arc::new(ScriptedRemote), true).await;
    let remote = RemoteA2aAgent::discover("researcher", &url).await.unwrap();

    // The card's description crossed over, which is what a delegating model reads.
    assert_eq!(
        remote.description(),
        "A remote agent that answers questions."
    );

    let runner = Runner::new("app", remote.shared(), services());
    let session = runner.create_session("u1", None).await.unwrap();
    let events = run_turn(&runner, &session.id, "what do you know?").await;

    let answer = events
        .iter()
        .filter(|e| e.author == "researcher")
        .find_map(|e| Some(e.text()).filter(|t| !t.is_empty()))
        .expect("expected an answer from the remote");
    assert_eq!(answer, "the remote answer");
}

#[tokio::test]
async fn a_non_streaming_remote_works_the_same_way() {
    let url = serve(Arc::new(ScriptedRemote), false).await;
    let remote = RemoteA2aAgent::discover("researcher", &url).await.unwrap();
    // The card says no streaming, so a blocking send is used.
    assert_eq!(
        remote.description(),
        "A remote agent that answers questions."
    );

    let runner = Runner::new("app", remote.shared(), services());
    let session = runner.create_session("u1", None).await.unwrap();
    let events = run_turn(&runner, &session.id, "what do you know?").await;

    assert!(events
        .iter()
        .any(|e| e.author == "researcher" && e.text() == "the remote answer"));
}

#[tokio::test]
async fn a_remote_failure_is_reported_with_an_error_code() {
    let url = serve(Arc::new(ScriptedRemote), true).await;
    let remote = RemoteA2aAgent::discover("researcher", &url).await.unwrap();

    let runner = Runner::new("app", remote.shared(), services());
    let session = runner.create_session("u1", None).await.unwrap();
    let events = run_turn(&runner, &session.id, "please fail").await;

    let failed = events
        .iter()
        .find(|e| e.error_code.is_some())
        .expect("expected a failure event");
    assert_eq!(failed.error_code.as_deref(), Some("REMOTE_AGENT_FAILED"));
    assert_eq!(
        failed.error_message.as_deref(),
        Some("the remote could not do that")
    );
}

#[tokio::test]
async fn a_remote_artifact_lands_in_the_local_store() {
    let url = serve(Arc::new(ScriptedRemote), true).await;
    let remote = RemoteA2aAgent::discover("researcher", &url).await.unwrap();

    let services = services();
    let artifacts = services.artifact.clone().unwrap();
    let runner = Runner::new("app", remote.shared(), services);
    let session = runner.create_session("u1", None).await.unwrap();
    let events = run_turn(&runner, &session.id, "write me a report").await;

    assert!(
        events
            .iter()
            .any(|e| e.actions.artifact_delta.contains_key("remote-report.txt")),
        "the artifact should be announced on an event"
    );
    let stored = artifacts
        .load_artifact("app", "u1", &session.id, "remote-report.txt", None)
        .await
        .unwrap()
        .expect("the artifact should be readable locally");
    assert_eq!(stored.as_text(), Some("findings from the remote"));
}

/// The remote asks a question; the next local turn answers it on the same
/// remote task, rather than starting a new one.
#[tokio::test]
async fn a_remote_input_request_is_answered_on_the_next_turn() {
    let url = serve(Arc::new(ScriptedRemote), true).await;
    let remote = RemoteA2aAgent::discover("approver", &url).await.unwrap();

    let runner = Runner::new("app", remote.shared(), services());
    let session = runner.create_session("u1", None).await.unwrap();

    let first = run_turn(&runner, &session.id, "please approve this").await;
    let asked = first
        .iter()
        .find(|e| e.request_input.is_some())
        .expect("expected the remote to ask for input");
    assert_eq!(
        asked.request_input.as_ref().unwrap().hint,
        "Do you approve?"
    );

    // The task id was recorded so the follow-up continues the same remote task.
    let saved = runner.session("u1", &session.id).await.unwrap().unwrap();
    let task_id = saved.state.get("a2a:approver:task_id").unwrap();
    assert!(task_id.is_string(), "expected a remembered task id");

    let second = run_turn(&runner, &session.id, "yes, approve").await;
    assert!(
        second.iter().any(|e| e.text().starts_with("acted on:")),
        "the remote should have resumed its task, got {:?}",
        second.iter().map(|e| e.text()).collect::<Vec<_>>()
    );

    // Once the remote finished, the pointer is cleared.
    let saved = runner.session("u1", &session.id).await.unwrap().unwrap();
    assert!(saved
        .state
        .get("a2a:approver:task_id")
        .is_none_or(|v| v.is_null()));
}

#[tokio::test]
async fn an_unreachable_remote_fails_the_run() {
    // Nothing is listening here; discovery must report that rather than hang.
    let error = RemoteA2aAgent::discover("nobody", "http://127.0.0.1:1")
        .await
        .expect_err("discovery should fail");
    assert!(
        error.to_string().contains("A2A discovery failed"),
        "unhelpful error: {error}"
    );
}

#[tokio::test]
async fn a_remote_agent_can_be_a_sub_agent_of_a_local_one() {
    let url = serve(Arc::new(ScriptedRemote), true).await;
    let remote = RemoteA2aAgent::discover("researcher", &url).await.unwrap();

    let root = LlmAgent::builder("concierge")
        .model(Arc::new(MockModel::new().push_text("handled locally")))
        .sub_agent(remote.shared())
        .build()
        .unwrap();

    // The remote shows up in the local hierarchy like any other sub-agent,
    // which is what makes it addressable for delegation.
    let found = root.find_agent("researcher").expect("remote is reachable");
    assert_eq!(
        found.description(),
        "A remote agent that answers questions."
    );
}

/// Both directions at once: an ADK agent served over A2A by this crate, and
/// consumed back through this crate. ADK -> A2A -> ADK.
#[tokio::test]
async fn an_adk_agent_served_over_a2a_can_be_consumed_as_one() {
    // The far side: an ordinary ADK agent behind `AdkAgentExecutor`.
    let far_agent = LlmAgent::builder("far")
        .model(Arc::new(
            MockModel::new().push_text("answered from the far side"),
        ))
        .description("The agent at the far end.")
        .build()
        .unwrap()
        .shared();
    let far_runner = Runner::new("far_app", far_agent, services());
    let url = serve(Arc::new(AdkAgentExecutor::new(Arc::new(far_runner))), true).await;

    // The near side: an ADK run that delegates to it.
    let remote = RemoteA2aAgent::discover("far", &url).await.unwrap();
    let near = Runner::new("near_app", remote.shared(), services());
    let session = near.create_session("u1", None).await.unwrap();

    let events = run_turn(&near, &session.id, "ask the far side").await;
    assert!(
        events
            .iter()
            .any(|e| e.text() == "answered from the far side"),
        "round trip failed, got {:?}",
        events.iter().map(|e| e.text()).collect::<Vec<_>>()
    );
}

/// And with a graph on the far side, a suspension survives the whole round
/// trip: far graph suspends -> A2A InputRequired -> near ADK RequestInput.
#[tokio::test]
async fn a_far_side_suspension_surfaces_locally_and_resumes() {
    let approve = FunctionNode::new("approve", NodeConfig::default(), |ctx| {
        let ctx = ctx.clone();
        Box::pin(async move {
            let answer = ctx.resume_or_request_input("Far side asks: proceed?", None)?;
            Ok(NodeOutcome::output(answer))
        })
    })
    .shared();
    let graph = Arc::new(Graph::new(vec![approve], chain(["approve"])).unwrap());
    let far_runner = Runner::new("far_app", graph, services());
    let url = serve(Arc::new(AdkAgentExecutor::new(Arc::new(far_runner))), true).await;

    let remote = RemoteA2aAgent::discover("far", &url).await.unwrap();
    let near = Runner::new("near_app", remote.shared(), services());
    let session = near.create_session("u1", None).await.unwrap();

    let first = run_turn(&near, &session.id, "start").await;
    let asked = first
        .iter()
        .find(|e| e.request_input.is_some())
        .expect("the far side's suspension should surface locally");
    assert_eq!(
        asked.request_input.as_ref().unwrap().hint,
        "Far side asks: proceed?"
    );

    let second = run_turn(&near, &session.id, "proceed").await;
    assert!(
        second.iter().all(|e| e.request_input.is_none()),
        "the far side should have resumed rather than asking again"
    );
}

/// A run with nothing to relay says so rather than sending an empty turn.
#[tokio::test]
async fn a_turn_with_no_user_message_is_reported() {
    let url = serve(Arc::new(ScriptedRemote), true).await;
    let remote = RemoteA2aAgent::discover("researcher", &url).await.unwrap();

    let ctx = InvocationContext::new(
        adk_core::Session::new("s1", "app", "u1"),
        services(),
        RunConfig::default(),
    );
    let events: Vec<Event> = remote
        .run(&ctx)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|e| e.unwrap())
        .collect();

    assert_eq!(events.len(), 1);
    assert_eq!(events[0].text(), "no user message to delegate");
}
