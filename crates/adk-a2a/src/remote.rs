//! [`RemoteA2aAgent`]: a remote A2A agent that behaves like a local ADK one.

use std::sync::Arc;

use adk_agents::Agent;
use adk_core::{AdkError, Content, Event, InvocationContext, RequestInput, Result};
use futures::stream::BoxStream;
use futures::StreamExt;
use rusty_a2a::client::A2aClient;
use rusty_a2a::types::{
    AgentCard, Message as A2aMessage, StreamResponse, Task, TaskState, TaskStatus,
};

use crate::convert::{content_to_message, message_to_content, part_to_adk};

/// State key holding the remote task this conversation is mid-way through.
///
/// Scoped by agent name so a session may delegate to several remote agents
/// without their tasks colliding.
fn task_key(agent_name: &str) -> String {
    format!("a2a:{agent_name}:task_id")
}

/// A remote A2A agent, usable anywhere an ADK [`Agent`] is.
///
/// The mirror of [`AdkAgentExecutor`](crate::AdkAgentExecutor): that serves an
/// ADK agent *to* A2A callers, this consumes a remote A2A agent *from* an ADK
/// run. Register one as a sub-agent, or wrap it in an `AgentNode`, and the fact
/// that it is a process away in another language stops mattering.
///
/// ```no_run
/// # tokio_test::block_on(async {
/// use adk_a2a::RemoteA2aAgent;
///
/// let researcher = RemoteA2aAgent::discover("researcher", "http://localhost:8080").await?;
/// // ...then `.sub_agent(Arc::new(researcher))` on an LlmAgent builder.
/// # Ok::<(), adk_core::AdkError>(())
/// # }).unwrap();
/// ```
///
/// # How a turn travels
///
/// The agent sends the session's latest user turn to the remote, and turns
/// what comes back into ADK events:
///
/// | A2A | ADK |
/// |---|---|
/// | `Message` reply | one event carrying the text |
/// | `Working` status | nothing — progress, not content |
/// | terminal status message | the final response event |
/// | `InputRequired` | an event carrying [`RequestInput`] |
/// | `Failed` / `Rejected` | an event with an error code |
/// | artifacts | saved to the local `ArtifactService` |
///
/// # Continuing a remote task
///
/// The remote task id is kept in session state under `a2a:<name>:task_id`, so
/// a follow-up turn continues the same remote task rather than starting a new
/// one. That is what lets a remote `InputRequired` be answered: the next thing
/// the user says is sent back on the task that asked. The key is cleared when
/// the remote reaches a terminal state.
///
/// Because it rides on ordinary state, it persists exactly as far as the
/// session service does — with the SQLite backend, a remote agent's question
/// can be answered after a restart.
///
/// # Streaming
///
/// Streaming is used when the remote's card declares it, so status and
/// artifact events arrive as they happen; otherwise a single blocking send is
/// made. Either way the ADK caller sees the same event shapes.
pub struct RemoteA2aAgent {
    name: String,
    description: String,
    client: A2aClient,
    streaming: bool,
}

impl RemoteA2aAgent {
    /// Wraps an already-configured client.
    ///
    /// Use this when the endpoint needs credentials or a custom HTTP client;
    /// [`RemoteA2aAgent::discover`] is the shorter path when it does not.
    pub fn new(name: impl Into<String>, description: impl Into<String>, client: A2aClient) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            client,
            streaming: false,
        }
    }

    /// Fetches the remote's agent card and configures from it.
    ///
    /// The card's description is what the local model reads when deciding
    /// whether to delegate, so taking it from the remote keeps the two in step
    /// — the remote is the authority on what it does. The local `name` stays a
    /// caller's choice, since it has to be unique among *its* siblings.
    pub async fn discover(name: impl Into<String>, base_url: &str) -> Result<Self> {
        let (client, card) = A2aClient::discover(base_url)
            .await
            .map_err(|e| AdkError::Other(format!("A2A discovery failed for {base_url}: {e}")))?;
        Ok(Self::from_card(name, client, &card))
    }

    /// Configures from a card already in hand.
    pub fn from_card(name: impl Into<String>, client: A2aClient, card: &AgentCard) -> Self {
        Self {
            name: name.into(),
            description: card.description.clone(),
            client,
            streaming: card.capabilities.streaming.unwrap_or(false),
        }
    }

    /// Overrides the description the local model reads.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Forces streaming on or off, rather than following the remote's card.
    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }

    /// Behind shared ownership, for registering as a sub-agent.
    pub fn shared(self) -> Arc<dyn Agent> {
        Arc::new(self)
    }

    /// The message to send: the session's most recent user turn.
    fn outbound_message(&self, ctx: &InvocationContext) -> Option<A2aMessage> {
        let content = ctx.with_session(|session| {
            session
                .events
                .iter()
                .rev()
                .find(|e| e.author == "user" && !e.is_partial())
                .and_then(|e| e.content.clone())
        })?;
        // `content_to_message` labels its output as the agent speaking, since
        // that is its job in the serving direction. Here the same content is
        // this side's user turn.
        let mut message = content_to_message(&content)?;
        message.role = rusty_a2a::types::Role::User;
        Some(message)
    }
}

/// Saves an A2A artifact locally and returns the delta entry for the event.
///
/// A remote's artifact is only useful to the rest of an ADK run if it lands in
/// the same `ArtifactService` a local tool would have written to.
async fn store_artifact(
    ctx: &InvocationContext,
    artifact: &rusty_a2a::types::Artifact,
) -> Option<(String, u64)> {
    let artifacts = ctx.services().artifact.as_ref()?;
    let part = artifact.parts.first().map(part_to_adk)?;
    // A2A names an artifact by id; `name` is a human label and may be absent.
    let filename = artifact.artifact_id.clone();
    match artifacts
        .save_artifact(
            &ctx.app_name,
            &ctx.user_id,
            &ctx.session_id,
            &filename,
            part,
        )
        .await
    {
        Ok(version) => Some((filename, version)),
        Err(error) => {
            tracing::warn!(%filename, %error, "could not store an artifact from a remote agent");
            None
        }
    }
}

/// Builds the ADK event for one terminal or interrupted remote status.
fn status_event(agent_name: &str, invocation_id: &str, status: &TaskStatus) -> Event {
    let mut event = Event::new(invocation_id, agent_name);
    if let Some(message) = &status.message {
        event = event.with_content(message_to_content(message));
    }
    match status.state {
        TaskState::InputRequired | TaskState::AuthRequired => {
            let hint = status
                .message
                .as_ref()
                .map(|m| m.text())
                .filter(|t| !t.trim().is_empty())
                .unwrap_or_else(|| "the remote agent needs more input".to_string());
            event = event.with_request_input(RequestInput::new(hint));
        }
        TaskState::Failed => {
            event = event.with_error("REMOTE_AGENT_FAILED", remote_reason(status));
        }
        TaskState::Rejected => {
            event = event.with_error("REMOTE_AGENT_REJECTED", remote_reason(status));
        }
        TaskState::Canceled => {
            event = event.with_error("REMOTE_AGENT_CANCELED", remote_reason(status));
        }
        _ => {}
    }
    event
}

fn remote_reason(status: &TaskStatus) -> String {
    status
        .message
        .as_ref()
        .map(|m| m.text())
        .filter(|t| !t.trim().is_empty())
        .unwrap_or_else(|| format!("the remote agent reported {:?}", status.state))
}

impl std::fmt::Debug for RemoteA2aAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `A2aClient` holds an HTTP client and possibly credentials, so it is
        // named rather than printed.
        f.debug_struct("RemoteA2aAgent")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("streaming", &self.streaming)
            .finish_non_exhaustive()
    }
}

#[async_trait::async_trait]
impl Agent for RemoteA2aAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn run<'a>(&'a self, ctx: &'a InvocationContext) -> BoxStream<'a, Result<Event>> {
        Box::pin(async_stream::try_stream! {
            let Some(mut message) = self.outbound_message(ctx) else {
                // Nothing to relay. Saying so beats sending the remote an empty
                // turn and reporting whatever it makes of that.
                yield Event::new(&ctx.invocation_id, &self.name).with_content(
                    Content::model_text("no user message to delegate"),
                );
                return;
            };

            // One ADK session is one remote conversation, and a remote task in
            // progress is continued rather than replaced.
            message = message.with_context_id(&ctx.session_id);
            let key = task_key(&self.name);
            if let Some(task_id) = ctx.get_state(&key).and_then(|v| v.as_str().map(str::to_string)) {
                message = message.with_task_id(task_id);
            }

            if self.streaming {
                let mut stream = self
                    .client
                    .send_streaming_message(message, None)
                    .await
                    .map_err(remote_error)?;

                while let Some(item) = stream.next().await {
                    let response = item.map_err(remote_error)?;
                    match response {
                        StreamResponse::Message { message } => {
                            yield carry(ctx, Event::new(&ctx.invocation_id, &self.name)
                                .with_content(message_to_content(&message)));
                        }
                        StreamResponse::Task { task } => {
                            remember(ctx, &key, &task.id, task.status.state);
                        }
                        StreamResponse::StatusUpdate { status_update } => {
                            // Recorded before the event is built, so the write
                            // rides out on it.
                            remember(
                                ctx,
                                &key,
                                &status_update.task_id,
                                status_update.status.state,
                            );
                            // `Working` is progress, not content; forwarding it
                            // would put an empty event in the transcript.
                            if status_update.status.state == TaskState::Working {
                                continue;
                            }
                            yield carry(ctx, status_event(
                                &self.name,
                                &ctx.invocation_id,
                                &status_update.status,
                            ));
                        }
                        StreamResponse::ArtifactUpdate { artifact_update } => {
                            if !artifact_update.last_chunk {
                                continue;
                            }
                            if let Some((filename, version)) =
                                store_artifact(ctx, &artifact_update.artifact).await
                            {
                                let mut event = Event::new(&ctx.invocation_id, &self.name);
                                event.actions.artifact_delta.insert(filename, version);
                                yield carry(ctx, event);
                            }
                        }
                    }
                }
            } else {
                let result = self
                    .client
                    .send_message(message, None)
                    .await
                    .map_err(remote_error)?;
                match result {
                    rusty_a2a::types::SendMessageResult::Message { message } => {
                        yield carry(ctx, Event::new(&ctx.invocation_id, &self.name)
                            .with_content(message_to_content(&message)));
                    }
                    rusty_a2a::types::SendMessageResult::Task { task } => {
                        remember(ctx, &key, &task.id, task.status.state);
                        for event in task_events(ctx, &self.name, &task).await {
                            yield carry(ctx, event);
                        }
                    }
                }
            }

            // A write staged after the last yield would never reach the session
            // service — the runner persists a delta only when an event carries
            // it — so anything left over needs an event of its own.
            let leftover = ctx.take_state_delta();
            if !leftover.is_empty() {
                let mut event = Event::new(&ctx.invocation_id, &self.name);
                event.actions.state_delta = leftover;
                yield event;
            }
        })
    }
}

/// Records — or clears — the remote task this conversation is mid-way through.
///
/// Staged into the invocation's state; [`carry`] is what actually gets it onto
/// an event, which is the only way the runner will persist it.
fn remember(ctx: &InvocationContext, key: &str, task_id: &str, state: TaskState) {
    if state.is_terminal() {
        ctx.set_state(key, serde_json::Value::Null);
    } else {
        ctx.set_state(key, task_id);
    }
}

/// Attaches whatever state this agent has staged to an outgoing event.
///
/// ADK persists a state write only when an event carries its delta, so an
/// agent that stages a write has to hand it to the next event it yields.
fn carry(ctx: &InvocationContext, mut event: Event) -> Event {
    let delta = ctx.take_state_delta();
    if !delta.is_empty() {
        event.actions.state_delta.extend(delta);
    }
    event
}

/// The events a whole `Task` snapshot implies, artifacts first.
async fn task_events(ctx: &InvocationContext, agent_name: &str, task: &Task) -> Vec<Event> {
    let mut events = Vec::new();
    for artifact in &task.artifacts {
        if let Some((filename, version)) = store_artifact(ctx, artifact).await {
            let mut event = Event::new(&ctx.invocation_id, agent_name);
            event.actions.artifact_delta.insert(filename, version);
            events.push(event);
        }
    }
    events.push(status_event(agent_name, &ctx.invocation_id, &task.status));
    events
}

/// A transport or protocol failure talking to the remote.
fn remote_error(error: rusty_a2a::client::ClientError) -> AdkError {
    AdkError::Other(format!("remote A2A agent failed: {error}"))
}
