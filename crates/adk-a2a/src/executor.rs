//! [`AdkAgentExecutor`]: an ADK [`Runner`] behind A2A's `AgentExecutor` trait.

use std::sync::Arc;

use adk_core::{AdkError, Event, Result as AdkResult, RunConfig};
use adk_graph::{PendingInterrupt, ResumeRequest, PENDING_STATE_KEY};
use adk_runner::Runner;
use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use rusty_a2a::error::{A2aError, Result as A2aResult};
use rusty_a2a::server::{AgentExecutor, EventSink, RequestContext};
use rusty_a2a::types::{Artifact, Message as A2aMessage, Part as A2aPart, TaskState};
use serde_json::Value;

use crate::convert::{content_to_message, message_to_content, part_to_a2a};

/// Decides which ADK user a given A2A request belongs to.
pub type UserResolver = Arc<dyn Fn(&RequestContext) -> String + Send + Sync>;

/// Serves an ADK [`Runner`] as an A2A agent.
///
/// Hand one to `rusty_a2a`'s `AgentServer` and the ADK agent behind it becomes
/// reachable by any A2A client, in any language, over whichever bindings that
/// server exposes.
///
/// # How the two models line up
///
/// | A2A | ADK | Note |
/// |---|---|---|
/// | `contextId` | session id | one A2A conversation is one ADK session |
/// | `taskId` | one invocation | a task spans a suspend/resume pair |
/// | `Working` | run started | emitted once, before the first agent event |
/// | `InputRequired` | graph suspension | see [below](#human-in-the-loop) |
/// | `Completed` | final response | carries the agent's answer |
/// | artifacts | `artifact_delta` | loaded from the `ArtifactService` |
///
/// A2A has no user identity on a message, so by default the `contextId` is
/// also the ADK user id. That is the conservative reading: it keeps one
/// conversation's `user:`-scoped state out of another's, at the cost of not
/// sharing anything across a caller's conversations. An application that knows
/// who is calling — from an `AuthVerifier`, or from message metadata — should
/// say so with [`AdkAgentExecutor::with_user_resolver`].
///
/// # Human-in-the-loop
///
/// The mapping worth having. When an ADK graph node suspends via
/// `resume_or_request_input`, the task goes to `InputRequired` with the node's
/// prompt attached. The client answers by sending another message carrying the
/// same `taskId`; the bridge reads the interrupt id the graph persisted and
/// resumes the run from exactly that node. Neither side has to model the
/// other's idea of waiting for a person.
///
/// # Streaming
///
/// Partial (token-level) ADK events are not forwarded. A2A streams task state
/// and artifacts, not tokens, so a status update per token would be both noisy
/// and a misuse of the field. Callers see `Working`, then artifacts as they are
/// produced, then a terminal status.
pub struct AdkAgentExecutor {
    runner: Arc<Runner>,
    user_for: UserResolver,
    run_config: Option<RunConfig>,
}

impl AdkAgentExecutor {
    /// Wraps a runner.
    pub fn new(runner: Arc<Runner>) -> Self {
        Self {
            runner,
            user_for: Arc::new(|ctx: &RequestContext| ctx.context_id.clone()),
            run_config: None,
        }
    }

    /// Sets how an A2A request maps to an ADK user id.
    pub fn with_user_resolver<F>(mut self, resolver: F) -> Self
    where
        F: Fn(&RequestContext) -> String + Send + Sync + 'static,
    {
        self.user_for = Arc::new(resolver);
        self
    }

    /// Applies a fixed [`RunConfig`] to every run.
    pub fn with_run_config(mut self, config: RunConfig) -> Self {
        self.run_config = Some(config);
        self
    }

    /// The runner this executor drives.
    pub fn runner(&self) -> &Arc<Runner> {
        &self.runner
    }

    /// Loads the session for this conversation, creating it on first contact.
    ///
    /// A2A has no "open a conversation" call — the first message on a new
    /// `contextId` is the opening — so the session is created lazily here.
    async fn session_for(&self, user_id: &str, session_id: &str) -> AdkResult<adk_core::Session> {
        let sessions = &self.runner.services().session;
        if let Some(session) = sessions
            .get_session(self.runner.app_name(), user_id, session_id)
            .await?
        {
            return Ok(session);
        }
        sessions
            .create_session(
                self.runner.app_name(),
                user_id,
                None,
                Some(session_id.to_string()),
            )
            .await
    }

    /// The interrupt a previous run left pending on this session, if any.
    ///
    /// The graph engine persists this itself under [`PENDING_STATE_KEY`], so
    /// the bridge reads ADK's own record rather than keeping a second one that
    /// could drift out of step with it.
    fn pending_interrupt(session: &adk_core::Session) -> Option<PendingInterrupt> {
        let raw = session.state.get(PENDING_STATE_KEY)?;
        if raw.is_null() {
            return None;
        }
        serde_json::from_value(raw.clone()).ok()
    }

    /// Turns the client's reply into the payload the suspended node receives.
    ///
    /// A `data` part is handed over as the structured value it is; anything
    /// else arrives as text, which is what a node prompting a human expects.
    fn resume_payload(message: &rusty_a2a::types::Message) -> Value {
        for part in &message.parts {
            if let rusty_a2a::types::PartContent::Data { data } = &part.content {
                return data.clone();
            }
        }
        Value::String(message.text())
    }

    /// Publishes every artifact an event produced.
    ///
    /// ADK events name artifacts by filename and version; the bytes live in
    /// the `ArtifactService`. With no artifact service configured there is
    /// nothing to load, and the delta is ignored.
    async fn emit_artifacts(
        &self,
        events: &EventSink,
        user_id: &str,
        session_id: &str,
        event: &Event,
    ) {
        let Some(artifacts) = self.runner.services().artifact.as_ref() else {
            return;
        };
        for (filename, version) in &event.actions.artifact_delta {
            let loaded = artifacts
                .load_artifact(
                    self.runner.app_name(),
                    user_id,
                    session_id,
                    filename,
                    Some(*version),
                )
                .await;
            let part = match loaded {
                Ok(Some(part)) => part,
                Ok(None) => continue,
                Err(error) => {
                    tracing::warn!(%filename, %error, "could not load artifact for an A2A peer");
                    continue;
                }
            };
            let Some(part) = part_to_a2a(&part) else {
                continue;
            };
            // The filename is the identity a caller re-requests it by, so it
            // is both the A2A artifact id and its name.
            events
                .artifact(Artifact::new(filename.clone(), vec![part]).with_name(filename.clone()));
        }
    }
}

/// ADK failures reach an A2A peer as the protocol's error vocabulary.
///
/// Only the two cases whose meanings genuinely coincide are mapped; the rest
/// become `Internal`, which is what they are from the caller's side. Reaching
/// for a closer-sounding variant would misinform a peer — exhausting a run
/// budget is not `UnsupportedOperation` (the agent does support the call, this
/// run ran out of room), and a cancelled run is not `TaskNotCancelable` (which
/// means the opposite: a cancel was refused).
fn to_a2a_error(error: AdkError) -> A2aError {
    match error {
        AdkError::SessionNotFound(id) => A2aError::TaskNotFound(id),
        AdkError::Validation { field, message } => {
            A2aError::InvalidParams(format!("{field}: {message}"))
        }
        other => A2aError::Internal(other.to_string()),
    }
}

#[async_trait]
impl AgentExecutor for AdkAgentExecutor {
    async fn execute(&self, ctx: RequestContext, events: EventSink) -> A2aResult<()> {
        let user_id = (self.user_for)(&ctx);
        let session_id = ctx.context_id.clone();

        let session = self
            .session_for(&user_id, &session_id)
            .await
            .map_err(to_a2a_error)?;

        // A message on an existing task that is parked on a suspension is an
        // answer to it, not a new turn.
        let resume = ctx
            .task
            .as_ref()
            .filter(|task| {
                task.status.state == TaskState::AuthRequired
                    || task.status.state == TaskState::InputRequired
            })
            .and_then(|_| Self::pending_interrupt(&session))
            .map(|pending| {
                ResumeRequest::new(pending.interrupt_id, Self::resume_payload(&ctx.message))
            });

        let mut stream: BoxStream<'_, AdkResult<Event>> = match resume {
            Some(resume) => {
                self.runner
                    .resume(&user_id, &session_id, resume, self.run_config.clone())
            }
            None => self.runner.run(
                &user_id,
                &session_id,
                message_to_content(&ctx.message),
                self.run_config.clone(),
            ),
        };

        events.status(TaskState::Working);

        let mut answer: Option<A2aMessage> = None;
        loop {
            let next = tokio::select! {
                biased;
                _ = ctx.cancellation.cancelled() => {
                    // The client called CancelTask. Stop consuming; dropping
                    // the stream is what ends the ADK run.
                    events.status(TaskState::Canceled);
                    return Ok(());
                }
                next = stream.next() => next,
            };
            let Some(event) = next else { break };
            let event = event.map_err(to_a2a_error)?;

            // The user's own turn is echoed back by the runner, and partials
            // are token-level chunks a later event supersedes.
            if event.author == "user" || event.is_partial() {
                continue;
            }

            self.emit_artifacts(&events, &user_id, &session_id, &event)
                .await;

            if let Some(request) = &event.request_input {
                // A2A's InputRequired closes the stream, so the run ends here
                // and the client's next message on this task resumes it.
                let mut prompt = A2aMessage::agent_text(&request.hint);
                if let Some(payload) = &request.payload {
                    prompt.parts.push(A2aPart::data(payload.clone()));
                }
                events.status_with_message(TaskState::InputRequired, Some(prompt));
                return Ok(());
            }

            if event.is_final_response() {
                if let Some(message) = event.content.as_ref().and_then(content_to_message) {
                    answer = Some(message);
                }
            }
        }

        events.status_with_message(TaskState::Completed, answer);
        Ok(())
    }
}
