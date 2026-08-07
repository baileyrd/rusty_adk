//! Serve a Rust ADK agent over the [Agent2Agent (A2A) protocol][a2a].
//!
//! ADK's SDKs share a protocol for *agents* — A2A — and none for tools. This
//! crate is the agent half of interop, the counterpart to `adk-mcp`'s tool
//! half: wrap an ADK [`Runner`](adk_runner::Runner) in an
//! [`AdkAgentExecutor`] and any A2A client, in any language, can talk to it.
//!
//! ```
//! # tokio_test::block_on(async {
//! use adk_a2a::{card_for_agent, AdkAgentExecutor};
//! use adk_agents::LlmAgent;
//! use adk_core::Services;
//! use adk_models::MockModel;
//! use adk_runner::Runner;
//! use adk_sessions::InMemorySessionService;
//! use rusty_a2a::server::AgentServer;
//! use rusty_a2a::types::AgentInterface;
//! use std::sync::Arc;
//!
//! let agent = LlmAgent::builder("greeter")
//!     .model(Arc::new(MockModel::new().push_text("Hello!")))
//!     .description("Greets people.")
//!     .build()?
//!     .shared();
//!
//! let card = card_for_agent(&agent, "0.1.0", AgentInterface::json_rpc("http://localhost:8080"));
//! let runner = Runner::new(
//!     "greeter_app",
//!     agent,
//!     Services::new(Arc::new(InMemorySessionService::new())),
//! );
//!
//! let server = AgentServer::new(card, Arc::new(AdkAgentExecutor::new(Arc::new(runner))));
//! // server.serve(([127, 0, 0, 1], 8080)).await
//! # let _ = server;
//! # Ok::<(), adk_core::AdkError>(())
//! # }).unwrap();
//! ```
//!
//! # What maps to what
//!
//! The two protocols agree on more than they disagree on, and
//! [`AdkAgentExecutor`] documents each correspondence. The one worth calling
//! out here is human-in-the-loop: an ADK graph suspension becomes an A2A
//! `InputRequired` task, and the client's reply on that task resumes the graph
//! at the node that suspended. Both sides already had the concept; the bridge
//! only has to line them up.
//!
//! # Direction
//!
//! This crate serves an ADK agent *to* A2A callers. Calling *out* to a remote
//! A2A agent from inside an ADK run is the mirror image and is not implemented
//! here — use `rusty_a2a`'s client directly from a tool.
//!
//! [a2a]: https://a2a-protocol.org/latest/

#![deny(missing_docs)]
#![warn(clippy::all)]

pub mod card;
pub mod convert;
pub mod executor;

pub use card::card_for_agent;
pub use convert::{content_to_message, message_to_content, part_to_a2a, part_to_adk};
pub use executor::{AdkAgentExecutor, UserResolver};

/// Re-exported so a consumer needs only this crate to build a server.
pub use rusty_a2a;
