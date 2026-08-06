//! Error types shared across the ADK crates.

use thiserror::Error;

/// The result type used throughout the ADK.
pub type Result<T, E = AdkError> = std::result::Result<T, E>;

/// Anything that can go wrong inside an ADK run.
#[derive(Debug, Error)]
pub enum AdkError {
    /// A node suspended the run to wait for human input.
    ///
    /// This is control flow, not a failure. It must propagate to the graph
    /// engine intact — catching it broadly is what ADK 2.0 warns breaks
    /// human-in-the-loop pausing, so treat it as unrecoverable in `?` chains
    /// and match on it explicitly where you mean to handle it.
    #[error("node interrupted awaiting input: {interrupt_id}")]
    NodeInterrupted {
        /// Correlates the suspension with the response that resumes it.
        interrupt_id: String,
    },

    /// A tool is waiting for the user to approve its invocation.
    #[error("tool call {function_call_id} awaiting confirmation")]
    ConfirmationRequired {
        /// The function call that needs approval.
        function_call_id: String,
    },

    /// A tool failed while executing.
    #[error("tool '{tool}' failed: {message}")]
    Tool {
        /// Name of the failing tool.
        tool: String,
        /// What went wrong.
        message: String,
    },

    /// The model provider returned an error or an unusable response.
    #[error("model '{model}' error: {message}")]
    Model {
        /// Model identifier.
        model: String,
        /// What went wrong.
        message: String,
    },

    /// A graph is malformed — a dangling edge, a missing entry point, a cycle
    /// where none is allowed.
    #[error("invalid graph: {0}")]
    Graph(String),

    /// Execution reached a node with no matching outgoing edge and no default.
    #[error("no route matched from node '{node}' for routes {routes:?}")]
    NoRoute {
        /// The node that emitted the unmatched routes.
        node: String,
        /// The route labels it emitted.
        routes: Vec<String>,
    },

    /// A session could not be found.
    #[error("session '{0}' not found")]
    SessionNotFound(String),

    /// An artifact could not be found.
    #[error("artifact '{0}' not found")]
    ArtifactNotFound(String),

    /// A value did not match its declared schema.
    #[error("schema validation failed for '{field}': {message}")]
    Validation {
        /// The offending field or parameter.
        field: String,
        /// Why it was rejected.
        message: String,
    },

    /// A persistent store rejected an operation.
    ///
    /// `retryable` distinguishes a transient condition — a locked or busy
    /// database, a dropped connection — from a permanent one such as a
    /// constraint violation, so callers need not parse the message.
    #[error("storage error: {message}")]
    Storage {
        /// What the store reported.
        message: String,
        /// Whether the same operation could succeed if tried again.
        retryable: bool,
    },

    /// A configuration value was missing or contradictory.
    #[error("configuration error: {0}")]
    Config(String),

    /// The run exceeded a limit — loop iterations, retries, or step budget.
    #[error("limit exceeded: {0}")]
    LimitExceeded(String),

    /// The run was cancelled by its caller.
    #[error("run cancelled")]
    Cancelled,

    /// JSON encoding or decoding failed.
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// An I/O operation failed.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Anything not covered above.
    #[error("{0}")]
    Other(String),
}

impl AdkError {
    /// Builds a tool failure.
    pub fn tool(tool: impl Into<String>, message: impl Into<String>) -> Self {
        AdkError::Tool {
            tool: tool.into(),
            message: message.into(),
        }
    }

    /// Builds a model failure.
    pub fn model(model: impl Into<String>, message: impl Into<String>) -> Self {
        AdkError::Model {
            model: model.into(),
            message: message.into(),
        }
    }

    /// Builds a validation failure.
    pub fn validation(field: impl Into<String>, message: impl Into<String>) -> Self {
        AdkError::Validation {
            field: field.into(),
            message: message.into(),
        }
    }

    /// Builds a permanent storage failure.
    pub fn storage(message: impl Into<String>) -> Self {
        AdkError::Storage {
            message: message.into(),
            retryable: false,
        }
    }

    /// Builds a transient storage failure, such as a busy or locked database.
    pub fn storage_retryable(message: impl Into<String>) -> Self {
        AdkError::Storage {
            message: message.into(),
            retryable: true,
        }
    }

    /// True when this is a control-flow signal rather than a failure.
    ///
    /// The graph engine's retry logic consults this so it never retries a
    /// deliberate suspension.
    pub fn is_control_flow(&self) -> bool {
        matches!(
            self,
            AdkError::NodeInterrupted { .. } | AdkError::ConfirmationRequired { .. }
        )
    }

    /// Whether retrying the operation could plausibly succeed.
    pub fn is_retryable(&self) -> bool {
        match self {
            _ if self.is_control_flow() => false,
            AdkError::Model { .. } | AdkError::Io(_) => true,
            AdkError::Tool { .. } => true,
            AdkError::Storage { retryable, .. } => *retryable,
            _ => false,
        }
    }

    /// A short, stable code suitable for [`crate::Event::error_code`].
    pub fn code(&self) -> &'static str {
        match self {
            AdkError::NodeInterrupted { .. } => "NODE_INTERRUPTED",
            AdkError::ConfirmationRequired { .. } => "CONFIRMATION_REQUIRED",
            AdkError::Tool { .. } => "TOOL_ERROR",
            AdkError::Model { .. } => "MODEL_ERROR",
            AdkError::Graph(_) => "INVALID_GRAPH",
            AdkError::NoRoute { .. } => "NO_ROUTE",
            AdkError::SessionNotFound(_) => "SESSION_NOT_FOUND",
            AdkError::ArtifactNotFound(_) => "ARTIFACT_NOT_FOUND",
            AdkError::Validation { .. } => "VALIDATION_ERROR",
            AdkError::Storage { .. } => "STORAGE_ERROR",
            AdkError::Config(_) => "CONFIG_ERROR",
            AdkError::LimitExceeded(_) => "LIMIT_EXCEEDED",
            AdkError::Cancelled => "CANCELLED",
            AdkError::Serde(_) => "SERDE_ERROR",
            AdkError::Io(_) => "IO_ERROR",
            AdkError::Other(_) => "ERROR",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_flow_errors_are_never_retried() {
        let e = AdkError::NodeInterrupted {
            interrupt_id: "i1".into(),
        };
        assert!(e.is_control_flow());
        assert!(!e.is_retryable());
    }

    #[test]
    fn model_errors_are_retryable() {
        assert!(AdkError::model("gemini", "503").is_retryable());
    }

    #[test]
    fn validation_errors_are_not_retryable() {
        assert!(!AdkError::validation("city", "required").is_retryable());
    }

    #[test]
    fn storage_errors_carry_their_own_retryability() {
        assert!(AdkError::storage_retryable("database is locked").is_retryable());
        assert!(!AdkError::storage("UNIQUE constraint failed").is_retryable());
    }
}
