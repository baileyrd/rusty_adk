//! Session, artifact, and memory backends for the Rust ADK.
//!
//! The traits themselves live in [`adk_core::services`] so that every crate
//! can depend on the abstraction without pulling in a backend. This crate
//! provides the implementations:
//!
//! - [`InMemorySessionService`], [`InMemoryArtifactService`], and
//!   [`InMemoryMemoryService`] keep everything in process memory. They are the
//!   right default for tests and for a single-run script.
//! - `SqliteSessionService` and `SqliteArtifactService` (feature `sqlite`) write
//!   threads, history, scoped state, and versioned artifacts to a database file,
//!   so a conversation — or a suspended human-in-the-loop run, or a generated
//!   report — survives a restart. They reproduce the in-memory semantics
//!   exactly; `SqliteStore` opens one database and hands out both, and the
//!   `sqlite` module documents the storage layout.
//!
//! # Example
//!
//! ```
//! # tokio_test::block_on(async {
//! use adk_core::{Event, SessionService};
//! use adk_sessions::InMemorySessionService;
//!
//! let service = InMemorySessionService::new();
//! let mut session = service.create_session("app", "u1", None, None).await.unwrap();
//!
//! let mut event = Event::new("inv-1", "agent").with_text("hi");
//! event.actions.set_state("user:login_count", 5);
//! event.actions.set_state("temp:scratch", 1);
//! service.append_event(&mut session, event).await.unwrap();
//!
//! // The durable key persisted; the temporary one did not.
//! assert_eq!(session.state.get("user:login_count").unwrap(), 5);
//! assert!(session.state.get("temp:scratch").is_none());
//! # });
//! ```

#![deny(missing_docs)]
#![warn(clippy::all)]

pub mod in_memory;
#[cfg(feature = "sqlite")]
pub mod sqlite;

pub use in_memory::{
    memory_entry, InMemoryArtifactService, InMemoryMemoryService, InMemorySessionService,
};

#[cfg(feature = "sqlite")]
pub use sqlite::{SqliteArtifactService, SqliteSessionService, SqliteStore};

#[cfg(test)]
mod tests {
    use super::*;
    use adk_core::{ArtifactService, Blob, Content, Event, MemoryService, Part, SessionService};
    use serde_json::json;

    #[tokio::test]
    async fn create_and_get_round_trip() {
        let svc = InMemorySessionService::new();
        let created = svc
            .create_session("app", "u1", None, Some("s1".into()))
            .await
            .unwrap();
        assert_eq!(created.id, "s1");

        let fetched = svc.get_session("app", "u1", "s1").await.unwrap().unwrap();
        assert_eq!(fetched.id, "s1");
        assert!(svc
            .get_session("app", "u1", "nope")
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn append_event_commits_session_state_and_history() {
        let svc = InMemorySessionService::new();
        let mut session = svc.create_session("app", "u1", None, None).await.unwrap();

        let mut event = Event::new("inv", "agent").with_text("hello");
        event.actions.set_state("step", "greeted");
        svc.append_event(&mut session, event).await.unwrap();

        assert_eq!(session.events.len(), 1);
        assert_eq!(session.state.get("step").unwrap(), &json!("greeted"));

        let reloaded = svc
            .get_session("app", "u1", &session.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(reloaded.events.len(), 1);
        assert_eq!(reloaded.state.get("step").unwrap(), &json!("greeted"));
    }

    #[tokio::test]
    async fn temp_keys_never_persist() {
        let svc = InMemorySessionService::new();
        let mut session = svc.create_session("app", "u1", None, None).await.unwrap();

        let mut event = Event::new("inv", "agent");
        event.actions.set_state("temp:scratch", 42);
        svc.append_event(&mut session, event).await.unwrap();

        assert!(session.state.get("temp:scratch").is_none());
    }

    #[tokio::test]
    async fn user_state_is_shared_across_that_users_sessions() {
        let svc = InMemorySessionService::new();
        let mut first = svc.create_session("app", "u1", None, None).await.unwrap();

        let mut event = Event::new("inv", "agent");
        event.actions.set_state("user:lang", "fr");
        svc.append_event(&mut first, event).await.unwrap();

        // A brand-new session for the same user sees it...
        let second = svc.create_session("app", "u1", None, None).await.unwrap();
        assert_eq!(second.state.get("user:lang").unwrap(), &json!("fr"));

        // ...but a different user does not.
        let other = svc.create_session("app", "u2", None, None).await.unwrap();
        assert!(other.state.get("user:lang").is_none());
    }

    #[tokio::test]
    async fn app_state_is_shared_across_users() {
        let svc = InMemorySessionService::new();
        let mut first = svc.create_session("app", "u1", None, None).await.unwrap();

        let mut event = Event::new("inv", "agent");
        event.actions.set_state("app:discount", "SUMMER");
        svc.append_event(&mut first, event).await.unwrap();

        let other_user = svc.create_session("app", "u2", None, None).await.unwrap();
        assert_eq!(
            other_user.state.get("app:discount").unwrap(),
            &json!("SUMMER")
        );

        // A different app is isolated.
        let other_app = svc.create_session("other", "u1", None, None).await.unwrap();
        assert!(other_app.state.get("app:discount").is_none());
    }

    #[tokio::test]
    async fn partial_events_are_not_recorded() {
        let svc = InMemorySessionService::new();
        let mut session = svc.create_session("app", "u1", None, None).await.unwrap();

        let mut chunk = Event::new("inv", "agent").with_text("Par").as_partial();
        chunk.actions.set_state("should_not", "commit");
        svc.append_event(&mut session, chunk).await.unwrap();

        assert!(session.events.is_empty());
        assert!(session.state.get("should_not").is_none());
    }

    #[tokio::test]
    async fn list_sessions_omits_history_and_filters_by_user() {
        let svc = InMemorySessionService::new();
        let mut s1 = svc.create_session("app", "u1", None, None).await.unwrap();
        svc.create_session("app", "u1", None, None).await.unwrap();
        svc.create_session("app", "u2", None, None).await.unwrap();
        svc.append_event(&mut s1, Event::new("inv", "agent").with_text("x"))
            .await
            .unwrap();

        let listed = svc.list_sessions("app", "u1").await.unwrap();
        assert_eq!(listed.len(), 2);
        assert!(listed.iter().all(|s| s.events.is_empty()));
    }

    #[tokio::test]
    async fn delete_session_reports_a_missing_thread() {
        let svc = InMemorySessionService::new();
        let session = svc.create_session("app", "u1", None, None).await.unwrap();
        assert!(svc.delete_session("app", "u1", &session.id).await.is_ok());
        assert!(svc.delete_session("app", "u1", &session.id).await.is_err());
    }

    #[tokio::test]
    async fn artifacts_are_versioned_per_filename() {
        let svc = InMemoryArtifactService::new();
        let blob = |d: &str| {
            Part::InlineData(Blob {
                mime_type: "text/plain".into(),
                data: d.to_string(),
            })
        };

        assert_eq!(
            svc.save_artifact("app", "u1", "s1", "notes.txt", blob("v0"))
                .await
                .unwrap(),
            0
        );
        assert_eq!(
            svc.save_artifact("app", "u1", "s1", "notes.txt", blob("v1"))
                .await
                .unwrap(),
            1
        );

        // Default load returns the latest.
        assert_eq!(
            svc.load_artifact("app", "u1", "s1", "notes.txt", None)
                .await
                .unwrap()
                .unwrap(),
            blob("v1")
        );
        assert_eq!(
            svc.load_artifact("app", "u1", "s1", "notes.txt", Some(0))
                .await
                .unwrap()
                .unwrap(),
            blob("v0")
        );
    }

    #[tokio::test]
    async fn user_scoped_artifacts_cross_sessions() {
        let svc = InMemoryArtifactService::new();
        let part = Part::text("profile");
        svc.save_artifact("app", "u1", "s1", "user:profile", part.clone())
            .await
            .unwrap();

        // Visible from a different session of the same user...
        assert!(svc
            .load_artifact("app", "u1", "s2", "user:profile", None)
            .await
            .unwrap()
            .is_some());
        // ...but a session-scoped one is not.
        svc.save_artifact("app", "u1", "s1", "scratch", part)
            .await
            .unwrap();
        assert!(svc
            .load_artifact("app", "u1", "s2", "scratch", None)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn missing_artifact_loads_as_none() {
        let svc = InMemoryArtifactService::new();
        assert!(svc
            .load_artifact("app", "u1", "s1", "absent", None)
            .await
            .unwrap()
            .is_none());
        assert!(svc
            .delete_artifact("app", "u1", "s1", "absent")
            .await
            .is_err());
    }

    #[tokio::test]
    async fn memory_search_ranks_by_term_overlap() {
        let svc = InMemoryMemoryService::new();
        let mut session = adk_core::Session::new("s1", "app", "u1");
        session.events.push(
            Event::new("inv", "user").with_content(Content::user_text("I love hiking in the Alps")),
        );
        session.events.push(
            Event::new("inv", "user").with_content(Content::user_text("My cat is called Milo")),
        );
        svc.add_session_to_memory(&session).await.unwrap();

        let hits = svc.search_memory("app", "u1", "hiking Alps").await.unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].content.text().contains("Alps"));
        assert_eq!(hits[0].score, Some(1.0));

        assert!(svc
            .search_memory("app", "u1", "quantum")
            .await
            .unwrap()
            .is_empty());
        // Another user's memory is not reachable.
        assert!(svc
            .search_memory("app", "u2", "hiking")
            .await
            .unwrap()
            .is_empty());
    }
}
