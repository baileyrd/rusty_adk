//! The SQLite-backed [`SessionService`].

use adk_core::{AdkError, Event, Result, Session, SessionService, State, StateScope};
use async_trait::async_trait;
use rusqlite::{params, Connection, OptionalExtension, Statement, ToSql};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

use super::{Db, SqlExt};

/// A [`SessionService`] that stores threads, history, and scoped state in a
/// SQLite database.
///
/// Behaviourally identical to
/// [`InMemorySessionService`](crate::InMemorySessionService): the same prefix
/// routing, the same hydration of `app:` and `user:` values onto each thread,
/// the same refusal to record partial events. See the
/// [module documentation](super) for the storage layout and the concurrency
/// model.
///
/// Build one with [`SqliteSessionService::open`], or take it from a
/// [`SqliteStore`](super::SqliteStore) to share a database with the artifact
/// service.
#[derive(Debug, Clone)]
pub struct SqliteSessionService {
    db: Db,
}

impl SqliteSessionService {
    /// Opens (or creates) a database file and applies the schema.
    pub async fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            db: Db::open(path).await?,
        })
    }

    /// Opens a private in-memory database.
    ///
    /// The data lives as long as this service does: the connection is held
    /// open, so unlike a bare `:memory:` connection per query, sessions written
    /// here are readable back. Useful in tests that want the SQLite code path
    /// without touching the filesystem.
    pub async fn in_memory() -> Result<Self> {
        Ok(Self {
            db: Db::in_memory().await?,
        })
    }

    /// Wraps an already-open connection, applying pragmas and the schema.
    ///
    /// Use this to hand in a connection configured elsewhere — an encrypted
    /// database, a custom VFS, a shared cache.
    pub fn from_connection(conn: Connection) -> Result<Self> {
        Ok(Self {
            db: Db::adopt(conn)?,
        })
    }

    pub(super) fn from_db(db: Db) -> Self {
        Self { db }
    }
}

/// Collects a `(key, value)` query's rows into `merged`, decoding the JSON.
fn collect_state(
    merged: &mut BTreeMap<String, Value>,
    stmt: &mut Statement<'_>,
    args: &[&dyn ToSql],
) -> Result<()> {
    let mut rows = stmt.query(args).sql()?;
    while let Some(row) = rows.next().sql()? {
        let key: String = row.get(0).sql()?;
        let raw: String = row.get(1).sql()?;
        merged.insert(key, serde_json::from_str(&raw)?);
    }
    Ok(())
}

/// Reassembles the flat state view an agent sees, from the three scope tables.
///
/// Order matters only for determinism — the scopes are disjoint by prefix — but
/// it mirrors the in-memory service so both behave identically.
fn read_state(
    conn: &Connection,
    app_name: &str,
    user_id: &str,
    session_id: &str,
) -> Result<BTreeMap<String, Value>> {
    let mut merged = BTreeMap::new();

    let mut app = conn
        .prepare_cached("SELECT key, value FROM app_state WHERE app_name = ?1")
        .sql()?;
    collect_state(&mut merged, &mut app, &[&app_name])?;

    let mut user = conn
        .prepare_cached("SELECT key, value FROM user_state WHERE app_name = ?1 AND user_id = ?2")
        .sql()?;
    collect_state(&mut merged, &mut user, &[&app_name, &user_id])?;

    let mut own = conn
        .prepare_cached(
            "SELECT key, value FROM session_state \
             WHERE app_name = ?1 AND user_id = ?2 AND session_id = ?3",
        )
        .sql()?;
    collect_state(&mut merged, &mut own, &[&app_name, &user_id, &session_id])?;

    Ok(merged)
}

/// Writes a state delta, sending each key to the table its prefix names.
///
/// `temp:` keys are dropped here: they stay on the event for observability but
/// never reach storage.
fn write_state(
    conn: &Connection,
    app_name: &str,
    user_id: &str,
    session_id: &str,
    delta: &BTreeMap<String, Value>,
) -> Result<()> {
    for (key, value) in delta {
        let encoded = serde_json::to_string(value)?;
        match StateScope::of(key) {
            StateScope::App => {
                conn.prepare_cached(
                    "INSERT INTO app_state (app_name, key, value) VALUES (?1, ?2, ?3) \
                     ON CONFLICT (app_name, key) DO UPDATE SET value = excluded.value",
                )
                .sql()?
                .execute(params![app_name, key, encoded])
                .sql()?;
            }
            StateScope::User => {
                conn.prepare_cached(
                    "INSERT INTO user_state (app_name, user_id, key, value) \
                     VALUES (?1, ?2, ?3, ?4) \
                     ON CONFLICT (app_name, user_id, key) DO UPDATE SET value = excluded.value",
                )
                .sql()?
                .execute(params![app_name, user_id, key, encoded])
                .sql()?;
            }
            StateScope::Temp => {}
            StateScope::Session => {
                conn.prepare_cached(
                    "INSERT INTO session_state (app_name, user_id, session_id, key, value) \
                     VALUES (?1, ?2, ?3, ?4, ?5) \
                     ON CONFLICT (app_name, user_id, session_id, key) \
                     DO UPDATE SET value = excluded.value",
                )
                .sql()?
                .execute(params![app_name, user_id, session_id, key, encoded])
                .sql()?;
            }
        }
    }
    Ok(())
}

/// Reads a thread's history, oldest first.
fn read_events(
    conn: &Connection,
    app_name: &str,
    user_id: &str,
    session_id: &str,
) -> Result<Vec<Event>> {
    let mut stmt = conn
        .prepare_cached(
            "SELECT payload FROM events \
             WHERE app_name = ?1 AND user_id = ?2 AND session_id = ?3 ORDER BY seq",
        )
        .sql()?;
    let mut rows = stmt.query(params![app_name, user_id, session_id]).sql()?;
    let mut events = Vec::new();
    while let Some(row) = rows.next().sql()? {
        let raw: String = row.get(0).sql()?;
        events.push(serde_json::from_str(&raw)?);
    }
    Ok(events)
}

/// Loads one thread, optionally without its history.
fn load_session(
    conn: &Connection,
    app_name: &str,
    user_id: &str,
    session_id: &str,
    with_events: bool,
) -> Result<Option<Session>> {
    let last_update_time: Option<f64> = conn
        .prepare_cached(
            "SELECT last_update_time FROM sessions \
             WHERE app_name = ?1 AND user_id = ?2 AND id = ?3",
        )
        .sql()?
        .query_row(params![app_name, user_id, session_id], |row| row.get(0))
        .optional()
        .sql()?;
    let Some(last_update_time) = last_update_time else {
        return Ok(None);
    };

    let mut session = Session::new(session_id, app_name, user_id);
    session.last_update_time = last_update_time;
    session.state = State::from_map(read_state(conn, app_name, user_id, session_id)?);
    if with_events {
        session.events = read_events(conn, app_name, user_id, session_id)?;
    }
    Ok(Some(session))
}

#[async_trait]
impl SessionService for SqliteSessionService {
    async fn create_session(
        &self,
        app_name: &str,
        user_id: &str,
        state: Option<State>,
        session_id: Option<String>,
    ) -> Result<Session> {
        let id = session_id.unwrap_or_else(|| adk_core::new_id("session"));
        let initial = state.map(|s| s.to_map()).unwrap_or_default();
        let (app_name, user_id) = (app_name.to_string(), user_id.to_string());

        self.db
            .with(move |conn| {
                let tx = conn.transaction().sql()?;
                let now = adk_core::now_seconds();

                // Reusing an id starts the thread over, as the in-memory service
                // does; the cascade clears whatever the old thread left behind.
                tx.execute(
                    "DELETE FROM sessions WHERE app_name = ?1 AND user_id = ?2 AND id = ?3",
                    params![app_name, user_id, id],
                )
                .sql()?;
                tx.execute(
                    "INSERT INTO sessions (app_name, user_id, id, create_time, last_update_time) \
                 VALUES (?1, ?2, ?3, ?4, ?4)",
                    params![app_name, user_id, id, now],
                )
                .sql()?;

                // Starting state may carry scoped keys; route them before storing.
                write_state(&tx, &app_name, &user_id, &id, &initial)?;

                let mut session = Session::new(&id, &app_name, &user_id);
                session.last_update_time = now;
                session.state = State::from_map(read_state(&tx, &app_name, &user_id, &id)?);
                tx.commit().sql()?;
                Ok(session)
            })
            .await
    }

    async fn get_session(
        &self,
        app_name: &str,
        user_id: &str,
        session_id: &str,
    ) -> Result<Option<Session>> {
        let (app_name, user_id, session_id) = (
            app_name.to_string(),
            user_id.to_string(),
            session_id.to_string(),
        );
        self.db
            .with(move |conn| load_session(conn, &app_name, &user_id, &session_id, true))
            .await
    }

    async fn list_sessions(&self, app_name: &str, user_id: &str) -> Result<Vec<Session>> {
        let (app_name, user_id) = (app_name.to_string(), user_id.to_string());
        self.db
            .with(move |conn| {
                let ids: Vec<String> = {
                    let mut stmt = conn
                    .prepare_cached(
                        "SELECT id FROM sessions WHERE app_name = ?1 AND user_id = ?2 ORDER BY id",
                    )
                    .sql()?;
                    let mut rows = stmt.query(params![app_name, user_id]).sql()?;
                    let mut ids = Vec::new();
                    while let Some(row) = rows.next().sql()? {
                        ids.push(row.get(0).sql()?);
                    }
                    ids
                };

                // Listings omit history: callers use this to pick a thread, and
                // materializing every event would be wasteful.
                let mut sessions = Vec::with_capacity(ids.len());
                for id in ids {
                    if let Some(session) = load_session(conn, &app_name, &user_id, &id, false)? {
                        sessions.push(session);
                    }
                }
                Ok(sessions)
            })
            .await
    }

    async fn delete_session(&self, app_name: &str, user_id: &str, session_id: &str) -> Result<()> {
        let (app_name, user_id, session_id) = (
            app_name.to_string(),
            user_id.to_string(),
            session_id.to_string(),
        );
        self.db
            .with(move |conn| {
                let removed = conn
                    .execute(
                        "DELETE FROM sessions WHERE app_name = ?1 AND user_id = ?2 AND id = ?3",
                        params![app_name, user_id, session_id],
                    )
                    .sql()?;
                if removed == 0 {
                    return Err(AdkError::SessionNotFound(session_id));
                }
                Ok(())
            })
            .await
    }

    async fn append_event(&self, session: &mut Session, event: Event) -> Result<()> {
        // A partial event is a streaming chunk. Forward it, but do not commit
        // its actions or record it — the final aggregated event carries both.
        if event.is_partial() {
            return Ok(());
        }

        let app_name = session.app_name.clone();
        let user_id = session.user_id.clone();
        let id = session.id.clone();
        // How much history the caller's handle already holds. If the stored
        // thread is exactly that long, the handle is current and appending in
        // place is enough — no need to deserialize the whole transcript again.
        let known = session.events.len();
        let recorded = event.clone();

        let (events, state, last_update_time) = self
            .db
            .with(move |conn| {
                let tx = conn.transaction().sql()?;

                let exists: Option<i64> = tx
                    .prepare_cached(
                        "SELECT 1 FROM sessions WHERE app_name = ?1 AND user_id = ?2 AND id = ?3",
                    )
                    .sql()?
                    .query_row(params![app_name, user_id, id], |row| row.get(0))
                    .optional()
                    .sql()?;
                if exists.is_none() {
                    return Err(AdkError::SessionNotFound(id));
                }

                write_state(&tx, &app_name, &user_id, &id, &event.actions.state_delta)?;

                let seq: i64 = tx
                    .prepare_cached(
                        "SELECT COALESCE(MAX(seq), -1) + 1 FROM events \
                         WHERE app_name = ?1 AND user_id = ?2 AND session_id = ?3",
                    )
                    .sql()?
                    .query_row(params![app_name, user_id, id], |row| row.get(0))
                    .sql()?;
                tx.execute(
                    "INSERT INTO events (app_name, user_id, session_id, seq, payload) \
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![app_name, user_id, id, seq, serde_json::to_string(&event)?],
                )
                .sql()?;

                let now = adk_core::now_seconds();
                tx.execute(
                    "UPDATE sessions SET last_update_time = ?4 \
                     WHERE app_name = ?1 AND user_id = ?2 AND id = ?3",
                    params![app_name, user_id, id, now],
                )
                .sql()?;

                let events = if seq as usize == known {
                    None
                } else {
                    Some(read_events(&tx, &app_name, &user_id, &id)?)
                };
                let state = read_state(&tx, &app_name, &user_id, &id)?;
                tx.commit().sql()?;
                Ok((events, state, now))
            })
            .await?;

        // Reflect the committed result back into the caller's handle, so code
        // resuming after the yield observes persisted state.
        match events {
            Some(all) => session.events = all,
            None => session.events.push(recorded),
        }
        session.state = State::from_map(state);
        session.last_update_time = last_update_time;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    async fn service() -> SqliteSessionService {
        SqliteSessionService::in_memory().await.unwrap()
    }

    /// A unique path under the system temp directory, removed by the caller.
    fn temp_db() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("{}.sqlite3", adk_core::new_id("adk-test")))
    }

    #[tokio::test]
    async fn create_and_get_round_trip() {
        let svc = service().await;
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
        // Another user's thread is not reachable under the same id.
        assert!(svc.get_session("app", "u2", "s1").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn append_event_commits_session_state_and_history() {
        let svc = service().await;
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
        let svc = service().await;
        let mut session = svc.create_session("app", "u1", None, None).await.unwrap();

        let mut event = Event::new("inv", "agent");
        event.actions.set_state("temp:scratch", 42);
        svc.append_event(&mut session, event).await.unwrap();

        assert!(session.state.get("temp:scratch").is_none());
        let reloaded = svc
            .get_session("app", "u1", &session.id)
            .await
            .unwrap()
            .unwrap();
        assert!(reloaded.state.get("temp:scratch").is_none());
    }

    #[tokio::test]
    async fn user_state_is_shared_across_that_users_sessions() {
        let svc = service().await;
        let mut first = svc.create_session("app", "u1", None, None).await.unwrap();

        let mut event = Event::new("inv", "agent");
        event.actions.set_state("user:lang", "fr");
        svc.append_event(&mut first, event).await.unwrap();

        let second = svc.create_session("app", "u1", None, None).await.unwrap();
        assert_eq!(second.state.get("user:lang").unwrap(), &json!("fr"));

        let other = svc.create_session("app", "u2", None, None).await.unwrap();
        assert!(other.state.get("user:lang").is_none());
    }

    #[tokio::test]
    async fn app_state_is_shared_across_users() {
        let svc = service().await;
        let mut first = svc.create_session("app", "u1", None, None).await.unwrap();

        let mut event = Event::new("inv", "agent");
        event.actions.set_state("app:discount", "SUMMER");
        svc.append_event(&mut first, event).await.unwrap();

        let other_user = svc.create_session("app", "u2", None, None).await.unwrap();
        assert_eq!(
            other_user.state.get("app:discount").unwrap(),
            &json!("SUMMER")
        );

        let other_app = svc.create_session("other", "u1", None, None).await.unwrap();
        assert!(other_app.state.get("app:discount").is_none());
    }

    #[tokio::test]
    async fn initial_state_is_routed_by_prefix() {
        let svc = service().await;
        let mut state = State::new();
        state.set("app:tier", "gold");
        state.set("user:lang", "de");
        state.set("temp:scratch", 1);
        state.set("step", "start");

        let created = svc
            .create_session("app", "u1", Some(state), None)
            .await
            .unwrap();
        assert!(created.state.get("temp:scratch").is_none());

        // The scoped keys landed in their own tables, so a sibling thread and
        // another user of the same app see them.
        let sibling = svc.create_session("app", "u1", None, None).await.unwrap();
        assert_eq!(sibling.state.get("app:tier").unwrap(), &json!("gold"));
        assert_eq!(sibling.state.get("user:lang").unwrap(), &json!("de"));
        // ...but the unprefixed key belongs to the thread that set it.
        assert!(sibling.state.get("step").is_none());
        assert_eq!(created.state.get("step").unwrap(), &json!("start"));
    }

    #[tokio::test]
    async fn partial_events_are_not_recorded() {
        let svc = service().await;
        let mut session = svc.create_session("app", "u1", None, None).await.unwrap();

        let mut chunk = Event::new("inv", "agent").with_text("Par").as_partial();
        chunk.actions.set_state("should_not", "commit");
        svc.append_event(&mut session, chunk).await.unwrap();

        assert!(session.events.is_empty());
        assert!(session.state.get("should_not").is_none());
        let reloaded = svc
            .get_session("app", "u1", &session.id)
            .await
            .unwrap()
            .unwrap();
        assert!(reloaded.events.is_empty());
    }

    #[tokio::test]
    async fn list_sessions_omits_history_and_filters_by_user() {
        let svc = service().await;
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
        let svc = service().await;
        let session = svc.create_session("app", "u1", None, None).await.unwrap();
        assert!(svc.delete_session("app", "u1", &session.id).await.is_ok());
        assert!(svc.delete_session("app", "u1", &session.id).await.is_err());
    }

    #[tokio::test]
    async fn deleting_a_thread_takes_its_history_and_own_state() {
        let svc = service().await;
        let mut session = svc.create_session("app", "u1", None, None).await.unwrap();
        let mut event = Event::new("inv", "agent").with_text("x");
        event.actions.set_state("step", "one");
        event.actions.set_state("user:lang", "fr");
        svc.append_event(&mut session, event).await.unwrap();

        svc.delete_session("app", "u1", &session.id).await.unwrap();

        // The thread and its own state are gone...
        let reborn = svc
            .create_session("app", "u1", None, Some(session.id.clone()))
            .await
            .unwrap();
        assert!(reborn.events.is_empty());
        assert!(reborn.state.get("step").is_none());
        // ...but user-scoped state outlives any single thread.
        assert_eq!(reborn.state.get("user:lang").unwrap(), &json!("fr"));
    }

    #[tokio::test]
    async fn appending_to_an_unknown_thread_is_an_error() {
        let svc = service().await;
        let mut orphan = Session::new("ghost", "app", "u1");
        assert!(svc
            .append_event(&mut orphan, Event::new("inv", "agent").with_text("x"))
            .await
            .is_err());
    }

    #[tokio::test]
    async fn a_stale_handle_is_refreshed_from_storage() {
        let svc = service().await;
        let mut session = svc.create_session("app", "u1", None, None).await.unwrap();
        svc.append_event(&mut session, Event::new("inv", "agent").with_text("one"))
            .await
            .unwrap();

        // A second handle that never saw the first event still ends up with the
        // full history once it appends.
        let mut stale = svc
            .get_session("app", "u1", &session.id)
            .await
            .unwrap()
            .unwrap();
        stale.events.clear();
        svc.append_event(&mut stale, Event::new("inv", "agent").with_text("two"))
            .await
            .unwrap();

        assert_eq!(stale.events.len(), 2);
        assert_eq!(stale.events[0].text(), "one");
        assert_eq!(stale.events[1].text(), "two");
    }

    #[tokio::test]
    async fn history_survives_reopening_the_file() {
        let path = temp_db();
        let session_id = {
            let svc = SqliteSessionService::open(&path).await.unwrap();
            let mut session = svc.create_session("app", "u1", None, None).await.unwrap();
            for text in ["one", "two", "three"] {
                let mut event = Event::new("inv", "agent").with_text(text);
                event.actions.set_state("last", text);
                svc.append_event(&mut session, event).await.unwrap();
            }
            session.id
        };

        // A fresh process would do exactly this: open the same file and read.
        let svc = SqliteSessionService::open(&path).await.unwrap();
        let session = svc
            .get_session("app", "u1", &session_id)
            .await
            .unwrap()
            .unwrap();
        let texts: Vec<String> = session.events.iter().map(|e| e.text()).collect();
        assert_eq!(texts, vec!["one", "two", "three"]);
        assert_eq!(session.state.get("last").unwrap(), &json!("three"));

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn adk_2_0_event_fields_round_trip() {
        let svc = service().await;
        let mut session = svc.create_session("app", "u1", None, None).await.unwrap();

        // `node_info` and `output` are what ADK 2.0 added to the event schema;
        // storing the event as JSON means they survive with no migration.
        let event = Event::new("inv", "planner")
            .with_text("done")
            .with_node_info(
                adk_core::NodeInfo::new("planner")
                    .with_type("agent")
                    .with_step(3)
                    .with_predecessor("triage"),
            )
            .with_output(json!({"plan": ["a", "b"]}));
        svc.append_event(&mut session, event).await.unwrap();

        let reloaded = svc
            .get_session("app", "u1", &session.id)
            .await
            .unwrap()
            .unwrap();
        let stored = &reloaded.events[0];
        let node_info = stored.node_info.as_ref().unwrap();
        assert_eq!(node_info.name, "planner");
        assert_eq!(node_info.step, Some(3));
        assert_eq!(node_info.predecessor.as_deref(), Some("triage"));
        assert_eq!(stored.output.as_ref().unwrap()["plan"][1], json!("b"));
    }

    /// Storing events as JSON is only safe if JSON is lossless for them. The
    /// `timestamp` is the field that tests it: an `f64` needs exact float
    /// parsing to come back bit-for-bit.
    #[tokio::test]
    async fn a_stored_event_comes_back_identical() {
        let svc = service().await;
        let mut session = svc.create_session("app", "u1", None, None).await.unwrap();

        let mut event = Event::new("inv", "agent").with_text("hello");
        event.actions.set_state("step", "greeted");
        let original = event.clone();
        svc.append_event(&mut session, event).await.unwrap();

        let reloaded = svc
            .get_session("app", "u1", &session.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(reloaded.events[0], original);
    }

    #[tokio::test]
    async fn a_reused_id_starts_the_thread_over() {
        let svc = service().await;
        let mut session = svc
            .create_session("app", "u1", None, Some("s1".into()))
            .await
            .unwrap();
        let mut event = Event::new("inv", "agent").with_text("x");
        event.actions.set_state("step", "one");
        svc.append_event(&mut session, event).await.unwrap();

        let reborn = svc
            .create_session("app", "u1", None, Some("s1".into()))
            .await
            .unwrap();
        assert!(reborn.events.is_empty());
        assert!(reborn.state.get("step").is_none());
    }
}
