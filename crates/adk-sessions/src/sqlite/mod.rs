//! Durable [`SessionService`](adk_core::SessionService),
//! [`ArtifactService`](adk_core::ArtifactService), and
//! [`MemoryService`](adk_core::MemoryService) backends, on SQLite.
//!
//! The in-memory services lose everything at process exit, which rules out the
//! things these exist to support: a conversation that spans restarts, a
//! human-in-the-loop run that suspends now and resumes tomorrow, a generated
//! file a user comes back for, something a user said last week. The session and
//! artifact backends keep the in-memory semantics exactly — the prefix routing,
//! the hydration, the refusal to record partial events, the per-filename
//! versioning — and write them to a file.
//!
//! [`SqliteMemoryService`] is the exception, and says why on its own docs: the
//! in-memory recall is a self-declared placeholder rather than a contract, so
//! this one uses FTS5 instead of reproducing it.
//!
//! # Storage layout
//!
//! | Table | Holds |
//! |---|---|
//! | `sessions` | one row per thread, with its last-update time |
//! | `events` | the history, one row per event, ordered by `seq` |
//! | `session_state` | unprefixed keys, scoped to one thread |
//! | `user_state` | `user:` keys, shared across one user's threads |
//! | `app_state` | `app:` keys, shared by every user of an app |
//! | `artifacts` | one row per filename *and version* |
//!
//! `temp:` keys have no table: they are dropped on the way in, which is what
//! makes them temporary.
//!
//! Events, artifact payloads, and memory entries are stored as serialized JSON
//! rather than as columns per field. That is deliberate. ADK 2.0 added `node_info` and `output`
//! to the event schema, and a column-per-field store would have needed a
//! migration to carry them; a JSON payload round-trips new fields with no schema
//! change at all. [`SCHEMA_VERSION`] still tracks changes that do need one.
//!
//! # Concurrency
//!
//! [`rusqlite`] is a blocking library, so every query runs on
//! [`tokio::task::spawn_blocking`] with the connection lock taken *inside* the
//! blocking closure — it is never held across an await point. The database is
//! opened in WAL mode with a busy timeout so a second process reading the same
//! file does not immediately fail.
//!
//! # Example
//!
//! One database, all three services:
//!
//! ```
//! # tokio_test::block_on(async {
//! use adk_core::{ArtifactService, Event, MemoryService, Part, SessionService};
//! use adk_sessions::SqliteStore;
//!
//! let store = SqliteStore::in_memory().await?;
//! let sessions = store.sessions();
//! let artifacts = store.artifacts();
//!
//! let mut session = sessions.create_session("app", "u1", None, None).await?;
//!
//! let mut event = Event::new("inv-1", "agent").with_text("here is your report");
//! event.actions.set_state("user:report_count", 1);
//! sessions.append_event(&mut session, event).await?;
//!
//! let version = artifacts
//!     .save_artifact("app", "u1", &session.id, "report.md", Part::text("# Q3"))
//!     .await?;
//! assert_eq!(version, 0);
//!
//! // What the conversation said is recallable later, across sessions.
//! store.memories().add_session_to_memory(&session).await?;
//! let hits = store.memories().search_memory("app", "u1", "report").await?;
//! assert_eq!(hits.len(), 1);
//!
//! // `store.services()` bundles all three for a Runner.
//! let services = store.services();
//! assert!(services.artifact.is_some() && services.memory.is_some());
//! # Ok::<(), adk_core::AdkError>(())
//! # }).unwrap();
//! ```

mod artifacts;
mod memory;
mod sessions;

pub use artifacts::SqliteArtifactService;
pub use memory::SqliteMemoryService;
pub use sessions::SqliteSessionService;

use adk_core::{AdkError, Result, Services};
use rusqlite::{Connection, ErrorCode};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The schema this build knows how to read and write.
///
/// Recorded in SQLite's `user_version`. Opening a file written by a newer build
/// fails rather than misreading it; an older file is migrated forward.
pub const SCHEMA_VERSION: i64 = MIGRATIONS.len() as i64;

/// How long to wait for a lock held by another connection before giving up.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// Schema steps, applied in order from whatever version a file is already at.
const MIGRATIONS: &[&str] = &[V1_SESSIONS, V2_ARTIFACTS, V3_MEMORIES];

const V1_SESSIONS: &str = r#"
CREATE TABLE IF NOT EXISTS sessions (
    app_name         TEXT NOT NULL,
    user_id          TEXT NOT NULL,
    id               TEXT NOT NULL,
    create_time      REAL NOT NULL,
    last_update_time REAL NOT NULL,
    PRIMARY KEY (app_name, user_id, id)
);

CREATE TABLE IF NOT EXISTS events (
    app_name   TEXT    NOT NULL,
    user_id    TEXT    NOT NULL,
    session_id TEXT    NOT NULL,
    seq        INTEGER NOT NULL,
    payload    TEXT    NOT NULL,
    PRIMARY KEY (app_name, user_id, session_id, seq),
    FOREIGN KEY (app_name, user_id, session_id)
        REFERENCES sessions (app_name, user_id, id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS session_state (
    app_name   TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    session_id TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    PRIMARY KEY (app_name, user_id, session_id, key),
    FOREIGN KEY (app_name, user_id, session_id)
        REFERENCES sessions (app_name, user_id, id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_state (
    app_name TEXT NOT NULL,
    user_id  TEXT NOT NULL,
    key      TEXT NOT NULL,
    value    TEXT NOT NULL,
    PRIMARY KEY (app_name, user_id, key)
);

CREATE TABLE IF NOT EXISTS app_state (
    app_name TEXT NOT NULL,
    key      TEXT NOT NULL,
    value    TEXT NOT NULL,
    PRIMARY KEY (app_name, key)
);
"#;

/// Artifacts are keyed by *scope* rather than by session id directly: a
/// `user:`-prefixed filename stores under the empty scope so it is reachable
/// from every one of that user's threads, mirroring the `user:` state prefix.
///
/// There is deliberately no foreign key to `sessions`. Deleting a thread does
/// not delete the files it produced — the in-memory service behaves the same
/// way, and a report outliving the conversation that generated it is the point.
const V2_ARTIFACTS: &str = r#"
CREATE TABLE IF NOT EXISTS artifacts (
    app_name TEXT    NOT NULL,
    user_id  TEXT    NOT NULL,
    scope    TEXT    NOT NULL,
    filename TEXT    NOT NULL,
    version  INTEGER NOT NULL,
    payload  TEXT    NOT NULL,
    PRIMARY KEY (app_name, user_id, scope, filename, version)
);
"#;

/// Long-term memory, indexed for search by SQLite's FTS5 extension rather than
/// stored in an ordinary table.
///
/// Everything except `text` is `UNINDEXED`: FTS5 would otherwise tokenize the
/// ids and the serialized entry, so a search for a word that happened to appear
/// inside a JSON payload would match. Only the recalled text is searchable.
///
/// `entry` is a whole serialized `MemoryEntry`, for the same reason events and
/// artifacts are stored as JSON — FTS5 columns are typeless, and reconstructing
/// one value beats reading a timestamp back out of a text column.
const V3_MEMORIES: &str = r#"
CREATE VIRTUAL TABLE IF NOT EXISTS memories USING fts5(
    app_name   UNINDEXED,
    user_id    UNINDEXED,
    session_id UNINDEXED,
    entry      UNINDEXED,
    text
);
"#;

/// Turns a `rusqlite` failure into an [`AdkError`], preserving whether a retry
/// could plausibly succeed.
fn map_sql_error(err: rusqlite::Error) -> AdkError {
    let transient = matches!(
        err.sqlite_error_code(),
        Some(ErrorCode::DatabaseBusy) | Some(ErrorCode::DatabaseLocked)
    );
    if transient {
        AdkError::storage_retryable(err.to_string())
    } else {
        AdkError::storage(err.to_string())
    }
}

/// Lets a `rusqlite` result join an ADK `?` chain.
pub(crate) trait SqlExt<T> {
    fn sql(self) -> Result<T>;
}

impl<T> SqlExt<T> for std::result::Result<T, rusqlite::Error> {
    fn sql(self) -> Result<T> {
        self.map_err(map_sql_error)
    }
}

/// Runs a blocking database closure on tokio's blocking pool.
async fn blocking<F, T>(f: F) -> Result<T>
where
    F: FnOnce() -> Result<T> + Send + 'static,
    T: Send + 'static,
{
    match tokio::task::spawn_blocking(f).await {
        Ok(result) => result,
        Err(join) => Err(AdkError::storage(format!(
            "sqlite worker did not finish: {join}"
        ))),
    }
}

/// Applies connection pragmas and brings the schema up to date.
fn prepare(conn: &Connection) -> Result<()> {
    conn.busy_timeout(BUSY_TIMEOUT).sql()?;
    conn.pragma_update(None, "foreign_keys", true).sql()?;
    // WAL lets a reader and a writer work at once. In-memory databases report
    // "memory" instead and that is fine, so the answer is read but not checked.
    let _: String = conn
        .query_row("PRAGMA journal_mode = WAL", [], |row| row.get(0))
        .sql()?;

    let version: i64 = conn
        .pragma_query_value(None, "user_version", |row| row.get(0))
        .sql()?;
    if version > SCHEMA_VERSION {
        return Err(AdkError::storage(format!(
            "database is at schema version {version}, but this build understands \
             at most {SCHEMA_VERSION}"
        )));
    }
    if version < SCHEMA_VERSION {
        for step in &MIGRATIONS[version as usize..] {
            conn.execute_batch(step).sql()?;
        }
        conn.pragma_update(None, "user_version", SCHEMA_VERSION)
            .sql()?;
    }
    Ok(())
}

/// A prepared connection, shared by the services built on it.
///
/// Cloning shares the same connection rather than opening a second one, so two
/// services built from one [`SqliteStore`] see each other's writes immediately
/// and never contend for the file.
#[derive(Debug, Clone)]
pub(crate) struct Db {
    conn: Arc<Mutex<Connection>>,
}

impl Db {
    /// Opens (or creates) a database file.
    pub(crate) async fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        Self::build(move || Connection::open(path)).await
    }

    /// Opens a private in-memory database.
    pub(crate) async fn in_memory() -> Result<Self> {
        Self::build(Connection::open_in_memory).await
    }

    /// Adopts an already-open connection.
    pub(crate) fn adopt(conn: Connection) -> Result<Self> {
        prepare(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    async fn build<F>(open: F) -> Result<Self>
    where
        F: FnOnce() -> std::result::Result<Connection, rusqlite::Error> + Send + 'static,
    {
        blocking(move || {
            let conn = open().sql()?;
            prepare(&conn)?;
            Ok(conn)
        })
        .await
        .map(|conn| Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Runs a database operation on the blocking pool.
    ///
    /// The lock is acquired inside the closure, so it is never held across an
    /// await point.
    pub(crate) async fn with<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&mut Connection) -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let conn = Arc::clone(&self.conn);
        blocking(move || {
            let mut guard = conn.lock().unwrap_or_else(|e| e.into_inner());
            f(&mut guard)
        })
        .await
    }
}

/// One SQLite database serving sessions, artifacts, and memory.
///
/// Open the file once and take whichever services you need; they share the
/// connection, so a session, the artifacts it produced, and what it left in
/// memory all land in the same database and the same transaction log.
///
/// ```
/// # tokio_test::block_on(async {
/// # use adk_sessions::SqliteStore;
/// let store = SqliteStore::in_memory().await?;
/// let services = store.services();
/// # Ok::<(), adk_core::AdkError>(())
/// # }).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SqliteStore {
    db: Db,
}

impl SqliteStore {
    /// Opens (or creates) a database file and applies the schema.
    pub async fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            db: Db::open(path).await?,
        })
    }

    /// Opens a private in-memory database.
    ///
    /// The data lives as long as this store does: the connection is held open,
    /// so unlike a bare `:memory:` connection per query, what is written here is
    /// readable back. Useful in tests that want the SQLite code path without
    /// touching the filesystem.
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

    /// The session service on this database.
    pub fn sessions(&self) -> SqliteSessionService {
        SqliteSessionService::from_db(self.db.clone())
    }

    /// The artifact service on this database.
    pub fn artifacts(&self) -> SqliteArtifactService {
        SqliteArtifactService::from_db(self.db.clone())
    }

    /// The memory service on this database.
    pub fn memories(&self) -> SqliteMemoryService {
        SqliteMemoryService::from_db(self.db.clone())
    }

    /// All three services, bundled for a `Runner`.
    pub fn services(&self) -> Services {
        Services::new(Arc::new(self.sessions()))
            .with_artifact(Arc::new(self.artifacts()))
            .with_memory(Arc::new(self.memories()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_core::{ArtifactService, Content, Event, MemoryService, Part, Session, SessionService};

    #[tokio::test]
    async fn all_three_services_share_one_database() {
        let store = SqliteStore::in_memory().await.unwrap();
        let sessions = store.sessions();
        let artifacts = store.artifacts();

        let session = sessions
            .create_session("app", "u1", None, None)
            .await
            .unwrap();
        artifacts
            .save_artifact("app", "u1", &session.id, "notes.txt", Part::text("hi"))
            .await
            .unwrap();

        let mut remembered = session.clone();
        remembered
            .events
            .push(Event::new("inv", "user").with_content(Content::user_text("about hiking")));
        store
            .memories()
            .add_session_to_memory(&remembered)
            .await
            .unwrap();

        // Services taken separately from the same store see those writes.
        assert!(store
            .artifacts()
            .load_artifact("app", "u1", &session.id, "notes.txt", None)
            .await
            .unwrap()
            .is_some());
        assert_eq!(
            store
                .memories()
                .search_memory("app", "u1", "hiking")
                .await
                .unwrap()
                .len(),
            1
        );
    }

    /// A file written before artifacts and memory existed catches up on open,
    /// rather than being refused or quietly missing its newer tables.
    #[tokio::test]
    async fn a_v1_database_migrates_all_the_way_forward() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(V1_SESSIONS).unwrap();
        conn.pragma_update(None, "user_version", 1).unwrap();

        let store = SqliteStore::from_connection(conn).unwrap();
        let mut session = store
            .sessions()
            .create_session("app", "u1", None, None)
            .await
            .unwrap();

        // Both later tables exist because the migrations ran, not because a
        // fresh database was created.
        store
            .artifacts()
            .save_artifact("app", "u1", &session.id, "f.txt", Part::text("x"))
            .await
            .unwrap();
        session
            .events
            .push(Event::new("inv", "user").with_content(Content::user_text("about hiking")));
        store
            .memories()
            .add_session_to_memory(&session)
            .await
            .unwrap();
        assert_eq!(
            store
                .memories()
                .search_memory("app", "u1", "hiking")
                .await
                .unwrap()
                .len(),
            1
        );
    }

    /// The step that matters for anyone already running the previous release:
    /// v2 on disk, memory added on top.
    #[tokio::test]
    async fn a_v2_database_gains_the_memory_table() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(V1_SESSIONS).unwrap();
        conn.execute_batch(V2_ARTIFACTS).unwrap();
        conn.pragma_update(None, "user_version", 2).unwrap();

        let store = SqliteStore::from_connection(conn).unwrap();
        let mut session = Session::new("s1", "app", "u1");
        session
            .events
            .push(Event::new("inv", "user").with_content(Content::user_text("about hiking")));
        store
            .memories()
            .add_session_to_memory(&session)
            .await
            .unwrap();
        assert_eq!(
            store
                .memories()
                .search_memory("app", "u1", "hiking")
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn a_newer_schema_version_is_refused() {
        let conn = Connection::open_in_memory().unwrap();
        conn.pragma_update(None, "user_version", SCHEMA_VERSION + 1)
            .unwrap();
        let err = SqliteStore::from_connection(conn).unwrap_err();
        assert!(matches!(err, AdkError::Storage { .. }), "{err}");
    }
}
