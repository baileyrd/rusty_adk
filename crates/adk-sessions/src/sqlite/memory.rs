//! The SQLite-backed [`MemoryService`], on FTS5.

use adk_core::{MemoryEntry, MemoryService, Result, Session};
use async_trait::async_trait;
use rusqlite::params;

use super::{Db, SqlExt};

/// A [`MemoryService`] backed by SQLite's FTS5 full-text index.
///
/// Unlike the session and artifact stores, this is not a faithful port of its
/// in-memory counterpart, and deliberately so.
/// [`InMemoryMemoryService`](crate::InMemoryMemoryService) scores by counting
/// how many query terms appear in an entry, and says of itself that it is
/// "enough to exercise memory-dependent agent logic in tests; swap in a real
/// vector store for production recall". Its scoring is a placeholder, not a
/// contract — so reproducing it here would mean carrying a stand-in into a
/// durable store while ignoring the retrieval engine SQLite already ships.
///
/// What both backends *do* guarantee is the same:
///
/// - one user's memory is invisible to another, and to another app;
/// - partial events and events with no text are never ingested;
/// - results come back best-first, each carrying a score where higher means
///   more relevant.
///
/// The scores themselves are **not comparable between backends**: this one
/// reports BM25 relevance (sign-flipped so higher is better, unbounded above),
/// where the in-memory service reports a `0.0..=1.0` term-overlap fraction. The
/// [`MemoryService`] trait calls the field "backend-specific relevance" for
/// exactly this reason.
///
/// # Re-ingesting a session
///
/// [`add_session_to_memory`](MemoryService::add_session_to_memory) replaces
/// whatever that session contributed before, so ingesting a growing
/// conversation repeatedly — the normal pattern, since a session gains events
/// over time — converges instead of accumulating duplicates. The in-memory
/// service appends unconditionally; duplicates there vanish at exit, and here
/// they would not.
#[derive(Debug, Clone)]
pub struct SqliteMemoryService {
    db: Db,
}

impl SqliteMemoryService {
    /// Opens (or creates) a database file and applies the schema.
    pub async fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Ok(Self {
            db: Db::open(path).await?,
        })
    }

    /// Opens a private in-memory database.
    ///
    /// The data lives as long as this service does. Useful in tests that want
    /// the SQLite code path without touching the filesystem.
    pub async fn in_memory() -> Result<Self> {
        Ok(Self {
            db: Db::in_memory().await?,
        })
    }

    /// Wraps an already-open connection, applying pragmas and the schema.
    pub fn from_connection(conn: rusqlite::Connection) -> Result<Self> {
        Ok(Self {
            db: Db::adopt(conn)?,
        })
    }

    pub(super) fn from_db(db: Db) -> Self {
        Self { db }
    }

    /// Splits text into search terms, exactly as the in-memory service does.
    fn terms(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(str::to_string)
            .collect()
    }

    /// Builds an FTS5 MATCH expression that ORs the query's terms.
    ///
    /// A caller's query is prose, not FTS5 syntax, and handing it over raw
    /// would let `-`, `*`, `NEAR` or an unbalanced quote change the search or
    /// fail it. Terms come out of [`Self::terms`] alphanumeric-only, so quoting
    /// each one makes every token a literal with nothing left to escape.
    fn match_expression(query: &str) -> Option<String> {
        let terms = Self::terms(query);
        if terms.is_empty() {
            return None;
        }
        Some(
            terms
                .iter()
                .map(|t| format!("\"{t}\""))
                .collect::<Vec<_>>()
                .join(" OR "),
        )
    }
}

#[async_trait]
impl MemoryService for SqliteMemoryService {
    async fn add_session_to_memory(&self, session: &Session) -> Result<()> {
        let app_name = session.app_name.clone();
        let user_id = session.user_id.clone();
        let session_id = session.id.clone();

        // Ingest the same events the in-memory service does: skip streaming
        // chunks, and skip anything with no text to recall.
        let mut rows: Vec<(String, String)> = Vec::new();
        for event in &session.events {
            if event.is_partial() {
                continue;
            }
            let Some(content) = &event.content else {
                continue;
            };
            let text = content.text();
            if text.trim().is_empty() {
                continue;
            }
            let entry = MemoryEntry {
                content: content.clone(),
                author: event.author.clone(),
                timestamp: event.timestamp,
                score: None,
            };
            rows.push((serde_json::to_string(&entry)?, text));
        }

        self.db
            .with(move |conn| {
                let tx = conn.transaction().sql()?;
                // Replace this session's contribution rather than adding to it.
                tx.execute(
                    "DELETE FROM memories \
                     WHERE app_name = ?1 AND user_id = ?2 AND session_id = ?3",
                    params![app_name, user_id, session_id],
                )
                .sql()?;
                for (entry, text) in rows {
                    tx.execute(
                        "INSERT INTO memories (app_name, user_id, session_id, entry, text) \
                         VALUES (?1, ?2, ?3, ?4, ?5)",
                        params![app_name, user_id, session_id, entry, text],
                    )
                    .sql()?;
                }
                tx.commit().sql()?;
                Ok(())
            })
            .await
    }

    async fn search_memory(
        &self,
        app_name: &str,
        user_id: &str,
        query: &str,
    ) -> Result<Vec<MemoryEntry>> {
        let Some(expression) = Self::match_expression(query) else {
            return Ok(Vec::new());
        };
        let (app_name, user_id) = (app_name.to_string(), user_id.to_string());

        self.db
            .with(move |conn| {
                let mut stmt = conn
                    .prepare_cached(
                        "SELECT entry, bm25(memories) FROM memories \
                         WHERE memories MATCH ?1 AND app_name = ?2 AND user_id = ?3 \
                         ORDER BY bm25(memories)",
                    )
                    .sql()?;
                let mut rows = stmt.query(params![expression, app_name, user_id]).sql()?;

                let mut hits = Vec::new();
                while let Some(row) = rows.next().sql()? {
                    let raw: String = row.get(0).sql()?;
                    // SQLite's bm25() is negative, most relevant most negative.
                    // Flipped so this backend agrees with the trait's "higher
                    // is more relevant", even though the scale differs.
                    let rank: f64 = row.get(1).sql()?;
                    let mut entry: MemoryEntry = serde_json::from_str(&raw)?;
                    entry.score = Some(-rank);
                    hits.push(entry);
                }
                Ok(hits)
            })
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_core::{Content, Event};

    async fn service() -> SqliteMemoryService {
        SqliteMemoryService::in_memory().await.unwrap()
    }

    fn temp_db() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("{}.sqlite3", adk_core::new_id("adk-memory-test")))
    }

    /// A session whose events are the given user utterances.
    fn session_with(app: &str, user: &str, id: &str, texts: &[&str]) -> Session {
        let mut session = Session::new(id, app, user);
        for text in texts {
            session
                .events
                .push(Event::new("inv", "user").with_content(Content::user_text(*text)));
        }
        session
    }

    #[tokio::test]
    async fn search_finds_the_matching_entry_and_ranks_it() {
        let svc = service().await;
        svc.add_session_to_memory(&session_with(
            "app",
            "u1",
            "s1",
            &["I love hiking in the Alps", "My cat is called Milo"],
        ))
        .await
        .unwrap();

        let hits = svc.search_memory("app", "u1", "hiking Alps").await.unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].content.text().contains("Alps"));
        assert!(hits[0].score.unwrap() > 0.0, "higher must mean better");

        assert!(svc
            .search_memory("app", "u1", "quantum")
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn another_user_and_another_app_cannot_see_it() {
        let svc = service().await;
        svc.add_session_to_memory(&session_with("app", "u1", "s1", &["I love hiking"]))
            .await
            .unwrap();

        assert!(svc
            .search_memory("app", "u2", "hiking")
            .await
            .unwrap()
            .is_empty());
        assert!(svc
            .search_memory("other", "u1", "hiking")
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn results_come_back_best_first() {
        let svc = service().await;
        let mut session = Session::new("s1", "app", "u1");
        for text in [
            "rust rust rust ownership and borrowing",
            "a passing mention of rust",
        ] {
            session
                .events
                .push(Event::new("inv", "user").with_content(Content::user_text(text)));
        }
        svc.add_session_to_memory(&session).await.unwrap();

        let hits = svc.search_memory("app", "u1", "rust").await.unwrap();
        assert_eq!(hits.len(), 2);
        assert!(
            hits[0].score.unwrap() >= hits[1].score.unwrap(),
            "scores must be descending: {:?}",
            hits.iter().map(|h| h.score).collect::<Vec<_>>()
        );
        assert!(hits[0].content.text().starts_with("rust rust rust"));
    }

    #[tokio::test]
    async fn partial_and_empty_events_are_not_ingested() {
        let svc = service().await;
        let mut session = Session::new("s1", "app", "u1");
        session.events.push(
            Event::new("inv", "agent")
                .with_content(Content::model_text("streaming chunk about hiking"))
                .as_partial(),
        );
        session.events.push(Event::new("inv", "agent")); // no content at all
        session
            .events
            .push(Event::new("inv", "agent").with_content(Content::model_text("   ")));
        svc.add_session_to_memory(&session).await.unwrap();

        assert!(svc
            .search_memory("app", "u1", "hiking")
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn an_empty_or_punctuation_only_query_matches_nothing() {
        let svc = service().await;
        svc.add_session_to_memory(&session_with("app", "u1", "s1", &["I love hiking"]))
            .await
            .unwrap();

        for query in ["", "   ", "?!.,"] {
            assert!(
                svc.search_memory("app", "u1", query)
                    .await
                    .unwrap()
                    .is_empty(),
                "query {query:?} should match nothing"
            );
        }
    }

    /// FTS5 reads its MATCH argument as a query language. A caller's prose is
    /// not that language, so these must be searched literally rather than
    /// erroring or changing the query's meaning.
    #[tokio::test]
    async fn fts_operators_in_a_query_are_treated_as_words() {
        let svc = service().await;
        svc.add_session_to_memory(&session_with(
            "app",
            "u1",
            "s1",
            &["I love hiking in the Alps"],
        ))
        .await
        .unwrap();

        for query in [
            "hiking OR NOT",
            "hiking -Alps",
            "hiking*",
            "NEAR(hiking Alps)",
            r#"hiking "unbalanced"#,
            "hiking AND (",
        ] {
            let hits = svc.search_memory("app", "u1", query).await;
            let hits = hits.unwrap_or_else(|e| panic!("query {query:?} errored: {e}"));
            assert_eq!(hits.len(), 1, "query {query:?} should still find the entry");
        }
    }

    #[tokio::test]
    async fn re_ingesting_a_session_replaces_rather_than_duplicates() {
        let svc = service().await;
        let first = session_with("app", "u1", "s1", &["I love hiking"]);
        svc.add_session_to_memory(&first).await.unwrap();
        assert_eq!(
            svc.search_memory("app", "u1", "hiking")
                .await
                .unwrap()
                .len(),
            1
        );

        // The same conversation, one turn longer — the normal case.
        let grown = session_with("app", "u1", "s1", &["I love hiking", "and also climbing"]);
        svc.add_session_to_memory(&grown).await.unwrap();

        assert_eq!(
            svc.search_memory("app", "u1", "hiking")
                .await
                .unwrap()
                .len(),
            1,
            "the first turn must not be stored twice"
        );
        assert_eq!(
            svc.search_memory("app", "u1", "climbing")
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn a_second_session_adds_to_the_same_users_memory() {
        let svc = service().await;
        svc.add_session_to_memory(&session_with("app", "u1", "s1", &["I love hiking"]))
            .await
            .unwrap();
        svc.add_session_to_memory(&session_with("app", "u1", "s2", &["I also love hiking"]))
            .await
            .unwrap();

        assert_eq!(
            svc.search_memory("app", "u1", "hiking")
                .await
                .unwrap()
                .len(),
            2
        );
    }

    #[tokio::test]
    async fn the_entry_carries_its_author_and_timestamp() {
        let svc = service().await;
        let mut session = Session::new("s1", "app", "u1");
        let event =
            Event::new("inv", "assistant").with_content(Content::model_text("about hiking"));
        let timestamp = event.timestamp;
        session.events.push(event);
        svc.add_session_to_memory(&session).await.unwrap();

        let hits = svc.search_memory("app", "u1", "hiking").await.unwrap();
        assert_eq!(hits[0].author, "assistant");
        assert_eq!(hits[0].timestamp, timestamp);
    }

    #[tokio::test]
    async fn memory_survives_reopening_the_file() {
        let path = temp_db();
        {
            let svc = SqliteMemoryService::open(&path).await.unwrap();
            svc.add_session_to_memory(&session_with(
                "app",
                "u1",
                "s1",
                &["I love hiking in the Alps"],
            ))
            .await
            .unwrap();
        }

        let svc = SqliteMemoryService::open(&path).await.unwrap();
        let hits = svc.search_memory("app", "u1", "Alps").await.unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].content.text().contains("Alps"));

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn search_is_case_insensitive() {
        let svc = service().await;
        svc.add_session_to_memory(&session_with("app", "u1", "s1", &["I love Hiking"]))
            .await
            .unwrap();
        assert_eq!(
            svc.search_memory("app", "u1", "HIKING")
                .await
                .unwrap()
                .len(),
            1
        );
    }
}
