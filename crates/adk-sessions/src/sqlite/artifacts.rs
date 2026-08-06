//! The SQLite-backed [`ArtifactService`].

use adk_core::{AdkError, ArtifactService, Part, Result, USER_PREFIX};
use async_trait::async_trait;
use rusqlite::{params, OptionalExtension};

use super::{Db, SqlExt};

/// An [`ArtifactService`] that stores versioned payloads in SQLite.
///
/// Behaviourally identical to
/// [`InMemoryArtifactService`](crate::InMemoryArtifactService): saving a
/// filename appends a version rather than overwriting, loading without a version
/// returns the latest, and a `user:`-prefixed filename is visible from every one
/// of that user's threads. See the [module documentation](super) for the storage
/// layout and the concurrency model.
///
/// Build one with [`SqliteArtifactService::open`], or take it from a
/// [`SqliteStore`](super::SqliteStore) to share a database with the session
/// service.
#[derive(Debug, Clone)]
pub struct SqliteArtifactService {
    db: Db,
}

impl SqliteArtifactService {
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

    /// Artifacts named `user:<name>` are shared across a user's sessions,
    /// mirroring the `user:` state prefix. Others belong to one thread.
    fn scope_key(session_id: &str, filename: &str) -> String {
        if filename.starts_with(USER_PREFIX) {
            String::new()
        } else {
            session_id.to_string()
        }
    }
}

#[async_trait]
impl ArtifactService for SqliteArtifactService {
    async fn save_artifact(
        &self,
        app_name: &str,
        user_id: &str,
        session_id: &str,
        filename: &str,
        part: Part,
    ) -> Result<u64> {
        let app_name = app_name.to_string();
        let user_id = user_id.to_string();
        let scope = Self::scope_key(session_id, filename);
        let filename = filename.to_string();

        self.db
            .with(move |conn| {
                let payload = serde_json::to_string(&part)?;
                // Reading the next version and writing it must be one unit, or
                // two concurrent saves would both claim the same number.
                let tx = conn.transaction().sql()?;
                let version: i64 = tx
                    .prepare_cached(
                        "SELECT COALESCE(MAX(version), -1) + 1 FROM artifacts \
                         WHERE app_name = ?1 AND user_id = ?2 AND scope = ?3 AND filename = ?4",
                    )
                    .sql()?
                    .query_row(params![app_name, user_id, scope, filename], |row| {
                        row.get(0)
                    })
                    .sql()?;
                tx.execute(
                    "INSERT INTO artifacts (app_name, user_id, scope, filename, version, payload) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![app_name, user_id, scope, filename, version, payload],
                )
                .sql()?;
                tx.commit().sql()?;
                Ok(version as u64)
            })
            .await
    }

    async fn load_artifact(
        &self,
        app_name: &str,
        user_id: &str,
        session_id: &str,
        filename: &str,
        version: Option<u64>,
    ) -> Result<Option<Part>> {
        let app_name = app_name.to_string();
        let user_id = user_id.to_string();
        let scope = Self::scope_key(session_id, filename);
        let filename = filename.to_string();

        self.db
            .with(move |conn| {
                let payload: Option<String> = match version {
                    Some(v) => conn
                        .prepare_cached(
                            "SELECT payload FROM artifacts WHERE app_name = ?1 AND user_id = ?2 \
                             AND scope = ?3 AND filename = ?4 AND version = ?5",
                        )
                        .sql()?
                        .query_row(
                            params![app_name, user_id, scope, filename, v as i64],
                            |row| row.get(0),
                        )
                        .optional()
                        .sql()?,
                    None => conn
                        .prepare_cached(
                            "SELECT payload FROM artifacts WHERE app_name = ?1 AND user_id = ?2 \
                             AND scope = ?3 AND filename = ?4 ORDER BY version DESC LIMIT 1",
                        )
                        .sql()?
                        .query_row(params![app_name, user_id, scope, filename], |row| {
                            row.get(0)
                        })
                        .optional()
                        .sql()?,
                };
                match payload {
                    Some(raw) => Ok(Some(serde_json::from_str(&raw)?)),
                    None => Ok(None),
                }
            })
            .await
    }

    async fn list_artifact_keys(
        &self,
        app_name: &str,
        user_id: &str,
        session_id: &str,
    ) -> Result<Vec<String>> {
        let app_name = app_name.to_string();
        let user_id = user_id.to_string();
        let session_id = session_id.to_string();

        self.db
            .with(move |conn| {
                // The empty scope holds this user's cross-session artifacts, so
                // both it and the thread's own scope are visible here.
                let mut stmt = conn
                    .prepare_cached(
                        "SELECT DISTINCT filename FROM artifacts \
                         WHERE app_name = ?1 AND user_id = ?2 AND (scope = '' OR scope = ?3) \
                         ORDER BY filename",
                    )
                    .sql()?;
                let mut rows = stmt.query(params![app_name, user_id, session_id]).sql()?;
                let mut keys = Vec::new();
                while let Some(row) = rows.next().sql()? {
                    keys.push(row.get(0).sql()?);
                }
                Ok(keys)
            })
            .await
    }

    async fn delete_artifact(
        &self,
        app_name: &str,
        user_id: &str,
        session_id: &str,
        filename: &str,
    ) -> Result<()> {
        let app_name = app_name.to_string();
        let user_id = user_id.to_string();
        let scope = Self::scope_key(session_id, filename);
        let filename = filename.to_string();

        self.db
            .with(move |conn| {
                // Deleting an artifact removes every version of it.
                let removed = conn
                    .execute(
                        "DELETE FROM artifacts WHERE app_name = ?1 AND user_id = ?2 \
                         AND scope = ?3 AND filename = ?4",
                        params![app_name, user_id, scope, filename],
                    )
                    .sql()?;
                if removed == 0 {
                    return Err(AdkError::ArtifactNotFound(filename));
                }
                Ok(())
            })
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_core::Blob;

    async fn service() -> SqliteArtifactService {
        SqliteArtifactService::in_memory().await.unwrap()
    }

    fn blob(data: &str) -> Part {
        Part::InlineData(Blob {
            mime_type: "text/plain".into(),
            data: data.to_string(),
        })
    }

    /// A unique path under the system temp directory, removed by the caller.
    fn temp_db() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("{}.sqlite3", adk_core::new_id("adk-artifact-test")))
    }

    #[tokio::test]
    async fn artifacts_are_versioned_per_filename() {
        let svc = service().await;

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
        // A version that was never written is absent, not the nearest one.
        assert!(svc
            .load_artifact("app", "u1", "s1", "notes.txt", Some(7))
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn user_scoped_artifacts_cross_sessions() {
        let svc = service().await;
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
    async fn another_user_and_another_app_are_isolated() {
        let svc = service().await;
        svc.save_artifact("app", "u1", "s1", "user:profile", Part::text("mine"))
            .await
            .unwrap();

        assert!(svc
            .load_artifact("app", "u2", "s1", "user:profile", None)
            .await
            .unwrap()
            .is_none());
        assert!(svc
            .load_artifact("other", "u1", "s1", "user:profile", None)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn missing_artifact_loads_as_none() {
        let svc = service().await;
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
    async fn listing_shows_this_thread_and_the_users_own_files() {
        let svc = service().await;
        svc.save_artifact("app", "u1", "s1", "report.md", Part::text("a"))
            .await
            .unwrap();
        // A second version must not produce a second listing entry.
        svc.save_artifact("app", "u1", "s1", "report.md", Part::text("b"))
            .await
            .unwrap();
        svc.save_artifact("app", "u1", "s1", "user:profile", Part::text("c"))
            .await
            .unwrap();
        svc.save_artifact("app", "u1", "s2", "other.md", Part::text("d"))
            .await
            .unwrap();

        assert_eq!(
            svc.list_artifact_keys("app", "u1", "s1").await.unwrap(),
            vec!["report.md".to_string(), "user:profile".to_string()]
        );
        // The other thread sees its own file plus the user-scoped one.
        assert_eq!(
            svc.list_artifact_keys("app", "u1", "s2").await.unwrap(),
            vec!["other.md".to_string(), "user:profile".to_string()]
        );
    }

    #[tokio::test]
    async fn deleting_removes_every_version() {
        let svc = service().await;
        for data in ["v0", "v1", "v2"] {
            svc.save_artifact("app", "u1", "s1", "notes.txt", blob(data))
                .await
                .unwrap();
        }
        svc.delete_artifact("app", "u1", "s1", "notes.txt")
            .await
            .unwrap();

        assert!(svc
            .load_artifact("app", "u1", "s1", "notes.txt", Some(0))
            .await
            .unwrap()
            .is_none());
        assert!(svc
            .list_artifact_keys("app", "u1", "s1")
            .await
            .unwrap()
            .is_empty());
        // Versioning restarts, since nothing is left to count from.
        assert_eq!(
            svc.save_artifact("app", "u1", "s1", "notes.txt", blob("fresh"))
                .await
                .unwrap(),
            0
        );
    }

    #[tokio::test]
    async fn versions_survive_reopening_the_file() {
        let path = temp_db();
        {
            let svc = SqliteArtifactService::open(&path).await.unwrap();
            for data in ["v0", "v1"] {
                svc.save_artifact("app", "u1", "s1", "notes.txt", blob(data))
                    .await
                    .unwrap();
            }
        }

        // A fresh process would do exactly this: open the same file and read.
        let svc = SqliteArtifactService::open(&path).await.unwrap();
        assert_eq!(
            svc.load_artifact("app", "u1", "s1", "notes.txt", Some(0))
                .await
                .unwrap()
                .unwrap(),
            blob("v0")
        );
        // The next save continues the sequence rather than restarting it.
        assert_eq!(
            svc.save_artifact("app", "u1", "s1", "notes.txt", blob("v2"))
                .await
                .unwrap(),
            2
        );

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn every_part_variant_round_trips() {
        let svc = service().await;
        // Payloads are stored as JSON, so a non-text part must survive too.
        let part = blob("YmluYXJ5");
        svc.save_artifact("app", "u1", "s1", "image.png", part.clone())
            .await
            .unwrap();
        assert_eq!(
            svc.load_artifact("app", "u1", "s1", "image.png", None)
                .await
                .unwrap()
                .unwrap(),
            part
        );
    }
}
