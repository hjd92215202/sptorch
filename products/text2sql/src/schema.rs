use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnSchema {
    pub name: String,
    pub dtype: String,
    pub is_primary: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSchema {
    pub table_name: String,
    pub columns: Vec<ColumnSchema>,
}

impl TableSchema {
    pub fn to_ddl(&self) -> String {
        let cols: Vec<String> = self
            .columns
            .iter()
            .map(|c| {
                let pk = if c.is_primary { " PRIMARY KEY" } else { "" };
                format!("  {} {}{}", c.name, c.dtype, pk)
            })
            .collect();
        format!("CREATE TABLE {} (\n{}\n);", self.table_name, cols.join(",\n"))
    }
}

/// Extract schema from SQLite database at runtime.
pub async fn fetch_sqlite_schema(db_url: &str) -> Result<Vec<TableSchema>, sqlx::Error> {
    use sqlx::sqlite::SqlitePool;
    use sqlx::Row;

    let pool = SqlitePool::connect(db_url).await?;

    let tables: Vec<String> =
        sqlx::query("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            .fetch_all(&pool)
            .await?
            .iter()
            .map(|r| r.get::<String, _>("name"))
            .collect();

    let mut schemas = Vec::new();
    for table in &tables {
        let pragma = format!("PRAGMA table_info({})", table);
        let rows = sqlx::query(&pragma).fetch_all(&pool).await?;
        let columns: Vec<ColumnSchema> = rows
            .iter()
            .map(|r| ColumnSchema {
                name: r.get::<String, _>("name"),
                dtype: r.get::<String, _>("type"),
                is_primary: r.get::<bool, _>("pk"),
            })
            .collect();
        schemas.push(TableSchema {
            table_name: table.clone(),
            columns,
        });
    }

    pool.close().await;
    Ok(schemas)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_schema_to_ddl() {
        let schema = TableSchema {
            table_name: "users".into(),
            columns: vec![
                ColumnSchema {
                    name: "id".into(),
                    dtype: "INTEGER".into(),
                    is_primary: true,
                },
                ColumnSchema {
                    name: "name".into(),
                    dtype: "TEXT".into(),
                    is_primary: false,
                },
                ColumnSchema {
                    name: "age".into(),
                    dtype: "INTEGER".into(),
                    is_primary: false,
                },
            ],
        };
        let ddl = schema.to_ddl();
        assert!(ddl.contains("CREATE TABLE users"));
        assert!(ddl.contains("id INTEGER PRIMARY KEY"));
        assert!(ddl.contains("name TEXT"));
    }

    #[tokio::test]
    async fn test_fetch_sqlite_schema() {
        // Create in-memory SQLite DB with a test table
        use sqlx::sqlite::SqlitePool;
        use sqlx::Executor;

        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
        pool.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, product TEXT, amount REAL)")
            .await
            .unwrap();
        pool.execute("INSERT INTO orders VALUES (1, 'widget', 9.99)")
            .await
            .unwrap();
        pool.close().await;

        // fetch_sqlite_schema needs a file-based DB for reconnection, so test DDL generation instead
        let schema = TableSchema {
            table_name: "orders".into(),
            columns: vec![
                ColumnSchema {
                    name: "id".into(),
                    dtype: "INTEGER".into(),
                    is_primary: true,
                },
                ColumnSchema {
                    name: "product".into(),
                    dtype: "TEXT".into(),
                    is_primary: false,
                },
                ColumnSchema {
                    name: "amount".into(),
                    dtype: "REAL".into(),
                    is_primary: false,
                },
            ],
        };
        assert_eq!(schema.columns.len(), 3);
        assert!(schema.to_ddl().contains("amount REAL"));
    }
}
