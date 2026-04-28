use std::sync::Arc;
use text2sql::schema::{TableSchema, ColumnSchema};
use text2sql::server::{AppState, start_server};

#[tokio::main]
async fn main() {
    let db_path = std::env::args().nth(1);

    let schemas = if let Some(ref path) = db_path {
        let url = format!("sqlite:{}", path);
        eprintln!("[sptorch-text2sql] loading schema from {}", url);
        match text2sql::schema::fetch_sqlite_schema(&url).await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("failed to fetch schema: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        eprintln!("[sptorch-text2sql] no database specified, using demo schema");
        demo_schemas()
    };

    eprintln!("[sptorch-text2sql] loaded {} tables:", schemas.len());
    for s in &schemas {
        eprintln!("  {} ({} columns)", s.table_name, s.columns.len());
    }

    let state = Arc::new(AppState { schema_info: schemas });
    let addr = "0.0.0.0:8080";
    eprintln!("[sptorch-text2sql] starting server on {}", addr);
    eprintln!("[sptorch-text2sql] POST /query {{\"question\": \"...\"}}\n");

    if let Err(e) = start_server(addr, state).await {
        eprintln!("server error: {}", e);
        std::process::exit(1);
    }
}

fn demo_schemas() -> Vec<TableSchema> {
    vec![
        TableSchema {
            table_name: "employees".into(),
            columns: vec![
                ColumnSchema { name: "id".into(), dtype: "INTEGER".into(), is_primary: true },
                ColumnSchema { name: "name".into(), dtype: "TEXT".into(), is_primary: false },
                ColumnSchema { name: "department".into(), dtype: "TEXT".into(), is_primary: false },
                ColumnSchema { name: "salary".into(), dtype: "REAL".into(), is_primary: false },
                ColumnSchema { name: "hire_date".into(), dtype: "TEXT".into(), is_primary: false },
            ],
        },
        TableSchema {
            table_name: "departments".into(),
            columns: vec![
                ColumnSchema { name: "id".into(), dtype: "INTEGER".into(), is_primary: true },
                ColumnSchema { name: "name".into(), dtype: "TEXT".into(), is_primary: false },
                ColumnSchema { name: "budget".into(), dtype: "REAL".into(), is_primary: false },
            ],
        },
        TableSchema {
            table_name: "sales".into(),
            columns: vec![
                ColumnSchema { name: "id".into(), dtype: "INTEGER".into(), is_primary: true },
                ColumnSchema { name: "employee_id".into(), dtype: "INTEGER".into(), is_primary: false },
                ColumnSchema { name: "amount".into(), dtype: "REAL".into(), is_primary: false },
                ColumnSchema { name: "date".into(), dtype: "TEXT".into(), is_primary: false },
            ],
        },
    ]
}
