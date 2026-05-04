use std::sync::Arc;

mod engine;

use engine::train_text2sql_model;
use text2sql::schema::{ColumnSchema, TableSchema};
use text2sql::server::{start_server, AppState, ProductInferenceEngine};

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("serve");

    match mode {
        "train" => run_train(&args).await,
        "serve" => run_serve(&args).await,
        "query" => run_query(&args),
        _ => {
            eprintln!("Usage: sptorch-text2sql <mode> [options]");
            eprintln!("  train [--steps N] [--lr F]   Train neural Text2SQL model");
            eprintln!("  serve [db_path]              Start HTTP server (default)");
            eprintln!("  query <question>             One-shot query (template mode)");
            std::process::exit(1);
        }
    }
}

async fn run_train(args: &[String]) {
    let mut steps = 200;
    let mut lr = 0.01f32;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => {
                steps = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(200);
                i += 2;
            }
            "--lr" => {
                lr = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0.01);
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    eprintln!("[sptorch-text2sql] training neural model: {} steps, lr={}", steps, lr);
    let schemas = demo_schemas();
    let (model, tok, final_loss) = train_text2sql_model(&schemas, steps, lr);
    eprintln!("[sptorch-text2sql] training complete, final loss: {:.4}", final_loss);

    let params = model.parameters();
    if let Err(e) = sptorch::serialize::save_checkpoint("text2sql_model.sptc", &params) {
        eprintln!("failed to save checkpoint: {}", e);
    } else {
        eprintln!("[sptorch-text2sql] model saved to text2sql_model.sptc");
    }

    let test_questions = [
        "How many employees are there?",
        "What is the average salary?",
        "total sales amount",
    ];
    eprintln!("\n[sptorch-text2sql] test generation:");
    for q in &test_questions {
        let sql = engine::generate_sql(&model, &tok, q, &schemas, 50);
        let valid = text2sql::sql_constraint::validate_sql(&sql);
        eprintln!(
            "  Q: {} => {} [{}]",
            q,
            sql,
            if valid.is_valid() { "valid" } else { "INVALID" }
        );
    }
}

async fn run_serve(args: &[String]) {
    let db_path = args.get(2);

    let schemas = if let Some(path) = db_path {
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
        eprintln!("[sptorch-text2sql] using demo schema");
        demo_schemas()
    };

    eprintln!("[sptorch-text2sql] loaded {} tables:", schemas.len());
    for s in &schemas {
        eprintln!("  {} ({} columns)", s.table_name, s.columns.len());
    }

    let engine = maybe_load_engine();
    let state = Arc::new(AppState {
        schema_info: schemas,
        engine,
    });
    let addr = "0.0.0.0:8080";
    eprintln!("[sptorch-text2sql] server on http://localhost:8080");
    eprintln!("  GET  /         Web UI");
    eprintln!("  POST /query    {{\"question\": \"...\"}}\n");

    if let Err(e) = start_server(addr, state).await {
        eprintln!("server error: {}", e);
        std::process::exit(1);
    }
}

fn maybe_load_engine() -> Option<Arc<dyn ProductInferenceEngine>> {
    if std::env::var("SPTORCH_TEXT2SQL_ENGINE").ok().as_deref() == Some("neural") {
        let schemas = demo_schemas();
        let (model, tok, _) = train_text2sql_model(&schemas, 120, 0.01);
        return Some(engine::NeuralText2SqlEngine::shared(model, tok));
    }
    None
}

fn run_query(args: &[String]) {
    let question = args[2..].join(" ");
    if question.is_empty() {
        eprintln!("Usage: sptorch-text2sql query <question>");
        std::process::exit(1);
    }

    let schemas = demo_schemas();
    let sql = text2sql::sql_constraint::generate_sql_stub(&question, &schemas);
    let valid = text2sql::sql_constraint::validate_sql(&sql);
    println!("{}", sql);
    if !valid.is_valid() {
        eprintln!("WARNING: generated SQL may be invalid");
    }
}

fn demo_schemas() -> Vec<TableSchema> {
    vec![
        TableSchema {
            table_name: "employees".into(),
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
                    name: "department".into(),
                    dtype: "TEXT".into(),
                    is_primary: false,
                },
                ColumnSchema {
                    name: "salary".into(),
                    dtype: "REAL".into(),
                    is_primary: false,
                },
                ColumnSchema {
                    name: "hire_date".into(),
                    dtype: "TEXT".into(),
                    is_primary: false,
                },
            ],
        },
        TableSchema {
            table_name: "departments".into(),
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
                    name: "budget".into(),
                    dtype: "REAL".into(),
                    is_primary: false,
                },
            ],
        },
        TableSchema {
            table_name: "sales".into(),
            columns: vec![
                ColumnSchema {
                    name: "id".into(),
                    dtype: "INTEGER".into(),
                    is_primary: true,
                },
                ColumnSchema {
                    name: "employee_id".into(),
                    dtype: "INTEGER".into(),
                    is_primary: false,
                },
                ColumnSchema {
                    name: "amount".into(),
                    dtype: "REAL".into(),
                    is_primary: false,
                },
                ColumnSchema {
                    name: "date".into(),
                    dtype: "TEXT".into(),
                    is_primary: false,
                },
            ],
        },
    ]
}
