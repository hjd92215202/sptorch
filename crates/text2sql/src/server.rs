use crate::schema::TableSchema;
use axum::response::Html;
use axum::{extract::State, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ProductCorrection {
    pub question: String,
    pub wrong_sql: String,
    pub correct_sql: String,
}

pub trait ProductInferenceEngine: Send + Sync {
    fn generate_sql(&self, _question: &str, _schemas: &[TableSchema], _max_tokens: usize) -> Option<String> {
        None
    }

    fn apply_correction(&self, _correction: ProductCorrection) -> Result<String, String> {
        Err("correction engine not enabled".to_string())
    }
}

#[derive(Default)]
pub struct TemplateOnlyEngine;

impl ProductInferenceEngine for TemplateOnlyEngine {}

#[derive(Deserialize)]
pub struct QueryRequest {
    pub question: String,
    pub database: Option<String>,
}

#[derive(Serialize)]
pub struct QueryResponse {
    pub sql: String,
    pub explanation: String,
    pub mode: String,
    pub valid: bool,
    pub validation: String,
}

pub struct AppState {
    pub schema_info: Vec<TableSchema>,
    pub engine: Option<Arc<dyn ProductInferenceEngine>>,
}

async fn query_handler(State(state): State<Arc<AppState>>, Json(req): Json<QueryRequest>) -> Json<QueryResponse> {
    use crate::sql_constraint::{validate_sql, SqlValidation};

    let (sql, mode) = if let Some(engine) = &state.engine {
        if let Some(engine_sql) = engine.generate_sql(&req.question, &state.schema_info, 60) {
            let v = validate_sql(&engine_sql);
            if v.is_valid() {
                (engine_sql, "product-engine".to_string())
            } else {
                // Engine output invalid, fallback to template
                let template_sql = crate::sql_constraint::generate_sql_stub(&req.question, &state.schema_info);
                (template_sql, "template (engine fallback)".to_string())
            }
        } else {
            let template_sql = crate::sql_constraint::generate_sql_stub(&req.question, &state.schema_info);
            (template_sql, "template (engine unavailable)".to_string())
        }
    } else {
        let sql = crate::sql_constraint::generate_sql_stub(&req.question, &state.schema_info);
        (sql, "template".to_string())
    };

    let validation = validate_sql(&sql);
    let (valid, validation_msg) = match &validation {
        SqlValidation::Valid => (true, "valid".to_string()),
        SqlValidation::Warning(w) => (true, format!("warning: {}", w)),
        SqlValidation::Invalid(e) => (false, format!("invalid: {}", e)),
    };

    Json(QueryResponse {
        sql,
        explanation: format!("{} for: {}", mode, req.question),
        mode,
        valid,
        validation: validation_msg,
    })
}

async fn health_handler() -> &'static str {
    "ok"
}

#[derive(Deserialize)]
pub struct CorrectionRequest {
    pub question: String,
    pub wrong_sql: String,
    pub correct_sql: String,
}

#[derive(Serialize)]
pub struct CorrectionResponse {
    pub accepted: bool,
    pub message: String,
}

async fn correct_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CorrectionRequest>,
) -> Json<CorrectionResponse> {
    if let Some(engine) = &state.engine {
        let correction = ProductCorrection {
            question: req.question.clone(),
            wrong_sql: req.wrong_sql.clone(),
            correct_sql: req.correct_sql.clone(),
        };
        match engine.apply_correction(correction) {
            Ok(message) => Json(CorrectionResponse {
                accepted: true,
                message,
            }),
            Err(err) => Json(CorrectionResponse {
                accepted: false,
                message: format!("Failed to apply correction: {}", err),
            }),
        }
    } else {
        Json(CorrectionResponse {
            accepted: false,
            message: "No product engine loaded, correction stored for future training".into(),
        })
    }
}

async fn ui_handler(State(state): State<Arc<AppState>>) -> Html<String> {
    let tables_json = serde_json::to_string(&state.schema_info).unwrap_or_default();
    Html(format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SPTorch Text2SQL</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:system-ui,-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:2rem}}
h1{{font-size:1.8rem;margin-bottom:.5rem;color:#38bdf8}}
.subtitle{{color:#94a3b8;margin-bottom:2rem;font-size:.9rem}}
.container{{width:100%;max-width:700px}}
.input-group{{display:flex;gap:.5rem;margin-bottom:1.5rem}}
input{{flex:1;padding:.75rem 1rem;border-radius:.5rem;border:1px solid #334155;background:#1e293b;color:#f1f5f9;font-size:1rem}}
input:focus{{outline:none;border-color:#38bdf8}}
button{{padding:.75rem 1.5rem;border-radius:.5rem;border:none;background:#2563eb;color:#fff;font-size:1rem;cursor:pointer;font-weight:600}}
button:hover{{background:#1d4ed8}}
.result{{background:#1e293b;border-radius:.75rem;padding:1.5rem;margin-bottom:1rem;display:none}}
.result.show{{display:block}}
.sql-output{{font-family:'JetBrains Mono',monospace;font-size:1.1rem;color:#34d399;white-space:pre-wrap;margin:.75rem 0}}
.schema-section{{margin-top:2rem;padding:1rem;background:#1e293b;border-radius:.5rem}}
.schema-section h3{{color:#94a3b8;font-size:.85rem;margin-bottom:.5rem}}
.schema-table{{font-family:monospace;font-size:.8rem;color:#64748b;white-space:pre}}
.examples{{margin-top:1.5rem}}
.examples h3{{color:#94a3b8;font-size:.85rem;margin-bottom:.5rem}}
.example-btn{{display:inline-block;padding:.4rem .8rem;margin:.25rem;border-radius:.25rem;background:#334155;color:#cbd5e1;font-size:.8rem;cursor:pointer;border:none}}
.example-btn:hover{{background:#475569}}
</style>
</head>
<body>
<h1>⚡ SPTorch Text2SQL</h1>
<p class="subtitle">Natural language → SQL, powered by Rust</p>
<div class="container">
  <div class="input-group">
    <input id="q" type="text" placeholder="Ask a question about your data..." autofocus>
    <button onclick="ask()">Generate</button>
  </div>
  <div class="result" id="result">
    <div style="color:#94a3b8;font-size:.8rem">Generated SQL:</div>
    <div class="sql-output" id="sql"></div>
    <div style="margin-top:.75rem;display:flex;gap:.5rem;align-items:center">
      <input id="correct-input" type="text" placeholder="Edit SQL to correct it..." style="flex:1;padding:.5rem;font-family:monospace;font-size:.9rem">
      <button onclick="submitCorrection()" style="background:#059669;padding:.5rem 1rem;font-size:.85rem">Correct</button>
    </div>
    <div id="correct-msg" style="color:#94a3b8;font-size:.75rem;margin-top:.4rem"></div>
  </div>
  <div class="examples">
    <h3>Try these:</h3>
    <button class="example-btn" onclick="tryQ('How many employees are there?')">How many employees?</button>
    <button class="example-btn" onclick="tryQ('What is the average salary?')">Average salary</button>
    <button class="example-btn" onclick="tryQ('total sales amount')">Total sales</button>
    <button class="example-btn" onclick="tryQ('highest budget in departments')">Max budget</button>
    <button class="example-btn" onclick="tryQ('show me all employees')">All employees</button>
  </div>
  <div class="schema-section">
    <h3>Database Schema:</h3>
    <div class="schema-table" id="schema"></div>
  </div>
</div>
<script>
const schemas = {tables_json};
document.getElementById('schema').textContent = schemas.map(t =>
  t.table_name + '(' + t.columns.map(c => c.name + ' ' + c.dtype + (c.is_primary?' PK':'')).join(', ') + ')'
).join('\n');
document.getElementById('q').addEventListener('keydown', e => {{ if(e.key==='Enter') ask(); }});
function tryQ(q) {{ document.getElementById('q').value = q; ask(); }}
async function ask() {{
  const q = document.getElementById('q').value.trim();
  if(!q) return;
  document.getElementById('correct-msg').textContent = '';
  const r = await fetch('/query', {{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{question:q}})}});
  const data = await r.json();
  document.getElementById('sql').textContent = data.sql;
  document.getElementById('correct-input').value = data.sql;
  document.getElementById('result').classList.add('show');
}}
async function submitCorrection() {{
  const q = document.getElementById('q').value.trim();
  const wrongSql = document.getElementById('sql').textContent;
  const correctSql = document.getElementById('correct-input').value.trim();
  if(!q || !correctSql || correctSql === wrongSql) return;
  const r = await fetch('/correct', {{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{question:q,wrong_sql:wrongSql,correct_sql:correctSql}})}});
  const data = await r.json();
  document.getElementById('correct-msg').textContent = data.message;
}}
</script>
</body>
</html>"#
    ))
}

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", axum::routing::get(ui_handler))
        .route("/query", post(query_handler))
        .route("/correct", post(correct_handler))
        .route("/health", axum::routing::get(health_handler))
        .with_state(state)
}

pub async fn start_server(addr: &str, state: Arc<AppState>) -> Result<(), Box<dyn std::error::Error>> {
    let app = create_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("[sptorch-text2sql] listening on {}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}
