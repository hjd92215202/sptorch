use axum::{extract::State, routing::post, Json, Router};
use axum::response::Html;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Deserialize)]
pub struct QueryRequest {
    pub question: String,
    pub database: Option<String>,
}

#[derive(Serialize)]
pub struct QueryResponse {
    pub sql: String,
    pub explanation: String,
}

pub struct AppState {
    pub schema_info: Vec<crate::schema::TableSchema>,
}

async fn query_handler(State(state): State<Arc<AppState>>, Json(req): Json<QueryRequest>) -> Json<QueryResponse> {
    let prompt = crate::rag::build_prompt(&req.question, &state.schema_info);
    let sql = crate::sql_constraint::generate_sql_stub(&req.question, &state.schema_info);
    Json(QueryResponse {
        sql,
        explanation: format!("Generated from: {}", prompt),
    })
}

async fn health_handler() -> &'static str {
    "ok"
}

async fn ui_handler(State(state): State<Arc<AppState>>) -> Html<String> {
    let tables_json = serde_json::to_string(&state.schema_info).unwrap_or_default();
    Html(format!(r#"<!DOCTYPE html>
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
  const r = await fetch('/query', {{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{question:q}})}});
  const data = await r.json();
  document.getElementById('sql').textContent = data.sql;
  document.getElementById('result').classList.add('show');
}}
</script>
</body>
</html>"#))
}

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", axum::routing::get(ui_handler))
        .route("/query", post(query_handler))
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
