use axum::{extract::State, routing::post, Json, Router};
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

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
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
