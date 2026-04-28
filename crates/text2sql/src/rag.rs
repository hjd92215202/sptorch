use crate::schema::TableSchema;

/// Build a RAG-style prompt: inject table DDLs before the user question.
pub fn build_prompt(question: &str, schemas: &[TableSchema]) -> String {
    let mut prompt = String::from("Given the following database schema:\n\n");
    for s in schemas {
        prompt.push_str(&s.to_ddl());
        prompt.push_str("\n\n");
    }
    prompt.push_str(&format!("Question: {}\nSQL:", question));
    prompt
}

/// Simple keyword-based table relevance scoring for schema filtering.
pub fn rank_tables(question: &str, schemas: &[TableSchema]) -> Vec<(usize, f32)> {
    let q_lower = question.to_lowercase();
    let mut scores: Vec<(usize, f32)> = schemas.iter().enumerate().map(|(i, s)| {
        let mut score = 0.0f32;
        if q_lower.contains(&s.table_name.to_lowercase()) {
            score += 5.0;
        }
        for col in &s.columns {
            if q_lower.contains(&col.name.to_lowercase()) {
                score += 2.0;
            }
        }
        (i, score)
    }).collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores
}

/// Filter to top-k most relevant tables for the prompt.
pub fn select_relevant_schemas(question: &str, schemas: &[TableSchema], top_k: usize) -> Vec<TableSchema> {
    let ranked = rank_tables(question, schemas);
    ranked.iter()
        .take(top_k)
        .filter(|(_, score)| *score > 0.0)
        .map(|(i, _)| schemas[*i].clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{ColumnSchema, TableSchema};

    fn sample_schemas() -> Vec<TableSchema> {
        vec![
            TableSchema {
                table_name: "users".into(),
                columns: vec![
                    ColumnSchema { name: "id".into(), dtype: "INTEGER".into(), is_primary: true },
                    ColumnSchema { name: "name".into(), dtype: "TEXT".into(), is_primary: false },
                ],
            },
            TableSchema {
                table_name: "orders".into(),
                columns: vec![
                    ColumnSchema { name: "id".into(), dtype: "INTEGER".into(), is_primary: true },
                    ColumnSchema { name: "user_id".into(), dtype: "INTEGER".into(), is_primary: false },
                    ColumnSchema { name: "amount".into(), dtype: "REAL".into(), is_primary: false },
                ],
            },
        ]
    }

    #[test]
    fn test_build_prompt() {
        let schemas = sample_schemas();
        let prompt = build_prompt("How many orders?", &schemas);
        assert!(prompt.contains("CREATE TABLE users"));
        assert!(prompt.contains("CREATE TABLE orders"));
        assert!(prompt.contains("Question: How many orders?"));
        assert!(prompt.ends_with("SQL:"));
    }

    #[test]
    fn test_rank_tables() {
        let schemas = sample_schemas();
        let ranked = rank_tables("total amount per user in orders", &schemas);
        // "orders" table should rank higher (table name match + "amount" column)
        assert_eq!(ranked[0].0, 1); // orders index
        assert!(ranked[0].1 > ranked[1].1);
    }

    #[test]
    fn test_select_relevant() {
        let schemas = sample_schemas();
        let selected = select_relevant_schemas("show all users", &schemas, 1);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].table_name, "users");
    }
}
