use crate::schema::{ColumnSchema, TableSchema};

/// SQL keyword whitelist for constrained generation.
const SQL_KEYWORDS: &[&str] = &[
    "SELECT",
    "FROM",
    "WHERE",
    "AND",
    "OR",
    "NOT",
    "IN",
    "BETWEEN",
    "LIKE",
    "IS",
    "NULL",
    "JOIN",
    "LEFT",
    "RIGHT",
    "INNER",
    "OUTER",
    "ON",
    "GROUP",
    "BY",
    "ORDER",
    "ASC",
    "DESC",
    "HAVING",
    "LIMIT",
    "OFFSET",
    "AS",
    "COUNT",
    "SUM",
    "AVG",
    "MIN",
    "MAX",
    "DISTINCT",
    "INSERT",
    "INTO",
    "VALUES",
    "UPDATE",
    "SET",
    "DELETE",
    "CREATE",
    "TABLE",
    "DROP",
    "ALTER",
    "ADD",
    "COLUMN",
    "INDEX",
    "PRIMARY",
    "KEY",
    "FOREIGN",
    "REFERENCES",
    "CASE",
    "WHEN",
    "THEN",
    "ELSE",
    "END",
    "UNION",
    "ALL",
    "EXISTS",
    "CAST",
    "COALESCE",
];

/// Build allowed token set from schema (table names + column names + SQL keywords).
pub fn build_sql_vocabulary(schemas: &[TableSchema]) -> Vec<String> {
    let mut vocab: Vec<String> = SQL_KEYWORDS.iter().map(|s| s.to_string()).collect();
    for s in schemas {
        vocab.push(s.table_name.clone());
        for col in &s.columns {
            vocab.push(col.name.clone());
        }
    }
    // Add common literals/operators
    vocab.extend(
        [
            "*", ",", ".", "(", ")", "=", "<", ">", "<=", ">=", "!=", ";", "'", "0", "1",
        ]
        .iter()
        .map(|s| s.to_string()),
    );
    vocab.sort();
    vocab.dedup();
    vocab
}

/// Stub SQL generator: pattern-match common question types to SQL templates.
/// In production this would be replaced by the neural model with constrained decoding.
pub fn generate_sql_stub(question: &str, schemas: &[TableSchema]) -> String {
    let q = question.to_lowercase();

    // Find most relevant table
    let table = schemas
        .iter()
        .max_by_key(|s| {
            let mut score = 0i32;
            if q.contains(&s.table_name.to_lowercase()) {
                score += 10;
            }
            for col in &s.columns {
                if q.contains(&col.name.to_lowercase()) {
                    score += 3;
                }
            }
            score
        })
        .map(|s| &s.table_name);

    let table_name = match table {
        Some(t) => t.as_str(),
        None => return "SELECT 1;".to_string(),
    };

    if q.contains("how many") || q.contains("count") {
        return format!("SELECT COUNT(*) FROM {};", table_name);
    }
    if q.contains("average") || q.contains("avg") {
        if let Some(col) = find_numeric_column(schemas, table_name) {
            return format!("SELECT AVG({}) FROM {};", col, table_name);
        }
    }
    if q.contains("total") || q.contains("sum") {
        if let Some(col) = find_numeric_column(schemas, table_name) {
            return format!("SELECT SUM({}) FROM {};", col, table_name);
        }
    }
    if q.contains("max") || q.contains("highest") || q.contains("largest") {
        if let Some(col) = find_numeric_column(schemas, table_name) {
            return format!("SELECT MAX({}) FROM {};", col, table_name);
        }
    }

    format!("SELECT * FROM {} LIMIT 10;", table_name)
}

fn find_numeric_column(schemas: &[TableSchema], table_name: &str) -> Option<String> {
    schemas.iter().find(|s| s.table_name == table_name).and_then(|s| {
        let numeric_cols: Vec<&ColumnSchema> = s
            .columns
            .iter()
            .filter(|c| {
                let dt = c.dtype.to_uppercase();
                (dt.contains("REAL") || dt.contains("FLOAT") || dt.contains("NUMERIC") || dt.contains("DECIMAL"))
                    && !c.is_primary
            })
            .collect();
        if !numeric_cols.is_empty() {
            return Some(numeric_cols[0].name.clone());
        }
        s.columns
            .iter()
            .find(|c| {
                let dt = c.dtype.to_uppercase();
                dt.contains("INT") && !c.is_primary
            })
            .map(|c| c.name.clone())
    })
}

/// Basic SQL syntax validator. Checks structural correctness, not semantics.
pub fn validate_sql(sql: &str) -> SqlValidation {
    let trimmed = sql.trim();
    if trimmed.is_empty() {
        return SqlValidation::Invalid("empty SQL".into());
    }

    let upper = trimmed.to_uppercase();

    // Must start with a known statement keyword
    let valid_starts = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"];
    if !valid_starts.iter().any(|k| upper.starts_with(k)) {
        return SqlValidation::Invalid(format!("SQL must start with a statement keyword, got: {}", &trimmed[..trimmed.len().min(20)]));
    }

    // SELECT must have FROM (unless it's SELECT 1 or SELECT expression)
    if upper.starts_with("SELECT") && !upper.contains("FROM") {
        let after_select = upper.strip_prefix("SELECT").unwrap().trim();
        // Allow SELECT <literal> or SELECT <function>
        if !after_select.starts_with("1") && !after_select.starts_with("COUNT") && !after_select.starts_with("'") {
            return SqlValidation::Warning("SELECT without FROM clause".into());
        }
    }

    // Check balanced parentheses
    let open = trimmed.chars().filter(|&c| c == '(').count();
    let close = trimmed.chars().filter(|&c| c == ')').count();
    if open != close {
        return SqlValidation::Invalid(format!("unbalanced parentheses: {} open, {} close", open, close));
    }

    // Check balanced quotes
    let single_quotes = trimmed.chars().filter(|&c| c == '\'').count();
    if single_quotes % 2 != 0 {
        return SqlValidation::Invalid("unbalanced single quotes".into());
    }

    // Should end with semicolon (warning, not error)
    if !trimmed.ends_with(';') {
        return SqlValidation::Warning("SQL should end with semicolon".into());
    }

    SqlValidation::Valid
}

#[derive(Debug, Clone, PartialEq)]
pub enum SqlValidation {
    Valid,
    Warning(String),
    Invalid(String),
}

impl SqlValidation {
    pub fn is_valid(&self) -> bool {
        matches!(self, SqlValidation::Valid | SqlValidation::Warning(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{ColumnSchema, TableSchema};

    fn sample_schemas() -> Vec<TableSchema> {
        vec![
            TableSchema {
                table_name: "orders".into(),
                columns: vec![
                    ColumnSchema {
                        name: "id".into(),
                        dtype: "INTEGER".into(),
                        is_primary: true,
                    },
                    ColumnSchema {
                        name: "user_id".into(),
                        dtype: "INTEGER".into(),
                        is_primary: false,
                    },
                    ColumnSchema {
                        name: "amount".into(),
                        dtype: "REAL".into(),
                        is_primary: false,
                    },
                ],
            },
            TableSchema {
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
                ],
            },
        ]
    }

    #[test]
    fn test_build_sql_vocabulary() {
        let schemas = sample_schemas();
        let vocab = build_sql_vocabulary(&schemas);
        assert!(vocab.contains(&"SELECT".to_string()));
        assert!(vocab.contains(&"orders".to_string()));
        assert!(vocab.contains(&"amount".to_string()));
        assert!(vocab.contains(&"*".to_string()));
    }

    #[test]
    fn test_generate_count() {
        let schemas = sample_schemas();
        let sql = generate_sql_stub("How many orders are there?", &schemas);
        assert_eq!(sql, "SELECT COUNT(*) FROM orders;");
    }

    #[test]
    fn test_generate_avg() {
        let schemas = sample_schemas();
        let sql = generate_sql_stub("What is the average amount in orders?", &schemas);
        assert_eq!(sql, "SELECT AVG(amount) FROM orders;");
    }

    #[test]
    fn test_generate_sum() {
        let schemas = sample_schemas();
        let sql = generate_sql_stub("total amount of all orders", &schemas);
        assert_eq!(sql, "SELECT SUM(amount) FROM orders;");
    }

    #[test]
    fn test_generate_fallback() {
        let schemas = sample_schemas();
        let sql = generate_sql_stub("show me users", &schemas);
        assert_eq!(sql, "SELECT * FROM users LIMIT 10;");
    }

    #[test]
    fn test_generate_max() {
        let schemas = sample_schemas();
        let sql = generate_sql_stub("highest amount in orders", &schemas);
        assert_eq!(sql, "SELECT MAX(amount) FROM orders;");
    }

    // --- SQL validation tests ---

    #[test]
    fn test_validate_valid_sql() {
        assert_eq!(validate_sql("SELECT * FROM users;"), SqlValidation::Valid);
        assert_eq!(validate_sql("SELECT COUNT(*) FROM orders;"), SqlValidation::Valid);
        assert_eq!(validate_sql("INSERT INTO users VALUES (1, 'test');"), SqlValidation::Valid);
    }

    #[test]
    fn test_validate_empty() {
        assert_eq!(validate_sql(""), SqlValidation::Invalid("empty SQL".into()));
    }

    #[test]
    fn test_validate_bad_start() {
        let r = validate_sql("HELLO world");
        assert!(!r.is_valid());
    }

    #[test]
    fn test_validate_unbalanced_parens() {
        let r = validate_sql("SELECT COUNT(* FROM orders;");
        assert_eq!(r, SqlValidation::Invalid("unbalanced parentheses: 1 open, 0 close".into()));
    }

    #[test]
    fn test_validate_unbalanced_quotes() {
        let r = validate_sql("SELECT * FROM users WHERE name = 'test;");
        assert_eq!(r, SqlValidation::Invalid("unbalanced single quotes".into()));
    }

    #[test]
    fn test_validate_no_semicolon() {
        let r = validate_sql("SELECT * FROM users");
        assert_eq!(r, SqlValidation::Warning("SQL should end with semicolon".into()));
        assert!(r.is_valid()); // warning is still "valid enough"
    }
}
