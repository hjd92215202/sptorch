//! Text2SQL training data: synthetic question-SQL pairs for LoRA fine-tuning.

/// A single training sample: natural language question → SQL answer.
#[derive(Debug, Clone)]
pub struct Text2SqlSample {
    pub question: String,
    pub sql: String,
}

/// Generate synthetic training data for the demo schema (employees/departments/sales).
pub fn generate_training_data() -> Vec<Text2SqlSample> {
    vec![
        // COUNT queries
        Text2SqlSample { question: "How many employees are there?".into(), sql: "SELECT COUNT(*) FROM employees;".into() },
        Text2SqlSample { question: "Count all employees".into(), sql: "SELECT COUNT(*) FROM employees;".into() },
        Text2SqlSample { question: "How many departments exist?".into(), sql: "SELECT COUNT(*) FROM departments;".into() },
        Text2SqlSample { question: "Total number of sales".into(), sql: "SELECT COUNT(*) FROM sales;".into() },
        Text2SqlSample { question: "How many sales records?".into(), sql: "SELECT COUNT(*) FROM sales;".into() },
        // AVG queries
        Text2SqlSample { question: "What is the average salary?".into(), sql: "SELECT AVG(salary) FROM employees;".into() },
        Text2SqlSample { question: "Average employee salary".into(), sql: "SELECT AVG(salary) FROM employees;".into() },
        Text2SqlSample { question: "Mean budget of departments".into(), sql: "SELECT AVG(budget) FROM departments;".into() },
        Text2SqlSample { question: "Average sales amount".into(), sql: "SELECT AVG(amount) FROM sales;".into() },
        // SUM queries
        Text2SqlSample { question: "Total salary expense".into(), sql: "SELECT SUM(salary) FROM employees;".into() },
        Text2SqlSample { question: "Sum of all sales".into(), sql: "SELECT SUM(amount) FROM sales;".into() },
        Text2SqlSample { question: "Total budget across departments".into(), sql: "SELECT SUM(budget) FROM departments;".into() },
        Text2SqlSample { question: "What is the total amount of sales?".into(), sql: "SELECT SUM(amount) FROM sales;".into() },
        // MAX queries
        Text2SqlSample { question: "Highest salary".into(), sql: "SELECT MAX(salary) FROM employees;".into() },
        Text2SqlSample { question: "Maximum budget".into(), sql: "SELECT MAX(budget) FROM departments;".into() },
        Text2SqlSample { question: "Largest sale amount".into(), sql: "SELECT MAX(amount) FROM sales;".into() },
        Text2SqlSample { question: "What is the highest budget in departments?".into(), sql: "SELECT MAX(budget) FROM departments;".into() },
        // MIN queries
        Text2SqlSample { question: "Lowest salary".into(), sql: "SELECT MIN(salary) FROM employees;".into() },
        Text2SqlSample { question: "Minimum budget".into(), sql: "SELECT MIN(budget) FROM departments;".into() },
        Text2SqlSample { question: "Smallest sale".into(), sql: "SELECT MIN(amount) FROM sales;".into() },
        // SELECT * queries
        Text2SqlSample { question: "Show all employees".into(), sql: "SELECT * FROM employees LIMIT 10;".into() },
        Text2SqlSample { question: "List all departments".into(), sql: "SELECT * FROM departments LIMIT 10;".into() },
        Text2SqlSample { question: "Show me the sales".into(), sql: "SELECT * FROM sales LIMIT 10;".into() },
        Text2SqlSample { question: "Display all employee records".into(), sql: "SELECT * FROM employees LIMIT 10;".into() },
        // SELECT specific columns
        Text2SqlSample { question: "List employee names".into(), sql: "SELECT name FROM employees;".into() },
        Text2SqlSample { question: "Show department names and budgets".into(), sql: "SELECT name, budget FROM departments;".into() },
        Text2SqlSample { question: "Get employee names and salaries".into(), sql: "SELECT name, salary FROM employees;".into() },
        // WHERE clauses
        Text2SqlSample { question: "Employees in engineering department".into(), sql: "SELECT * FROM employees WHERE department = 'engineering';".into() },
        Text2SqlSample { question: "Sales above 1000".into(), sql: "SELECT * FROM sales WHERE amount > 1000;".into() },
        Text2SqlSample { question: "Departments with budget over 50000".into(), sql: "SELECT * FROM departments WHERE budget > 50000;".into() },
        // ORDER BY
        Text2SqlSample { question: "Employees ordered by salary".into(), sql: "SELECT * FROM employees ORDER BY salary DESC;".into() },
        Text2SqlSample { question: "Top sales by amount".into(), sql: "SELECT * FROM sales ORDER BY amount DESC LIMIT 5;".into() },
        // GROUP BY
        Text2SqlSample { question: "Count employees per department".into(), sql: "SELECT department, COUNT(*) FROM employees GROUP BY department;".into() },
        Text2SqlSample { question: "Total sales per employee".into(), sql: "SELECT employee_id, SUM(amount) FROM sales GROUP BY employee_id;".into() },
    ]
}

/// Simple character-level tokenizer for SQL training.
/// Maps chars to token IDs for the tiny GPT model.
pub struct SqlTokenizer {
    pub char_to_id: std::collections::HashMap<char, usize>,
    pub id_to_char: Vec<char>,
    pub vocab_size: usize,
}

impl SqlTokenizer {
    pub fn new() -> Self {
        let chars: Vec<char> = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.,;()*><=!'\"?\n".chars().collect();
        let char_to_id: std::collections::HashMap<char, usize> = chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let vocab_size = chars.len();
        SqlTokenizer { char_to_id, id_to_char: chars, vocab_size }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().map(|c| *self.char_to_id.get(&c).unwrap_or(&0)).collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|&id| {
            if id < self.id_to_char.len() { self.id_to_char[id] } else { '?' }
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_data_count() {
        let data = generate_training_data();
        assert!(data.len() >= 30, "should have at least 30 training samples");
    }

    #[test]
    fn test_tokenizer_roundtrip() {
        let tok = SqlTokenizer::new();
        let text = "SELECT COUNT(*) FROM employees;";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_tokenizer_vocab_size() {
        let tok = SqlTokenizer::new();
        assert!(tok.vocab_size > 50);
        assert!(tok.vocab_size < 200);
    }
}
