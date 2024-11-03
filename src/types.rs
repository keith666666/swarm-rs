use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Serialize, Deserialize)]
pub struct Tool {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) parameters: Value,
}

impl Tool {
    pub fn new(name: &str, description: &str, parameters: Value) -> Self {
        Tool {
            name: name.to_string(),
            description: description.to_string(),
            parameters,
        }
    }
}

impl Clone for Tool {
    fn clone(&self) -> Self {
        Tool {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
        }
    }
}

impl Default for Tool {
    fn default() -> Self {
        Tool {
            name: String::new(),
            description: String::new(),
            parameters: Value::Null,
        }
    }
}

impl std::fmt::Debug for Tool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("parameters", &self.parameters)
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub value: String,
    pub agent: Option<Agent>,
    pub context_variables: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub name: String,
    pub model: String,
    pub instructions: String,
    pub tools: Vec<Tool>,
    pub tool_choice: Option<String>,
    pub parallel_tool_calls: bool,
}

impl Default for Agent {
    fn default() -> Self {
        Agent {
            name: "Agent".to_string(),
            model: "gpt-4".to_string(),
            instructions: "You are a helpful agent.".to_string(),
            tools: Vec::new(),
            tool_choice: None,
            parallel_tool_calls: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub messages: Vec<async_openai::types::ChatCompletionRequestMessage>,
    pub agent: Option<Agent>,
    pub context_variables: HashMap<String, String>,
}

pub struct ToolRegistry {
    tools: HashMap<String, Tool>,
    functions: HashMap<String, Arc<dyn Fn(Value) -> Value + Send + Sync>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        ToolRegistry {
            tools: HashMap::new(),
            functions: HashMap::new(),
        }
    }

    pub fn register_tool(
        &mut self,
        name: &str,
        description: &str,
        parameters: Value,
        function: Box<dyn Fn(Value) -> Value + Send + Sync>,
    ) {
        let tool = Tool::new(name, description, parameters);
        self.tools.insert(name.to_string(), tool);
        self.functions.insert(name.to_string(), Arc::from(function));
    }

    pub fn get_function(&self, name: &str) -> Option<Arc<dyn Fn(Value) -> Value + Send + Sync>> {
        self.functions.get(name).cloned()
    }

    pub fn get_tool(&self, name: &str) -> Option<&Tool> {
        self.tools.get(name)
    }
}
