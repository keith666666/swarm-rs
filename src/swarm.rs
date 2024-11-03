use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionMessageToolCall, ChatCompletionRequestAssistantMessage,
        ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage,
        ChatCompletionRequestToolMessage, ChatCompletionRequestToolMessageContent,
        ChatCompletionResponseMessage, ChatCompletionTool, ChatCompletionToolArgs,
        ChatCompletionToolType, CreateChatCompletionRequestArgs, FunctionObjectArgs,
    },
    Client,
};
use serde_json::Value;
use std::collections::HashMap;

use crate::types::{Agent, Response, ToolRegistry, ToolResult};

// Main struct for managing AI swarm interactions
pub struct Swarm {
    client: Client<OpenAIConfig>,
    registry: ToolRegistry,
}

impl Swarm {
    // Creates a new Swarm instance with optional OpenAI client
    pub fn new(client: Option<Client<OpenAIConfig>>) -> Self {
        Swarm {
            client: client.unwrap_or_default(),
            registry: ToolRegistry::new(),
        }
    }

    // Registers a new tool with the swarm
    pub fn register_tool(
        &mut self,
        name: &str,
        description: &str,
        parameters: Value,
        function: Box<dyn Fn(Value) -> Value + Send + Sync>,
    ) {
        self.registry
            .register_tool(name, description, parameters, function);
    }

    // Gets chat completion from OpenAI API
    pub async fn get_chat_completion(
        &self,
        agent: &Agent,
        history: &[ChatCompletionRequestMessage],
    ) -> Result<ChatCompletionResponseMessage, Box<dyn std::error::Error>> {
        // 1. Convert agent tools to ChatCompletionTool format
        let tools: Vec<ChatCompletionTool> = agent
            .tools
            .iter()
            .map(|f| {
                ChatCompletionToolArgs::default()
                    .r#type(ChatCompletionToolType::Function)
                    .function(
                        FunctionObjectArgs::default()
                            .name(&f.name)
                            .description(&f.description)
                            .parameters(f.parameters.clone())
                            .build()
                            .unwrap(),
                    )
                    .build()
                    .unwrap()
            })
            .collect();

        // 2. Build chat completion request based on tools presence
        let request = if tools.is_empty() {
            CreateChatCompletionRequestArgs::default()
                .max_tokens(512u32)
                .model(agent.model.clone())
                .messages(history.to_vec())
                .build()?
        } else {
            CreateChatCompletionRequestArgs::default()
                .max_tokens(512u32)
                .model(agent.model.clone())
                .messages(history.to_vec())
                .tools(tools)
                .build()?
        };

        // 3. Send request and return first choice message
        let response_message = self
            .client
            .chat()
            .create(request)
            .await?
            .choices
            .first()
            .unwrap()
            .message
            .clone();
        Ok(response_message)
    }

    // Processes function result into ToolResult format
    fn handle_function_result(&self, raw_result: Value, debug: bool) -> ToolResult {
        // 1. Handle object with 'value' key
        match raw_result {
            Value::Object(obj) if obj.contains_key("value") => {
                let obj_clone = obj.clone();
                serde_json::from_value(Value::Object(obj)).unwrap_or_else(|e| {
                    if debug {
                        println!("Error parsing Result: {}", e);
                    }
                    ToolResult {
                        value: obj_clone["value"].as_str().unwrap_or("").to_string(),
                        agent: None,
                        context_variables: HashMap::new(),
                    }
                })
            }
            // 2. Handle object with 'assistant' key
            Value::Object(obj) if obj.contains_key("assistant") => ToolResult {
                value: serde_json::to_string(&obj).unwrap(),
                agent: Some(serde_json::from_value(Value::Object(obj)).unwrap()),
                context_variables: HashMap::new(),
            },
            // 3. Handle other cases
            _ => {
                let value = raw_result.as_str().map(String::from).unwrap_or_else(|| {
                    if debug {
                        println!("Failed to cast response to string: {:?}", raw_result);
                    }
                    raw_result.to_string()
                });
                ToolResult {
                    value,
                    agent: None,
                    context_variables: HashMap::new(),
                }
            }
        }
    }

    // Processes tool calls and returns response
    fn handle_tool_calls(
        &self,
        tool_calls: &Vec<ChatCompletionMessageToolCall>,
        context_variables: &mut HashMap<String, String>,
        debug: bool,
    ) -> Response {
        let mut partial_response = Response {
            messages: Vec::new(),
            agent: None,
            context_variables: HashMap::new(),
        };

        // Process each tool call sequentially
        for tool_call in tool_calls {
            let name = &tool_call.function.name;

            // 1. Get function from registry
            if let Some(func) = self.registry.get_function(name) {
                // 2. Parse arguments
                let args: Value = serde_json::from_str(&tool_call.function.arguments)
                    .expect("Failed to parse arguments");

                if debug {
                    println!("processing tool call: {} with arguments {:?}", name, args);
                }

                // 3. Add context variables to arguments
                let mut args_with_context = args.as_object().unwrap().clone();
                args_with_context.insert(
                    "context_variables".to_string(),
                    serde_json::to_value(&context_variables).unwrap(),
                );

                // 4. Execute function and process result
                let raw_result = func(Value::Object(args_with_context));
                if debug {
                    println!("raw result: {:?}", raw_result);
                }
                let result = self.handle_function_result(raw_result, debug);
                if debug {
                    println!("tool result: {:?}", result);
                }

                // 5. Update response with results
                partial_response
                    .messages
                    .push(ChatCompletionRequestMessage::Tool(
                        ChatCompletionRequestToolMessage {
                            content: ChatCompletionRequestToolMessageContent::Text(result.value),
                            tool_call_id: tool_call.id.clone(),
                        },
                    ));

                partial_response
                    .context_variables
                    .extend(result.context_variables);
                if let Some(agent) = result.agent {
                    partial_response.agent = Some(agent);
                }
            } else {
                if debug {
                    println!("tool {} not found in function map.", name);
                }
                partial_response
                    .messages
                    .push(ChatCompletionRequestMessage::Tool(
                        ChatCompletionRequestToolMessage {
                            content: ChatCompletionRequestToolMessageContent::Text(format!(
                                "error: tool {} not found.",
                                name
                            )),
                            tool_call_id: tool_call.id.clone(),
                        },
                    ));
            }
        }

        partial_response
    }

    // Main execution loop for the swarm
    pub async fn run(
        &self,
        agent: Agent,
        messages: Vec<ChatCompletionRequestMessage>,
        context_variables: Option<HashMap<String, String>>,
        model_override: Option<String>,
        stream: bool,
        debug: bool,
        max_turns: Option<usize>,
        execute_tools: bool,
    ) -> Result<Response, Box<dyn std::error::Error>> {
        // 1. Handle streaming request
        if stream {
            return self.run_and_stream(
                agent,
                messages,
                context_variables,
                model_override,
                debug,
                max_turns,
                execute_tools,
            );
        }

        // 2. Initialize execution context
        let mut active_agent = agent;
        let mut context_variables = context_variables.unwrap_or_default();
        let mut history = messages.clone();
        let init_len = messages.len();
        let max_turns = max_turns.unwrap_or(usize::MAX);

        // 3. Main execution loop
        while history.len() - init_len < max_turns {
            // 3.1 Get completion
            let completion: ChatCompletionResponseMessage =
                self.get_chat_completion(&active_agent, &history).await?;

            if debug {
                println!("Received completion: {:?}", completion);
            }

            // 3.2 Add assistant message to history
            history.push(ChatCompletionRequestMessage::Assistant(
                ChatCompletionRequestAssistantMessage {
                    content: completion
                        .content
                        .map(ChatCompletionRequestAssistantMessageContent::Text),
                    tool_calls: completion.tool_calls.clone(),
                    refusal: completion.refusal,
                    ..Default::default()
                },
            ));

            // 3.3 Break if no tool calls
            if completion.tool_calls.is_none() {
                if debug {
                    println!("Ending turn.");
                }
                break;
            }

            // 3.4 Handle tool calls and update state
            let partial_response = self.handle_tool_calls(
                &completion.tool_calls.unwrap(),
                &mut context_variables,
                debug,
            );

            history.extend(partial_response.messages);
            context_variables.extend(partial_response.context_variables);
            if let Some(new_agent) = partial_response.agent {
                active_agent = new_agent;
            }
        }

        // 4. Return final response
        Ok(Response {
            messages: history[init_len..].to_vec(),
            agent: Some(active_agent),
            context_variables,
        })
    }

    // Placeholder for streaming implementation
    fn run_and_stream(
        &self,
        agent: Agent,
        messages: Vec<ChatCompletionRequestMessage>,
        context_variables: Option<HashMap<String, String>>,
        model_override: Option<String>,
        debug: bool,
        max_turns: Option<usize>,
        execute_tools: bool,
    ) -> Result<Response, Box<dyn std::error::Error>> {
        unimplemented!()
    }
}
