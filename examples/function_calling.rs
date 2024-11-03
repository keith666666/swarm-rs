use async_openai::types::{
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage,
    ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
};
use serde_json::json;
use swarm_rs::{
    swarm::Swarm,
    types::{Agent, Tool},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Swarm Simulation Started!");

    // 1. Setup swarm and tools
    let mut swarm = Swarm::new(None);

    // Register weather tool with mock implementation
    swarm.register_tool(
        "get_weather",
        "Get the weather for a given location",
        json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get weather for"
                }
            },
            "required": ["location"]
        }),
        Box::new(|args| {
            let location = args["location"].as_str().unwrap_or("Unknown");
            println!("Getting weather for {}", location);
            json!({"temp": 67, "unit": "F", "location": location})
        }),
    );

    // 2. Configure weather agent
    let agent = Agent {
        name: "Weather Agent".to_string(),
        model: "gpt-4".to_string(),
        instructions:
            "You are a helpful weather assistant. Use the weather tool to check conditions."
                .to_string(),
        tools: vec![Tool::new(
            "get_weather",
            "Get the weather for a given location",
            json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get weather for"
                    }
                },
                "required": ["location"]
            }),
        )],
        tool_choice: None,
        parallel_tool_calls: true,
    };

    // 3. Prepare conversation
    let user_prompt = "What's the weather like in Boston and Atlanta?";
    let messages: Vec<ChatCompletionRequestMessage> = vec![ChatCompletionRequestMessage::User(
        ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Text(user_prompt.into()),
            name: None,
        },
    )];

    // 4. Execute and handle response
    let max_turns = 10;
    let response = swarm
        .run(
            agent,
            messages,
            None,
            None,
            false,
            true, // Enable debug output
            Some(max_turns),
            true, // Enable tool execution
        )
        .await?;

    // 5. Process and display results
    if let Some(last_message) = response.messages.last() {
        match last_message {
            ChatCompletionRequestMessage::Assistant(msg) => {
                if let Some(content) = &msg.content {
                    match content {
                        ChatCompletionRequestAssistantMessageContent::Text(text) => {
                            println!("Assistant's response: {}", text)
                        }
                        _ => println!("No text response received"),
                    }
                }
            }
            _ => println!("Received non-assistant message"),
        }
    }

    Ok(())
}
