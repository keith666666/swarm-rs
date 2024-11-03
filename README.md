# swarm-rs 🦀 (experimental, educational)

A Rust implementation of [OpenAI's Swarm framework](https://github.com/openai/swarm) for orchestrating AI agents that can use tools and work together to accomplish tasks.

> **Note**: This is an experimental educational project exploring ergonomic interfaces for multi-agent systems in Rust. It is not intended for production use.

## Features

- 🤖 Create AI agents with custom tools and instructions
- 🛠️ Define custom tools that agents can use
- 🔗 Built on top of OpenAI's Chat API
- 📦 Easy to integrate into Rust applications

## Quick Start

```rust
use swarm_rs::{swarm::Swarm, types::{Agent, Tool}};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new swarm
    let mut swarm = Swarm::new(None);

    // Register a custom tool
    swarm.register_tool(
        "get_weather",
        "Get the weather for a location",
        json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }),
        Box::new(|args| {
            json!({"temp": 72, "location": args["location"]})
        }),
    );

    // Create an agent
    let agent = Agent {
        name: "Weather Agent".into(),
        model: "gpt-4".into(),
        instructions: "You help with weather information.".into(),
        tools: vec![/* Add tools here */],
        tool_choice: None,
        parallel_tool_calls: true,
    };

    // Run the agent
    let response = swarm.run(
        agent,
        messages,
        None,
        None,
        false,
        true,
        Some(10),
        true,
    ).await?;

    Ok(())
}
```

## Examples & Usage

The project includes example code demonstrating different use cases:

### Available Examples

- `function_calling.rs`: Shows how to create an agent that uses a weather tool to get temperature information for different locations

### Running the Examples

1. Set your OpenAI API key:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

2. Run an example:

```bash
# Run the function calling example
cargo run --example function_calling
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
