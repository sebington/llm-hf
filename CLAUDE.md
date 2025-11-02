# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project is `llm-hf`, a plugin for the [`llm` CLI tool](https://llm.datasette.io/) that adds support for Hugging Face Inference Providers. The plugin enables users to access hundreds of LLM models through Hugging Face's unified API using an OpenAI-compatible interface.

**Current Status**: Functional plugin with streaming support. Pre-registers popular models from Meta Llama, Mistral, Qwen, DeepSeek, and Google Gemma families.

## Development Commands

### Installation and Testing

```bash
# Install the plugin in editable mode for development
llm install -e .

# Verify plugin installation
llm plugins

# Test the plugin with a model
llm -m <model-id> "test prompt"

# View recent logs
llm logs -n 1

# Uninstall plugin (useful for testing reinstallation)
llm uninstall llm-hf -y
```

### Environment Setup

- Python >=3.8 required (as specified in pyproject.toml)
- Authentication requires a Hugging Face token with "Make calls to Inference Providers" permissions
- Set `HF_TOKEN` or `HF_API_KEY` environment variable for API access
- Create token at: https://huggingface.co/settings/tokens/new?tokenType=fineGrained

### Troubleshooting

If the plugin breaks and prevents `llm` from starting:
```bash
LLM_LOAD_PLUGINS='' llm uninstall llm-hf
```

## Plugin Architecture

### Core Components Required

1. **Entry Point** (`pyproject.toml`):
   - `[project.entry-points.llm]` section registers the plugin
   - Must define plugin name and point to the Python module

2. **Main Module** (`llm_hf.py`):
   - `register_models()` function with `@llm.hookimpl` decorator
   - `HuggingFaceChat` class extending `llm.Model`
   - `Options` inner class extending `llm.Options` (using Pydantic 2)

3. **Model Class Requirements**:
   - `model_id`: Unique identifier for the model
   - `can_stream`: Boolean indicating streaming support
   - `execute()` method: Core logic for API requests and response handling
     - Signature: `def execute(self, prompt, stream, response, conversation)`
     - Must yield tokens for streaming models
     - Access prompt text via `prompt.prompt`
     - Access options via `prompt.options.<option_name>`

### Options and Validation

- Use Pydantic 2 `Field()` for option definitions with descriptions
- Use `@field_validator` decorators for custom validation
- Options automatically logged to database in `options_json`

## Hugging Face API Integration

### API Endpoints

- **OpenAI-compatible**: `https://router.huggingface.co/v1/chat/completions`
- **Native API**: Use `huggingface_hub.InferenceClient`

### Provider Selection

Models can specify a provider by appending it to the model ID:
- Format: `"model-name:provider"` (e.g., `"meta-llama/Llama-3.1-8B-Instruct:sambanova"`)
- Default: Automatic provider selection based on availability

### Authentication

- Token should be passed in `Authorization: Bearer $HF_TOKEN` header
- Or use `api_key` parameter with client libraries

## Key Implementation Notes

- The plugin uses OpenAI Python client with custom `base_url="https://router.huggingface.co/v1"`
- Both streaming and non-streaming modes are supported (`can_stream = True`)
- Response metadata (usage stats) is stored in `response.response_json` for logging
- All prompts and responses are automatically logged to SQLite database
- The `execute()` method builds messages array including system prompt and conversation history
- Provider selection is done by appending `:provider` to model ID in the API call

## Current Implementation

### Dynamic Model Loading (llm_hf.py:8-24)

The plugin automatically fetches all available models from the Hugging Face API at registration time:
- Uses the OpenAI-compatible `/v1/models` endpoint
- Dynamically discovers ~118 models from various providers
- Models are sorted alphabetically for easy browsing

**Fallback Behavior**: If no `HF_TOKEN` or `HF_API_KEY` is set, or if the API call fails, the plugin falls back to registering 13 curated popular models:
- Meta Llama 3.x models (70B, 8B, 3B, 1B variants)
- Mistral models (7B, Mixtral 8x7B, 8x22B)
- Qwen models (72B, Coder 32B)
- DeepSeek V3
- Google Gemma 2 (9B, 27B)

### Viewing Available Models

```bash
# List all available models
llm models | grep HuggingFaceChat

# Count total models
llm models | grep HuggingFaceChat | wc -l
```

### Model Options

Available options (HuggingFaceChat.Options class in llm_hf.py):
- `provider`: Force specific provider (e.g., "sambanova", "together")
- `temperature`: 0.0-2.0 (validated)
- `max_tokens`: >= 1 (validated)
- `top_p`: 0.0-1.0 (validated)

## Common Usage Examples

```bash
# Basic usage with pre-registered model
llm -m meta-llama/Llama-3.1-8B-Instruct "Explain Python decorators"

# With specific provider
llm -m meta-llama/Llama-3.1-8B-Instruct -o provider sambanova "Hello"

# With options
llm -m Qwen/Qwen2.5-Coder-32B-Instruct -o temperature 0.7 -o max_tokens 500 "Write a function"

# With system prompt
llm -m meta-llama/Llama-3.1-8B-Instruct -s "You are a helpful coding assistant" "How to sort in Python?"

# Start a conversation
llm chat -m meta-llama/Llama-3.1-8B-Instruct

# Using any HF model (not pre-registered)
llm -m NousResearch/Hermes-3-Llama-3.1-8B "test prompt"

# Check available models
llm models | grep meta-llama
```

## Reference Documentation

- `developing-a-model-plugin.txt`: Complete tutorial on plugin development
- `HF_Inference_Providers.md`: Hugging Face API documentation and examples
- `test_hf.py`: Working example of HF API usage with OpenAI client
