# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project is `llm-hf`, a plugin for the [`llm` CLI tool](https://llm.datasette.io/) that adds support for Hugging Face Inference Providers. The plugin enables users to access hundreds of LLM models through Hugging Face's unified API using an OpenAI-compatible interface.

**Current Status**: Functional plugin with streaming support. Dynamically fetches ~118 models from Hugging Face Inference Providers API.

## Development Commands

### Installation and Setup

```bash
# Install the plugin in editable mode for development
llm install -e .

# Verify plugin installation
llm plugins

# Uninstall plugin (useful for testing reinstallation or troubleshooting)
llm uninstall llm-hf -y
```

### Testing and Validation

```bash
# Test the plugin with a model (manual testing)
llm -m <model-id> "test prompt"

# Test with specific options
llm -m meta-llama/Llama-3.1-8B-Instruct -o temperature 0.7 "Your prompt"

# Run the provided test script (requires HF_API_KEY)
python test_hf.py

# View recent logs to verify execution
llm logs -n 1

# List all available models
llm models | grep HuggingFaceChat
```

### Development Workflow

When making changes to the plugin:

1. **Edit `llm_hf.py`** directly - the editable install (`llm install -e .`) picks up changes immediately
2. **Test your changes** with `llm -m <model-id> "test prompt"`
3. **Check logs** with `llm logs` to see any errors or debug output
4. **If plugin breaks**: Run `LLM_LOAD_PLUGINS='' llm uninstall llm-hf -y` to safely uninstall

### Environment Setup

- Python >=3.8 required (as specified in pyproject.toml)
- Authentication requires a Hugging Face token with "Make calls to Inference Providers" permissions
- Set the key using: `llm keys set hf your-token-here`
- Or use environment variable: `export HF_TOKEN="your-token-here"` or `export HF_API_KEY="your-token-here"`
- Create token at: https://huggingface.co/settings/tokens/new?tokenType=fineGrained

### Troubleshooting

**Plugin breaks and prevents `llm` from starting:**
```bash
LLM_LOAD_PLUGINS='' llm uninstall llm-hf -y
```

**Plugin doesn't pick up recent changes:**
- The editable install should pick up changes immediately, but if not, try:
```bash
llm uninstall llm-hf -y
llm install -e .
```

**No models appear when running `llm models | grep HuggingFaceChat`:**
- Verify the plugin installed: `llm plugins | grep hf`
- Check environment variable is set: `echo $HF_TOKEN` or `echo $HF_API_KEY`
- Check logs for errors: `llm logs -n 5`

**API authentication errors:**
- Ensure token has "Make calls to Inference Providers" permission
- Create token at: https://huggingface.co/settings/tokens/new?tokenType=fineGrained
- Try `python test_hf.py` to test raw API connectivity

**Model-specific errors:**
- Some models may not be available on all providers
- Try with a different provider: `-o provider sambanova` or `-o provider together`
- Fallback models are always available if API fails

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

### Dynamic Model Loading (llm_hf.py:8-35)

The plugin automatically fetches all available models from the Hugging Face API at registration time:
- Uses the OpenAI-compatible `/v1/models` endpoint
- Dynamically discovers ~118 models from various providers
- Models are sorted alphabetically for easy browsing
- **Requires valid HF_TOKEN or HF_API_KEY**: If no API key is available, no models are registered

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

## Test File

`test_hf.py` is a standalone script that demonstrates basic API connectivity:
- Tests direct OpenAI client connection to Hugging Face API
- Useful for verifying your HF_API_KEY is working before testing the plugin
- Run with: `python test_hf.py`
- Not a unit test suite, but a manual testing utility

## Understanding the Codebase

### File Structure

- **`llm_hf.py`** (main plugin):
  - `get_huggingface_models()` - Fetches available models from HF API
  - `register_models()` - Plugin entry point that registers all models with the LLM CLI
  - `HuggingFaceChat` - Model class that handles API calls and streaming
  - `HuggingFaceChat.Options` - Pydantic model for user-configurable options

- **`pyproject.toml`** - Project metadata and plugin entry point definition
- **`test_hf.py`** - Standalone script for testing raw API connectivity

### Key Flow

1. **Plugin Registration** (`register_models()` hook):
   - Attempts to fetch models from HF API using `get_huggingface_models()`
   - Falls back to hardcoded list if API unavailable
   - Creates `HuggingFaceChat` instance for each model

2. **Execution** (`HuggingFaceChat.execute()` method):
   - Receives user prompt and conversation history
   - Builds message array with system prompt + conversation + current prompt
   - Makes OpenAI-compatible API call to `https://router.huggingface.co/v1`
   - Handles both streaming and non-streaming responses
   - Logs usage statistics for database storage

## Reference Documentation

- `developing-a-model-plugin.txt`: Complete tutorial on plugin development
- `HF_Inference_Providers.md`: Hugging Face API documentation and examples
