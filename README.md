# llm-hf

[LLM](https://llm.datasette.io/) plugin for accessing [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/index) - giving you access to hundreds of open-weight models through a unified API.

## Project Status

This is a personal project that is still in development. Contributions and feedback are welcome, but please note that support may be limited.

## Installation

Make sure [LLM](https://llm.datasette.io/) is installed on your machine.

Then clone this repository:

```bash
git clone https://github.com/sebington/llm-hf.git
```

```bash
cd llm-hf
```

```bash
llm install -e .
```

## Configuration

You need a Hugging Face access token with "Make calls to Inference Providers" permissions.

First, create a token at https://huggingface.co/settings/tokens/new?tokenType=fineGrained

Then configure it using one of these methods:

**Option 1: Using llm keys (recommended)**
```bash
llm keys set hf
```
```
<paste token here>
```

**Option 2: Using environment variable**
```bash
export HF_TOKEN="your-token-here"
```

**Note:** For backward compatibility, `HF_API_KEY` is also supported, but `HF_TOKEN` is recommended as it matches Hugging Face's official naming convention.

## Usage

### Plugin Commands

The plugin provides an `llm hf` command group for managing Hugging Face models:

```bash
# List all available Hugging Face models
llm hf models

# Refresh the model list from the API and see what changed
llm hf refresh
```

The `llm hf refresh` command is particularly useful to:
- Check if new models have been added to Hugging Face Inference Providers
- See which models have been removed from the service
- Verify your token is working correctly

**Alternative way to list models:**

```bash
llm models | grep HuggingFaceChat
```

Both methods show ~116 models dynamically fetched from the Hugging Face API.

### Basic Usage

Simply use the model name directly:

```bash
llm -m meta-llama/Llama-3.1-8B-Instruct "Explain quantum computing"
```

### Available Models

The plugin automatically discovers ~116 models from the Hugging Face API (when you have an `HF_TOKEN` set), including:

- Meta Llama models (various sizes and versions)
- Mistral and Mixtral models
- Qwen and QwQ models
- DeepSeek models
- Google Gemma models
- Cohere Command and Aya models
- NousResearch Hermes models
- MiniMax models
- And many more!

### Examples

**Using plugin commands:**

```bash
# List available models
llm hf models

# Check for model updates
llm hf refresh
```

**Basic usage:**

```bash
llm -m meta-llama/Llama-3.1-8B-Instruct "Write a poem about translation"
```

**With options:**

```bash
llm -m Qwen/Qwen2.5-Coder-32B-Instruct \
  -o temperature 0.7 \
  -o max_tokens 500 \
  "Write a Python function to sort a list"
```

**With a specific provider:**

```bash
llm -m meta-llama/Llama-3.1-8B-Instruct \
  -o provider sambanova \
  "What is the capital of France?"
```

**In a conversation:**

```bash
llm chat -m meta-llama/Llama-3.1-8B-Instruct
```

**With system prompt:**

```bash
llm -m Qwen/Qwen2.5-Coder-32B-Instruct \
  -s "You are a helpful coding assistant" \
  "How do I sort a list in Python?"
```

### Available Options

- `provider` (optional): Specify a provider (e.g., `sambanova`, `together`, `fireworks-ai`, `groq`)
  - If not specified, Hugging Face automatically selects the best available provider
  - Note: Not all providers support all models
- `temperature`: Sampling temperature between 0.0 and 2.0 (default: provider default)
- `max_tokens`: Maximum number of tokens to generate (default: provider default)
- `top_p`: Nucleus sampling parameter between 0.0 and 1.0 (default: provider default)

### Supported Providers

When using the `provider` option, you can choose from:

- `sambanova`
- `together`
- `fireworks-ai`
- `groq`
- `cerebras`
- `hyperbolic`
- `featherless-ai`
- `nebius`
- `novita`
- And more!

**Note:** Each provider supports different models. If you request a model from a provider that doesn't support it, you'll get an error message.

### Finding More Models

All models available through Hugging Face Inference Providers are automatically discoverable:

```bash
# Recommended: Use the plugin command
llm hf models

# Or use the global command with filtering
llm models | grep HuggingFaceChat
```

You can also browse available models at:
- [Hugging Face Inference Playground](https://huggingface.co/playground)
- [Chat completion models](https://huggingface.co/models?inference_provider=all&sort=trending&other=conversational)

The plugin uses the same model list as the Hugging Face API, so any model shown in the playground should work with this plugin. Run `llm hf refresh` periodically to update your local model list.

## Logging

All prompts and responses are automatically logged. View logs with:

```bash
llm logs
```

View the most recent entry:

```bash
llm logs -n 1
```

## Development

To set up this plugin for development:

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-hf
cd llm-hf

# Install in editable mode
llm install -e .

# Verify installation
llm plugins

# Check that models appear (either method works)
llm hf models
# or
llm models | grep HuggingFaceChat
```

### How Model Registration Works

The plugin automatically fetches all available models from the Hugging Face API at startup. The `get_huggingface_models()` function in `llm_hf.py` queries the `/v1/models` endpoint.

## License

Apache 2.0
