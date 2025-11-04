# llm-hf

[LLM](https://llm.datasette.io/) plugin for accessing [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/index) - giving you access to hundreds of open-weight models through a unified API.

## Project Status

This is a personal project. Contributions and feedback are welcome, but please note that support may be limited.

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

You need a Hugging Face "access token" with "Make calls to Inference Providers" permissions.

First create a token at https://huggingface.co/settings/tokens/new?tokenType=fineGrained

Configure the token using the `llm keys set hf` command:
```bash
llm keys set hf
```
```
<paste token here>
```

Alternatively, set the environment variable:

```bash
export HF_TOKEN="your-token-here"
```

## Usage

### List Available Models

The plugin automatically discovers all available models from Hugging Face:

```bash
llm models | grep HuggingFaceChat
```

This will show ~118 models dynamically fetched from the Hugging Face API.

### Basic Usage

Simply use the model name directly:

```bash
llm -m meta-llama/Llama-3.1-8B-Instruct "Explain quantum computing"
```

### Available Models

**Dynamic Discovery**: When you have an `HF_TOKEN` set, the plugin automatically discovers ~116 models from the Hugging Face API, including:

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

**Basic usage:**

```bash
llm -m meta-llama/Llama-3.1-8B-Instruct "Write a poem"
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

All models available through Hugging Face Inference Providers are automatically discoverable via `llm models`. You can also browse them at:
- [Hugging Face Inference Playground](https://huggingface.co/playground)
- [Chat completion models](https://huggingface.co/models?inference_provider=all&sort=trending&other=conversational)

The plugin uses the same model list as the Hugging Face API, so any model shown in the playground should work with this plugin.

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

# Check that models appear
llm models | grep HuggingFaceChat
```

### How Model Registration Works

The plugin automatically fetches all available models from the Hugging Face API at startup. The `get_huggingface_models()` function in `llm_hf.py` queries the `/v1/models` endpoint.

## License

Apache 2.0
