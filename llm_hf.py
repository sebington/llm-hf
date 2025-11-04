import llm
import os
import click
from openai import OpenAI
from typing import Optional
from pydantic import Field, field_validator


def get_huggingface_models(api_key=None):
    """Fetch available models from Hugging Face API."""
    try:
        if not api_key:
            # Try to get from stored key, then environment variables
            api_key = llm.get_key("", "hf", "HF_TOKEN")
            if not api_key:
                api_key = os.environ.get("HF_API_KEY")

        if not api_key:
            # No token available, return empty list
            return []

        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key,
        )
        models_response = client.models.list()
        return sorted([model.id for model in models_response.data])
    except Exception:
        # If API call fails, return empty list
        return []


@llm.hookimpl
def register_models(register):
    """Register Hugging Face models with the LLM CLI."""
    # Try to fetch models dynamically from the API
    model_ids = get_huggingface_models()

    # Only register models if token is available
    for model_id in model_ids:
        register(HuggingFaceChat(model_id))


class HuggingFaceChat(llm.Model):
    """
    Model for accessing Hugging Face Inference Providers via OpenAI-compatible API.

    Usage:
        llm -m meta-llama/Llama-3.1-8B-Instruct "Hello!"
        llm -m meta-llama/Llama-3.1-8B-Instruct -o provider sambanova "Hello!"
    """

    can_stream = True
    needs_key = "hf"
    key_env_var = "HF_TOKEN"

    class Options(llm.Options):
        provider: Optional[str] = Field(
            description="Specific provider to use (e.g., 'sambanova', 'together', 'fireworks-ai')",
            default=None
        )
        temperature: Optional[float] = Field(
            description="Sampling temperature (0.0 to 2.0)",
            default=None
        )
        max_tokens: Optional[int] = Field(
            description="Maximum number of tokens to generate",
            default=None
        )
        top_p: Optional[float] = Field(
            description="Nucleus sampling parameter",
            default=None
        )

        @field_validator("temperature")
        def validate_temperature(cls, temperature):
            if temperature is None:
                return None
            if not 0 <= temperature <= 2:
                raise ValueError("temperature must be between 0 and 2")
            return temperature

        @field_validator("top_p")
        def validate_top_p(cls, top_p):
            if top_p is None:
                return None
            if not 0 <= top_p <= 1:
                raise ValueError("top_p must be between 0 and 1")
            return top_p

        @field_validator("max_tokens")
        def validate_max_tokens(cls, max_tokens):
            if max_tokens is None:
                return None
            if max_tokens < 1:
                raise ValueError("max_tokens must be at least 1")
            return max_tokens

    def __init__(self, hf_model_id, description=None):
        self.model_id = hf_model_id
        self.hf_model_id = hf_model_id
        if description:
            self.__class__.__doc__ = description

    def _get_client(self, api_key):
        """Initialize OpenAI client for Hugging Face API."""
        return OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key
        )

    def execute(self, prompt, stream, response, conversation):
        """Execute a prompt against the Hugging Face API."""
        api_key = self.get_key()
        client = self._get_client(api_key)

        # Use the model ID from the instance
        model = self.hf_model_id

        # Append provider suffix if specified
        if prompt.options.provider:
            model = f"{model}:{prompt.options.provider}"

        # Build messages array
        messages = []

        # Add system prompt if provided
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})

        # Add conversation history if available
        if conversation:
            for prev_response in conversation.responses:
                messages.append({"role": "user", "content": prev_response.prompt.prompt})
                messages.append({"role": "assistant", "content": prev_response.text()})

        # Add current prompt
        messages.append({"role": "user", "content": prompt.prompt})

        # Build API call parameters
        api_params = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        # Add optional parameters
        if prompt.options.temperature is not None:
            api_params["temperature"] = prompt.options.temperature
        if prompt.options.max_tokens is not None:
            api_params["max_tokens"] = prompt.options.max_tokens
        if prompt.options.top_p is not None:
            api_params["top_p"] = prompt.options.top_p

        # Make API call
        completion = client.chat.completions.create(**api_params)

        if stream:
            # Streaming mode
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        else:
            # Non-streaming mode
            if completion.choices:
                yield completion.choices[0].message.content

        # Store metadata in response_json for logging
        if hasattr(completion, 'usage') and completion.usage:
            response.response_json = {
                "usage": {
                    "prompt_tokens": getattr(completion.usage, 'prompt_tokens', None),
                    "completion_tokens": getattr(completion.usage, 'completion_tokens', None),
                    "total_tokens": getattr(completion.usage, 'total_tokens', None),
                }
            }


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def hf():
        """Commands relating to the llm-hf plugin"""

    @hf.command()
    def refresh():
        """Refresh the list of available Hugging Face models"""
        # Try to get the token
        api_key = None
        try:
            api_key = llm.get_key("", "hf", "HF_TOKEN")
            if not api_key:
                api_key = os.environ.get("HF_API_KEY")
        except Exception:
            pass

        if not api_key:
            click.echo("Error: No token found. Set one with 'llm keys set hf' or HF_TOKEN environment variable.", err=True)
            return

        # Get current models
        click.echo("Fetching models from Hugging Face API...")
        current_models = set(get_huggingface_models(api_key))

        if not current_models:
            click.echo("Error: Failed to fetch models from Hugging Face API.", err=True)
            return

        # Get previously cached models from the plugin registry
        previous_models = set()
        for model in llm.get_models():
            if isinstance(model, HuggingFaceChat):
                previous_models.add(model.model_id)

        # Calculate differences
        added = current_models - previous_models
        removed = previous_models - current_models

        # Report results
        if added or removed:
            if added:
                click.echo(f"\nAdded models ({len(added)}):")
                for model_id in sorted(added):
                    click.echo(f"  + {model_id}")
            if removed:
                click.echo(f"\nRemoved models ({len(removed)}):")
                for model_id in sorted(removed):
                    click.echo(f"  - {model_id}")
            click.echo(f"\nTotal models available: {len(current_models)}")
        else:
            click.echo(f"No changes. Total models available: {len(current_models)}")

    @hf.command()
    def models():
        """List all available Hugging Face models"""
        # Get all registered HuggingFace models
        hf_models = []
        for model in llm.get_models():
            if isinstance(model, HuggingFaceChat):
                hf_models.append(model.model_id)

        if not hf_models:
            click.echo("No Hugging Face models registered. Run 'llm hf refresh' or set your HF_TOKEN.", err=True)
            return

        click.echo(f"Available Hugging Face models ({len(hf_models)}):\n")
        for model_id in sorted(hf_models):
            click.echo(f"  {model_id}")
