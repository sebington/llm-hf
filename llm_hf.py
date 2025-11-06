import llm
import os
import json
import click
from openai import OpenAI, AsyncOpenAI
from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator


def refresh_models():
    """Fetch models from HuggingFace API and cache them."""
    user_dir = llm.user_dir()
    hf_models = user_dir / "hf_models.json"
    
    # Try to get API key
    api_key = llm.get_key("", "hf", "HF_TOKEN")
    if not api_key:
        api_key = os.environ.get("HF_API_KEY")
    
    if not api_key:
        raise click.ClickException(
            "You must set the 'hf' key or the HF_TOKEN environment variable."
        )
    
    try:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key,
        )
        models_response = client.models.list()
        model_list = sorted([model.id for model in models_response.data])
        
        # Cache the models
        hf_models.write_text(json.dumps({"models": model_list}, indent=2))
        return model_list
    except Exception as e:
        raise click.ClickException(f"Failed to fetch models from Hugging Face API: {e}")


def get_model_details():
    """Get cached models or fetch them if not cached."""
    user_dir = llm.user_dir()
    hf_models = user_dir / "hf_models.json"
    
    if hf_models.exists():
        models = json.loads(hf_models.read_text())
        return models.get("models", [])
    elif llm.get_key("", "hf", "HF_TOKEN") or os.environ.get("HF_API_KEY"):
        try:
            return refresh_models()
        except click.ClickException:
            return []
    return []


@llm.hookimpl
def register_models(register):
    """Register Hugging Face models with the LLM CLI."""
    model_ids = get_model_details()

    for model_id in model_ids:
        register(
            HuggingFaceChat(model_id),
            HuggingFaceChatAsync(model_id),
        )


class _Options(llm.Options):
    provider: Optional[str] = Field(
        description="Specific provider to use (e.g., 'sambanova', 'together', 'fireworks-ai')",
        default=None
    )
    temperature: Optional[float] = Field(
        description=(
            "Controls randomness of responses. Lower values result in more "
            "predictable outputs, while higher values lead to more varied and "
            "creative outputs."
        ),
        ge=0,
        le=2,
        default=None
    )
    max_tokens: Optional[int] = Field(
        description="The maximum number of tokens to generate in the completion.",
        ge=1,
        default=None
    )
    top_p: Optional[float] = Field(
        description=(
            "Controls randomness by considering the top P probability mass. "
            "A lower value makes the output more focused but less creative."
        ),
        ge=0,
        le=1,
        default=None
    )


class _Shared:
    """Shared functionality between sync and async models."""
    
    def __init__(self, hf_model_id):
        self.model_id = hf_model_id
        self.hf_model_id = hf_model_id

    def build_messages(self, prompt, conversation):
        """Build messages array for the API call."""
        messages = []
        
        if not conversation:
            if prompt.system:
                messages.append({"role": "system", "content": prompt.system})
            messages.append({"role": "user", "content": prompt.prompt})
            return messages

        # Handle conversation with system prompt tracking
        current_system = None
        for prev_response in conversation.responses:
            if (
                prev_response.prompt.system
                and prev_response.prompt.system != current_system
            ):
                messages.append(
                    {"role": "system", "content": prev_response.prompt.system}
                )
                current_system = prev_response.prompt.system

            messages.append(
                {"role": "user", "content": prev_response.prompt.prompt}
            )
            messages.append(
                {"role": "assistant", "content": prev_response.text_or_raise()}
            )
        
        if prompt.system and current_system != prompt.system:
            messages.append({"role": "system", "content": prompt.system})

        messages.append({"role": "user", "content": prompt.prompt})
        return messages

    def build_api_params(self, prompt, messages, stream):
        """Build parameters for API call."""
        model = self.hf_model_id
        
        # Append provider suffix if specified
        if prompt.options.provider:
            model = f"{model}:{prompt.options.provider}"

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

        return api_params

    def set_usage(self, response, usage):
        """Set usage information on response."""
        if usage:
            response.set_usage(
                input=getattr(usage, 'prompt_tokens', None),
                output=getattr(usage, 'completion_tokens', None),
                details={
                    "prompt_tokens": getattr(usage, 'prompt_tokens', None),
                    "completion_tokens": getattr(usage, 'completion_tokens', None),
                    "total_tokens": getattr(usage, 'total_tokens', None),
                }
            )


class HuggingFaceChat(llm.Model, _Shared):
    """
    Model for accessing Hugging Face Inference Providers via OpenAI-compatible API.

    Usage:
        llm -m meta-llama/Llama-3.1-8B-Instruct "Hello!"
        llm -m meta-llama/Llama-3.1-8B-Instruct -o provider sambanova "Hello!"
    """

    can_stream = True
    needs_key = "hf"
    key_env_var = "HF_TOKEN"
    Options = _Options

    def execute(self, prompt, stream, response, conversation):
        """Execute a prompt against the Hugging Face API."""
        api_key = self.get_key()
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key
        )

        messages = self.build_messages(prompt, conversation)
        api_params = self.build_api_params(prompt, messages, stream)

        completion = client.chat.completions.create(**api_params)
        usage = None

        try:
            if stream:
                for chunk in completion:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                    # Check for usage in streaming response
                    if hasattr(chunk, 'usage') and chunk.usage:
                        usage = chunk.usage
            else:
                if completion.choices:
                    yield completion.choices[0].message.content
                usage = getattr(completion, 'usage', None)
        except AttributeError as e:
            if "NoneType" in str(e):
                if completion.choices:
                    yield completion.choices[0].message.content
            else:
                raise e
        finally:
            if usage:
                self.set_usage(response, usage)


class HuggingFaceChatAsync(llm.AsyncModel, _Shared):
    """Async version of HuggingFaceChat model."""

    can_stream = True
    needs_key = "hf"
    key_env_var = "HF_TOKEN"
    Options = _Options

    async def execute(self, prompt, stream, response, conversation):
        """Execute a prompt against the Hugging Face API asynchronously."""
        api_key = self.get_key()
        client = AsyncOpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key
        )

        messages = self.build_messages(prompt, conversation)
        api_params = self.build_api_params(prompt, messages, stream)

        completion = await client.chat.completions.create(**api_params)
        usage = None

        try:
            if stream:
                async for chunk in completion:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                    # Check for usage in streaming response
                    if hasattr(chunk, 'usage') and chunk.usage:
                        usage = chunk.usage
            else:
                if completion.choices:
                    yield completion.choices[0].message.content
                usage = getattr(completion, 'usage', None)
        except AttributeError as e:
            if "NoneType" in str(e):
                if completion.choices:
                    yield completion.choices[0].message.content
            else:
                raise e
        finally:
            if usage:
                self.set_usage(response, usage)


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def hf():
        """Commands relating to the llm-hf plugin"""

    @hf.command()
    def refresh():
        """Refresh the list of available Hugging Face models"""
        user_dir = llm.user_dir()
        hf_models = user_dir / "hf_models.json"
        
        # Get previous models if they exist
        previous = set()
        if hf_models.exists():
            try:
                cached = json.loads(hf_models.read_text())
                previous = set(cached.get("models", []))
            except Exception:
                pass

        try:
            models = refresh_models()
        except click.ClickException as e:
            click.echo(str(e), err=True)
            return

        current = set(models)
        added = current - previous
        removed = previous - current

        if added:
            click.echo(f"Added models ({len(added)}):", err=True)
            for model_id in sorted(added):
                click.echo(f"  + {model_id}", err=True)
        
        if removed:
            click.echo(f"Removed models ({len(removed)}):", err=True)
            for model_id in sorted(removed):
                click.echo(f"  - {model_id}", err=True)

        if added or removed:
            click.echo(f"\nTotal models available: {len(current)}", err=True)
        else:
            click.echo(f"No changes. Total models available: {len(current)}", err=True)

    @hf.command()
    def models():
        """List all available Hugging Face models"""
        model_list = get_model_details()

        if not model_list:
            click.echo(
                "No Hugging Face models cached. Run 'llm hf refresh' to fetch models.",
                err=True
            )
            return

        click.echo(f"Available Hugging Face models ({len(model_list)}):\n")
        for model_id in model_list:
            click.echo(f"  {model_id}")
