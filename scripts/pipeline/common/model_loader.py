"""Load a Hugging Face causal LM in the configured dtype, format prompts with its chat template, and stop batched
generation once each row's answer line is complete."""

from __future__ import annotations

import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria

from scripts.pipeline.common.config import ModelConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


GATED_PREFIXES = ("meta-llama/", "google/gemma")


def check_hf_access(model_id: str) -> None:
    """Stop early when a gated model (Llama, Gemma) has no Hugging Face token: HF_TOKEN or a saved login."""
    if model_id.startswith(GATED_PREFIXES):
        from huggingface_hub import get_token

        if not get_token():
            raise SystemExit(f"{model_id} is gated: set HF_TOKEN (or run `huggingface-cli login`) after accepting "
                             "its licence on the Hugging Face Hub.")


def load_model(
    cfg: ModelConfig,
    device_map: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load HuggingFace model + tokenizer from config (cache: the HF_HOME environment variable)."""
    dtype = DTYPE_MAP.get(cfg.dtype, torch.bfloat16)
    hf_token = os.environ.get("HF_TOKEN", None) or None

    print(f"Loading model: {cfg.model_id}")
    print(f"  dtype={cfg.dtype}")

    check_hf_access(cfg.model_id)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        device_map=device_map,
        token=hf_token,
    )
    model.eval()
    return model, tokenizer


def get_layers(model, cfg: ModelConfig):
    """Navigate to the layer list using the layer_accessor path."""
    obj = model
    for attr in cfg.layer_accessor.split("."):
        obj = getattr(obj, attr)
    return obj


def chat_messages(cfg: Any, user_text: str) -> list[dict[str, str]]:
    """One user turn; the system prompt is its own turn, or is prepended to the user text if the model has no
    system role."""
    messages = []
    effective_user = user_text
    if cfg.system_prompt and cfg.has_system_role:
        messages.append({"role": "system", "content": cfg.system_prompt})
    elif cfg.system_prompt and not cfg.has_system_role:
        effective_user = f"{cfg.system_prompt}\n\n{user_text}"
    messages.append({"role": "user", "content": effective_user})
    return messages


def format_chat(tokenizer: Any, cfg: Any, user_text: str) -> str:
    """Chat-templated prompt text, ending with the assistant generation prompt."""
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if cfg.enable_thinking is not None:
        kwargs["enable_thinking"] = cfg.enable_thinking
    return tokenizer.apply_chat_template(chat_messages(cfg, user_text), **kwargs)


class ChatFormatter:
    """Build chat-templated input_ids respecting per-model quirks."""

    def __init__(self, tokenizer: AutoTokenizer, cfg: ModelConfig):
        self.tokenizer = tokenizer
        self.cfg = cfg

    def format(self, user_text: str) -> torch.Tensor:
        kwargs = {"add_generation_prompt": True, "return_tensors": "pt"}
        if self.cfg.enable_thinking is not None:
            kwargs["enable_thinking"] = self.cfg.enable_thinking

        ids = self.tokenizer.apply_chat_template(chat_messages(self.cfg, user_text), **kwargs)
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        return ids.to(DEVICE)

    def format_multiturn(self, turns: list[dict]) -> torch.Tensor:
        """Format a multi-turn conversation: [{"role": ..., "content": ...}, ...]."""
        messages = []
        if self.cfg.system_prompt and self.cfg.has_system_role:
            messages.append({"role": "system", "content": self.cfg.system_prompt})

        for turn in turns:
            if turn["role"] == "user" and not messages and self.cfg.system_prompt and not self.cfg.has_system_role:
                messages.append({"role": "user", "content": f"{self.cfg.system_prompt}\n\n{turn['content']}"})
            else:
                messages.append(turn)

        kwargs = {"add_generation_prompt": True, "return_tensors": "pt"}
        if self.cfg.enable_thinking is not None:
            kwargs["enable_thinking"] = self.cfg.enable_thinking

        ids = self.tokenizer.apply_chat_template(messages, **kwargs)
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        return ids.to(DEVICE)


class AnswerLineStopping(StoppingCriteria):
    """Finish a row once its answer line is complete; saves most decode steps.

    The main-task prompt asks for reasoning and then a single final line
    ``Final answer: <x>``. Once a row has emitted that line plus a newline (or a
    safety margin of characters after it) nothing further can change the parsed
    answer, so the row is marked done. Returning a per-row tensor lets
    `generate` pad finished rows instead of running the whole batch to the cap.
    """

    def __init__(self, tokenizer: Any, prompt_width: int, stride: int = 8, margin: int = 60):
        self.tokenizer = tokenizer
        self.prompt_width = prompt_width
        self.stride = max(1, stride)
        self.margin = margin
        self.step = 0
        self.done: list[bool] = []

    def __call__(self, input_ids: torch.Tensor, scores: Any, **kwargs: Any) -> torch.Tensor:
        batch = input_ids.shape[0]
        if not self.done:
            self.done = [False] * batch
        self.step += 1
        if self.step % self.stride == 0:
            texts = self.tokenizer.batch_decode(
                input_ids[:, self.prompt_width :], skip_special_tokens=True
            )
            for i, text in enumerate(texts):
                if self.done[i]:
                    continue
                low = text.lower()
                idx = low.rfind("final answer:")
                if idx >= 0 and ("\n" in text[idx + 13 :] or len(text) - idx > self.margin):
                    self.done[i] = True
        return torch.tensor(self.done, dtype=torch.bool, device=input_ids.device)
