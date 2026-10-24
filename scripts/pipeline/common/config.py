"""Typed configuration loaded from the YAML files in data/prompts/ (models, datasets, prompts, judges)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


# ── Config dataclasses ────────────────────────────────────────────────────────

@dataclass
class DecodingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 2048


@dataclass
class ModelConfig:
    model_id: str
    short_name: str
    has_system_role: bool
    system_prompt: str | None
    canonical_decoding: DecodingParams
    layer_accessor: str = "model.model.layers"
    dtype: str = "bfloat16"
    enable_thinking: bool | None = None


@dataclass
class DatasetConfig:
    name: str
    hf_path: str
    hf_name: str | None
    split: str
    answer_type: str           # "span" or "mcq"
    max_answer_len: int = 80
    pilot_size: int | None = None
    main_size: int | None = None
    filter: dict = field(default_factory=dict)


@dataclass
class PromptTemplate:
    id: str
    template: str


@dataclass
class JudgeConfig:
    provider: str
    model_id: str
    base_url: str
    max_tokens: int
    temperature: float
    max_workers: int
    max_retries: int
    retry_base_delay: float


@dataclass
class JudgeViewConfig:
    id: str
    prompt: str


@dataclass
class ProjectConfig:
    models: dict[str, ModelConfig]
    datasets: dict[str, DatasetConfig]
    prompt_templates: dict[str, PromptTemplate]
    assistant_prefill: str | None
    judge: JudgeConfig
    judge_views: dict[str, JudgeViewConfig]


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_configs(path: Path) -> dict[str, ModelConfig]:
    raw = _load_yaml(path)
    configs = {}
    for key, val in raw.items():
        dec = val.get("canonical_decoding") or {}
        configs[key] = ModelConfig(
            model_id=val["model_id"],
            short_name=val.get("short_name", key),
            has_system_role=val["has_system_role"],
            system_prompt=val.get("system_prompt"),
            canonical_decoding=DecodingParams(
                temperature=dec.get("temperature", 0.0),
                top_p=dec.get("top_p", 1.0),
                max_new_tokens=dec.get("max_new_tokens", 2048),
            ),
            layer_accessor=val.get("layer_accessor", "model.model.layers"),
            dtype=val.get("dtype", "bfloat16"),
            enable_thinking=val.get("enable_thinking"),
        )
    return configs


def load_dataset_configs(path: Path) -> dict[str, DatasetConfig]:
    raw = _load_yaml(path)
    configs = {}
    for key, val in raw.items():
        configs[key] = DatasetConfig(
            name=key,
            hf_path=val["hf_path"],
            hf_name=val.get("hf_name"),
            split=val["split"],
            answer_type=val["answer_type"],
            max_answer_len=val.get("max_answer_len", 80),
            pilot_size=val.get("pilot_size"),
            main_size=val.get("main_size"),
            filter=val.get("filter", {}),
        )
    return configs


def load_prompt_configs(path: Path) -> tuple[dict[str, PromptTemplate], str | None]:
    raw = _load_yaml(path)
    templates = {}
    for key, val in raw.get("templates", {}).items():
        templates[key] = PromptTemplate(id=val["id"], template=val["template"])
    prefill = raw.get("assistant_prefill")
    return templates, prefill


def load_judge_configs(path: Path) -> tuple[JudgeConfig, dict[str, JudgeViewConfig]]:
    raw = _load_yaml(path)
    jm = raw["judge_model"]
    judge = JudgeConfig(
        provider=jm["provider"],
        model_id=jm["model_id"],
        base_url=jm["base_url"],
        max_tokens=jm["max_tokens"],
        temperature=jm["temperature"],
        max_workers=jm["max_workers"],
        max_retries=jm["max_retries"],
        retry_base_delay=jm["retry_base_delay"],
    )
    views = {}
    for key, val in raw.get("input_views", {}).items():
        views[key] = JudgeViewConfig(id=val["id"], prompt=val["prompt"])
    return judge, views


def load_project_config(config_dir: str | Path) -> ProjectConfig:
    """Load all YAML configs from a directory into a single ProjectConfig."""
    config_dir = Path(config_dir)
    models = load_model_configs(config_dir / "models.yaml")
    datasets = load_dataset_configs(config_dir / "datasets.yaml")
    templates, prefill = load_prompt_configs(config_dir / "prompts.yaml")
    judge, judge_views = load_judge_configs(config_dir / "judges.yaml")
    return ProjectConfig(
        models=models,
        datasets=datasets,
        prompt_templates=templates,
        assistant_prefill=prefill,
        judge=judge,
        judge_views=judge_views,
    )
