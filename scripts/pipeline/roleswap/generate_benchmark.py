"""Role-swapped synthetic benchmark (app:roleswap): prompts with the two wrong candidates' roles swapped, letter scores,
and grouped (base_id) probes of the sign of the pair-centred effect log p(h1|W) - log p(h1|G) minus the pair mean.
--build-manifest (CPU) -> role_swapped_manifest.csv; --run (GPU; HF_TOKEN with --use-hf-token) -> letter log-probs and
last-prompt-token states; --analyze (CPU, fixed C=1 logistic, alpha=10 ridge) -> role_swapped_{effect_rows_all,
predictions,binary_summary,regression_summary}.csv. The 7-9B role_swapped_effect_rows_all.csv is released as
data/scores/roleswap/effect_rows.csv.gz (5,184 rows, without w_prompt).
The paper ran scikit-learn 1.9.0; below 1.9 GroupKFold assigns the equal-size base_id groups to other folds.
Paper run (Qwen3-14B: --output-dir <dir_qwen14> --models qwen3_14b --model-shorts qwen3_14b --hidden-batch-size 4):

    python -m scripts.pipeline.roleswap.generate_benchmark --build-manifest --output-dir <dir>
    python -m scripts.pipeline.roleswap.generate_benchmark --run --analyze --output-dir <dir> \
        --models mistral_7b gemma_9b llama_8b --model-shorts mistral gemma llama --hidden-batch-size 8 --use-hf-token
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import os
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.pipeline.common.config import load_project_config
from scripts.pipeline.common.model_loader import DTYPE_MAP, format_chat, check_hf_access

LETTERS = ["A", "B", "C", "D"]

DEFAULT_LAYERS = {
    "mistral": [8, 16, 24, 31],
    "gemma": [8, 16, 24, 31],
    "llama": [8, 16, 24, 31],
    "qwen3_14b": [8, 16, 24, 39],
    "qwen3_32b": [8, 24, 48, 63],
    "qwq_32b": [8, 24, 48, 63],
}


# --------------------------------------------------------------------------
# Manifest (CPU)
# --------------------------------------------------------------------------
def stable_int(*parts: Any, modulo: int) -> int:
    text = "::".join(str(part) for part in parts)
    value = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)
    return value % modulo


def compute_rule(rule: str, values: dict[str, int]) -> int:
    a, b, c, d = values["a"], values["b"], values["c"], values["d"]
    e, f, g, h = values["e"], values["f"], values["g"], values["h"]
    if rule == "weighted6":
        return (2 * a + 3 * b + 5 * c + 7 * d + 11 * e + 13 * f) % 4
    if rule == "two_stage":
        u = (a + 2 * b + 3 * c + 5 * d) % 7
        v = (e + 2 * f + 3 * g + 5 * h) % 7
        return (u * v + a + 2 * h) % 4
    if rule == "nested":
        u = (a * b + c * d + e * f + g * h) % 11
        v = ((a + c + e + g) * (b + d + f + h)) % 13
        return (u + 2 * v + a + h) % 4
    raise ValueError(rule)


def rule_text(rule: str) -> str:
    if rule == "weighted6":
        return "r = (2a + 3b + 5c + 7d + 11e + 13f) mod 4"
    if rule == "two_stage":
        return (
            "u = (a + 2b + 3c + 5d) mod 7; "
            "v = (e + 2f + 3g + 5h) mod 7; "
            "r = (u*v + a + 2h) mod 4"
        )
    if rule == "nested":
        return (
            "u = (a*b + c*d + e*f + g*h) mod 11; "
            "v = ((a+c+e+g)*(b+d+f+h)) mod 13; "
            "r = (u + 2v + a + h) mod 4"
        )
    raise ValueError(rule)


def values_for_correct(seed: int, rule: str, base_idx: int, correct_idx: int) -> dict[str, int]:
    for attempt in range(40_000):
        values = {
            name: 10 + stable_int(seed, rule, base_idx, attempt, name, modulo=90)
            for name in "abcdefgh"
        }
        if compute_rule(rule, values) == correct_idx:
            return values
    raise RuntimeError(f"could not find values for {rule=} {base_idx=} {correct_idx=}")


def note_block(candidate: str, strength: str) -> str:
    if strength == "weak":
        return (
            "Reference note from an earlier pass:\n"
            f"- candidate: {candidate}\n"
            "- confidence: low\n"
            "This note may be wrong. Verify independently.\n"
        )
    if strength == "assertive":
        return (
            "Reference note from an earlier verifier:\n"
            f"- candidate: {candidate}\n"
            "- confidence: medium\n"
            "Use this as a useful clue, but still verify the arithmetic independently.\n"
        )
    raise ValueError(strength)


def build_prompt(row: dict[str, Any], condition: str) -> str:
    if condition == "W":
        candidate = row["h1"]
    elif condition == "G":
        candidate = row["h2"]
    elif condition == "neutral":
        candidate = ""
    else:
        raise ValueError(condition)

    body = f"""Question: Solve this synthetic checksum task.

Rule:
- Compute {rule_text(row['rule'])}.
- If r = 0, choose A.
- If r = 1, choose B.
- If r = 2, choose C.
- If r = 3, choose D.

Values:
a = {row['a']}
b = {row['b']}
c = {row['c']}
d = {row['d']}
e = {row['e']}
f = {row['f']}
g = {row['g']}
h = {row['h']}

Options:
A) option A
B) option B
C) option C
D) option D
"""
    if condition == "neutral":
        body += "\nNo reference note is available.\n\n"
    else:
        body += "\n" + note_block(candidate, row["strength"]) + "\n"
    body += (
        "Do not show your reasoning.\n"
        "Write exactly one line and nothing else:\n"
        "Final answer: <option letter>\n"
    )
    return body


def build_manifest(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    base_idx = 0
    for strength in args.strengths:
        for rule in args.rules:
            for correct in LETTERS:
                correct_idx = LETTERS.index(correct)
                for repeat in range(args.repeats_per_correct):
                    values = values_for_correct(args.seed, rule, base_idx, correct_idx)
                    base_id = f"{strength}_{rule}_{correct}_{repeat:03d}"
                    wrong = [letter for letter in LETTERS if letter != correct]
                    for pair_idx, (left, right) in enumerate(combinations(wrong, 2)):
                        swap_pair_id = f"{base_id}_pair{pair_idx}_{left}{right}"
                        for h1, h2 in [(left, right), (right, left)]:
                            orientation_id = f"{swap_pair_id}_{h1}{h2}"
                            base = {
                                "dataset": "synthetic_role_swap",
                                "base_id": base_id,
                                "swap_pair_id": swap_pair_id,
                                "orientation_id": orientation_id,
                                "pair_index": pair_idx,
                                "strength": strength,
                                "rule": rule,
                                **values,
                                "r": correct_idx,
                                "correct_letter": correct,
                                "h1": h1,
                                "h2": h2,
                                "unordered_pair": "".join(sorted([h1, h2])),
                            }
                            for condition in ["W", "G", "neutral"]:
                                prompt = build_prompt(base, condition)
                                rows.append(
                                    {
                                        **base,
                                        "condition": condition,
                                        "row_id": f"{orientation_id}_{condition}",
                                        "candidate": h1 if condition == "W" else (h2 if condition == "G" else ""),
                                        "prompt": prompt,
                                    }
                                )
                    base_idx += 1

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "role_swapped_manifest.csv", index=False)
    summary = manifest.groupby(["strength", "rule", "condition"]).size().reset_index(name="n")
    summary.to_csv(output_dir / "role_swapped_manifest_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"wrote manifest -> {output_dir / 'role_swapped_manifest.csv'}", flush=True)


# --------------------------------------------------------------------------
# Letter scores and hidden states (GPU)
# --------------------------------------------------------------------------
@torch.inference_mode()
def hidden_batch_layers(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    layers: list[int],
) -> dict[int, np.ndarray]:
    # add_special_tokens stays at its default (True), as in the recorded run; on chat-templated text this can
    # add a second BOS token.
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    last_positions = encoded["attention_mask"].sum(dim=1) - 1
    out = model(**encoded, output_hidden_states=True, use_cache=False)
    result = {}
    for layer in layers:
        hs = out.hidden_states[layer]
        result[layer] = torch.stack(
            [hs[i, int(pos.item())] for i, pos in enumerate(last_positions)]
        ).float().cpu().numpy()
    return result


def parse_layer_args(model_short: str, layer_args: list[int] | None, n_layers: int) -> list[int]:
    if layer_args:
        layers = layer_args
    else:
        layers = DEFAULT_LAYERS.get(model_short, [max(1, n_layers // 4), max(1, n_layers // 2), n_layers - 1])
    return sorted({layer if layer >= 0 else n_layers + layer for layer in layers if -n_layers <= layer < n_layers})


@torch.inference_mode()
def score_letters_for_prompt(model: Any, tokenizer: Any, prompt_text: str) -> dict[str, float]:
    """Score MCQ letters with one batched forward for all letter variants."""
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    variants: list[tuple[str, list[int]]] = []
    for letter in LETTERS:
        for text in [f" {letter}", letter, f" {letter})"]:
            answer_ids = tokenizer(text, add_special_tokens=False).input_ids
            if answer_ids:
                variants.append((letter, answer_ids))
    if not prompt_ids or not variants:
        return {f"logp_{letter}": float("-inf") for letter in LETTERS}

    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    full_rows = [prompt_ids + answer_ids for _, answer_ids in variants]
    max_len = max(len(row) for row in full_rows)
    input_ids = torch.full((len(full_rows), max_len), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(full_rows), max_len), dtype=torch.long, device=device)
    for idx, row in enumerate(full_rows):
        input_ids[idx, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        attention_mask[idx, : len(row)] = 1

    out = model(input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1], use_cache=False)
    log_probs = torch.nn.functional.log_softmax(out.logits.float(), dim=-1)
    best: dict[str, float] = {letter: float("-inf") for letter in LETTERS}
    start = len(prompt_ids)
    for idx, (letter, answer_ids) in enumerate(variants):
        token_scores = []
        for offset, target_id in enumerate(answer_ids):
            pred_pos = start + offset - 1
            if 0 <= pred_pos < log_probs.shape[1]:
                token_scores.append(float(log_probs[idx, pred_pos, target_id].item()))
        if token_scores:
            best[letter] = max(best[letter], float(np.mean(token_scores)))
    return {f"logp_{letter}": best[letter] for letter in LETTERS}


def run_model(args: argparse.Namespace, model_key: str) -> None:
    cfg = load_project_config(Path(args.config))
    model_cfg = cfg.models[model_key]
    model_short = model_cfg.short_name
    output_dir = Path(args.output_dir)
    manifest = pd.read_csv(output_dir / "role_swapped_manifest.csv")
    model_dir = output_dir / model_short
    model_dir.mkdir(parents=True, exist_ok=True)
    score_path = model_dir / "role_swapped_letter_scores.csv"
    feature_rows_path = model_dir / "role_swapped_hidden_rows.csv"
    feature_path = model_dir / "role_swapped_prompt_hidden.npz"

    hf_token = None
    if args.use_hf_token:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError("--use-hf-token was given but HF_TOKEN is not set")
    check_hf_access(model_cfg.model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_id, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_id,
        torch_dtype=DTYPE_MAP.get(model_cfg.dtype, torch.bfloat16),
        device_map="auto",
        token=hf_token,
    )
    model.eval()
    n_layers = int(getattr(model.config, "num_hidden_layers", 0))
    layers = parse_layer_args(model_short, args.layers, n_layers)
    print(f"[{model_short}] using layers={layers}", flush=True)

    records = manifest.to_dict("records")
    prompts = [format_chat(tokenizer, model_cfg, str(row["prompt"])) for row in records]

    done: set[str] = set()
    if score_path.exists():
        prev = pd.read_csv(score_path)
        done = set(prev["row_id"].astype(str))
    remaining = [(row, prompt) for row, prompt in zip(records, prompts) if str(row["row_id"]) not in done]
    print(f"[{model_short}] score remaining={len(remaining)}", flush=True)
    write_header = not score_path.exists()
    with score_path.open("a") as f:
        for idx, (row, prompt) in enumerate(tqdm(remaining, desc=f"{model_short} scores"), start=1):
            answer_prompt = prompt + args.answer_prefix
            out = {
                key: row[key]
                for key in [
                    "row_id",
                    "dataset",
                    "base_id",
                    "swap_pair_id",
                    "orientation_id",
                    "pair_index",
                    "strength",
                    "rule",
                    "correct_letter",
                    "h1",
                    "h2",
                    "unordered_pair",
                    "condition",
                ]
            }
            out["model"] = model_short
            out.update(score_letters_for_prompt(model, tokenizer, answer_prompt))
            pd.DataFrame([out]).to_csv(f, header=write_header, index=False)
            write_header = False
            if idx % 100 == 0:
                f.flush()

    if not feature_path.exists():
        tokenizer.padding_side = "right"
        chunks: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
        for start in tqdm(range(0, len(prompts), args.hidden_batch_size), desc=f"{model_short} hidden"):
            batch = hidden_batch_layers(model, tokenizer, prompts[start : start + args.hidden_batch_size], layers)
            for layer, arr in batch.items():
                chunks[layer].append(arr)
        arrays = {f"layer_{layer}": np.concatenate(parts, axis=0) for layer, parts in chunks.items()}
        arrays["layers"] = np.asarray(layers, dtype=np.int32)
        np.savez_compressed(feature_path, **arrays)
        row_meta = manifest[
            [
                "row_id",
                "base_id",
                "swap_pair_id",
                "orientation_id",
                "strength",
                "rule",
                "h1",
                "h2",
                "condition",
            ]
        ].copy()
        row_meta.insert(0, "row_idx", np.arange(len(row_meta)))
        row_meta.insert(1, "model", model_short)
        row_meta.to_csv(feature_rows_path, index=False)
        print(f"[{model_short}] wrote hidden -> {feature_path}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# Analysis (CPU)
# --------------------------------------------------------------------------
def safe_corr(y: np.ndarray, s: np.ndarray, kind: str) -> float:
    mask = np.isfinite(y) & np.isfinite(s)
    if mask.sum() < 3 or np.std(y[mask]) == 0 or np.std(s[mask]) == 0:
        return math.nan
    if kind == "pearson":
        return float(pearsonr(y[mask], s[mask]).statistic)
    if kind == "spearman":
        return float(spearmanr(y[mask], s[mask]).statistic)
    raise ValueError(kind)


def grouped_ridge_oof_fixed(
    x: np.ndarray | sparse.spmatrix,
    y: np.ndarray,
    groups: np.ndarray,
    alpha: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.full(len(y), np.nan, dtype=float)
    used = np.zeros(len(y), dtype=bool)
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        return pred, used
    outer = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in outer.split(x, y, groups):
        scaler = StandardScaler(with_mean=not sparse.issparse(x))
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        reg = Ridge(alpha=alpha, solver="lsqr", tol=1e-3, max_iter=5000)
        reg.fit(x_train, y[train_idx])
        pred[test_idx] = reg.predict(x_test)
        used[test_idx] = True
    return pred, used


def grouped_logistic_oof_fixed(
    x: np.ndarray | sparse.spmatrix,
    y: np.ndarray,
    groups: np.ndarray,
    c_value: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.full(len(y), np.nan, dtype=float)
    used = np.zeros(len(y), dtype=bool)
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2 or len(np.unique(y)) < 2:
        return pred, used
    outer = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in outer.split(x, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            continue
        scaler = StandardScaler(with_mean=not sparse.issparse(x))
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        clf = LogisticRegression(C=c_value, class_weight="balanced", max_iter=1000, solver="liblinear")
        clf.fit(x_train, y[train_idx])
        pred[test_idx] = clf.predict_proba(x_test)[:, 1]
        used[test_idx] = True
    return pred, used


def summarize_regression(y: np.ndarray, pred: np.ndarray, used: np.ndarray) -> dict[str, float | int]:
    mask = used & np.isfinite(y) & np.isfinite(pred)
    if mask.sum() < 3:
        return {"n_used": int(mask.sum()), "pearson": math.nan, "spearman": math.nan, "mse": math.nan, "r2": math.nan}
    return {
        "n_used": int(mask.sum()),
        "pearson": safe_corr(y[mask], pred[mask], "pearson"),
        "spearman": safe_corr(y[mask], pred[mask], "spearman"),
        "mse": float(mean_squared_error(y[mask], pred[mask])),
        "r2": float(r2_score(y[mask], pred[mask])),
    }


def summarize_binary(y: np.ndarray, pred: np.ndarray, used: np.ndarray) -> dict[str, float | int]:
    mask = used & np.isfinite(pred)
    if mask.sum() < 3 or len(np.unique(y[mask])) < 2:
        return {"n_used": int(mask.sum()), "auroc": math.nan, "auprc": math.nan}
    return {
        "n_used": int(mask.sum()),
        "auroc": float(roc_auc_score(y[mask], pred[mask])),
        "auprc": float(average_precision_score(y[mask], pred[mask])),
    }


def numeric_features(df: pd.DataFrame) -> np.ndarray:
    pieces = []
    for col in ["h1", "h2", "correct_letter", "unordered_pair", "rule", "strength", "pair_index"]:
        pieces.append(pd.get_dummies(df[col].astype(str), prefix=col))
    numeric_cols = [col for col in list("abcdefgh") + ["r"] if col in df.columns]
    pieces.append(df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0))
    return pd.concat(pieces, axis=1).to_numpy(dtype=np.float32)


def load_hidden(model_dir: Path) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    meta = pd.read_csv(model_dir / "role_swapped_hidden_rows.csv")
    npz = np.load(model_dir / "role_swapped_prompt_hidden.npz")
    layers = [int(layer) for layer in npz["layers"].tolist()]
    arrays = {layer: npz[f"layer_{layer}"] for layer in layers}
    return meta, arrays


def build_effect_rows(output_dir: Path, model_short: str) -> pd.DataFrame:
    manifest = pd.read_csv(output_dir / "role_swapped_manifest.csv")
    scores = pd.read_csv(output_dir / model_short / "role_swapped_letter_scores.csv")
    score_piv = {cond: frame.set_index("orientation_id") for cond, frame in scores.groupby("condition")}
    manifest_w = manifest[manifest["condition"] == "W"].drop_duplicates("orientation_id")
    rows = []
    for row in manifest_w.to_dict("records"):
        oid = row["orientation_id"]
        if oid not in score_piv.get("W", pd.DataFrame()).index or oid not in score_piv.get("G", pd.DataFrame()).index:
            continue
        w = score_piv["W"].loc[oid]
        g = score_piv["G"].loc[oid]
        neutral = score_piv.get("neutral", pd.DataFrame()).loc[oid] if "neutral" in score_piv and oid in score_piv["neutral"].index else None
        h1 = str(row["h1"]).upper()
        h2 = str(row["h2"]).upper()
        effect = float(w[f"logp_{h1}"] - g[f"logp_{h1}"])
        anti_effect = float(g[f"logp_{h2}"] - w[f"logp_{h2}"])
        neutral_h1 = float(neutral[f"logp_{h1}"]) if neutral is not None else math.nan
        neutral_h2 = float(neutral[f"logp_{h2}"]) if neutral is not None else math.nan
        rows.append(
            {
                **{k: row[k] for k in [
                    "dataset",
                    "base_id",
                    "swap_pair_id",
                    "orientation_id",
                    "pair_index",
                    "strength",
                    "rule",
                    "a",
                    "b",
                    "c",
                    "d",
                    "e",
                    "f",
                    "g",
                    "h",
                    "r",
                    "correct_letter",
                    "h1",
                    "h2",
                    "unordered_pair",
                ]},
                "model": model_short,
                "effect_h1_w_minus_g": effect,
                "effect_h2_g_minus_w": anti_effect,
                "neutral_logp_h1": neutral_h1,
                "neutral_logp_h2": neutral_h2,
                "neutral_gap_h1_minus_h2": neutral_h1 - neutral_h2,
                "w_margin_h1_minus_h2": float(w[f"logp_{h1}"] - w[f"logp_{h2}"]),
                "g_margin_h1_minus_h2": float(g[f"logp_{h1}"] - g[f"logp_{h2}"]),
                "w_prompt": str(row["prompt"]),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["pair_mean_effect"] = df.groupby("swap_pair_id")["effect_h1_w_minus_g"].transform("mean")
    df["pair_centered_effect"] = df["effect_h1_w_minus_g"] - df["pair_mean_effect"]
    df["pair_abs_gap"] = df.groupby("swap_pair_id")["effect_h1_w_minus_g"].transform(lambda x: float(np.max(x) - np.min(x)))
    df["binary_label"] = (df["pair_centered_effect"] > 0).astype(int)
    return df


def hidden_index(meta: pd.DataFrame) -> dict[tuple[str, str], int]:
    return {(str(row.orientation_id), str(row.condition)): idx for idx, row in enumerate(meta.itertuples())}


def hidden_features_from_loaded(
    effect_rows: pd.DataFrame,
    hidden: np.ndarray,
    index: dict[tuple[str, str], int],
    source: str,
) -> np.ndarray:
    feats = []
    for _, row in effect_rows.iterrows():
        orientation_id = str(row["orientation_id"])
        w = hidden[index[(orientation_id, "W")]]
        g = hidden[index[(orientation_id, "G")]]
        if source == "W_hidden":
            feats.append(w)
        elif source == "G_hidden":
            feats.append(g)
        elif source == "W_minus_G":
            feats.append(w - g)
        elif source == "W_minus_neutral":
            feats.append(w - hidden[index[(orientation_id, "neutral")]])
        else:
            raise ValueError(source)
    return np.asarray(feats, dtype=np.float32)


def centered_features(x: np.ndarray, effect_rows: pd.DataFrame) -> np.ndarray:
    centered = np.zeros_like(x)
    for _, idx in effect_rows.groupby("swap_pair_id").groups.items():
        arr = x[list(idx)]
        centered[list(idx)] = arr - arr.mean(axis=0, keepdims=True)
    return centered


def prediction_rows_for(
    model_short: str,
    method: str,
    layer: float,
    effect_rows: pd.DataFrame,
    pred_r: np.ndarray,
    used_r: np.ndarray,
    pred_b: np.ndarray,
    used_b: np.ndarray,
) -> list[dict[str, Any]]:
    # One row per (orientation_id, method). `prediction`/`used` are the ridge OOF prediction of the continuous
    # target; `pred_r` is an alias of `prediction`; `pred_b`/`used_b` are the logistic OOF probability and its
    # coverage mask behind the binary AUROC (grouped by base_id), so bootstrap_cis.py can recompute its interval.
    out = []
    for row, score, used, score_b, used_bin in zip(effect_rows.to_dict("records"), pred_r, used_r, pred_b, used_b):
        out.append(
            {
                "model": model_short,
                "method": method,
                "layer": layer,
                "base_id": row["base_id"],
                "swap_pair_id": row["swap_pair_id"],
                "orientation_id": row["orientation_id"],
                "pair_centered_effect": row["pair_centered_effect"],
                "binary_label": row["binary_label"],
                "prediction": float(score) if np.isfinite(score) else math.nan,
                "used": bool(used),
                "pred_r": float(score) if np.isfinite(score) else math.nan,
                "pred_b": float(score_b) if np.isfinite(score_b) else math.nan,
                "used_b": bool(used_bin),
            }
        )
    return out


def analyze(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    regression_rows = []
    binary_rows = []
    prediction_rows = []
    effect_all = []

    for model_short in args.model_shorts:
        model_dir = output_dir / model_short
        if not (model_dir / "role_swapped_letter_scores.csv").exists():
            continue
        if not (model_dir / "role_swapped_prompt_hidden.npz").exists():
            continue
        effect_rows = build_effect_rows(output_dir, model_short)
        if effect_rows.empty:
            continue
        effect_rows.to_csv(model_dir / "role_swapped_effect_rows.csv", index=False)
        effect_all.append(effect_rows)
        groups = effect_rows["base_id"].astype(str).to_numpy()
        y_reg = effect_rows["pair_centered_effect"].to_numpy(dtype=float)
        y_bin = effect_rows["binary_label"].to_numpy(dtype=int)

        def fit(method: str, layer: float, x: np.ndarray | sparse.spmatrix) -> None:
            pred_r, used_r = grouped_ridge_oof_fixed(x, y_reg, groups)
            regression_rows.append(
                {
                    "model": model_short,
                    "method": method,
                    "layer": layer,
                    "n": len(effect_rows),
                    "target_mean": float(np.mean(y_reg)),
                    "target_std": float(np.std(y_reg)),
                    **summarize_regression(y_reg, pred_r, used_r),
                }
            )
            pred_b, used_b = grouped_logistic_oof_fixed(x, y_bin, groups)
            binary_rows.append(
                {
                    "model": model_short,
                    "method": method,
                    "layer": layer,
                    "n": len(effect_rows),
                    "positives": int(y_bin.sum()),
                    "negatives": int((1 - y_bin).sum()),
                    **summarize_binary(y_bin, pred_b, used_b),
                }
            )
            prediction_rows.extend(
                prediction_rows_for(model_short, method, layer, effect_rows, pred_r, used_r, pred_b, used_b)
            )

        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=4000)
        x_text = vectorizer.fit_transform(effect_rows["w_prompt"].astype(str).tolist())
        x_num = numeric_features(effect_rows)
        x_num_sparse = sparse.csr_matrix(x_num)
        fit("numeric_public", math.nan, x_num)
        fit("tfidf_w_prompt", math.nan, x_text)
        fit("numeric_plus_tfidf", math.nan, sparse.hstack([x_num_sparse, x_text]).tocsr())
        fit(
            "neutral_logp_features",
            math.nan,
            effect_rows[["neutral_logp_h1", "neutral_logp_h2", "neutral_gap_h1_minus_h2"]].to_numpy(dtype=np.float32),
        )

        meta, arrays = load_hidden(model_dir)
        index = hidden_index(meta)
        for layer in sorted(arrays):
            hidden = arrays[layer]
            x_w_minus_g = hidden_features_from_loaded(effect_rows, hidden, index, "W_minus_G")
            fit(f"W_hidden_L{layer}", layer, hidden_features_from_loaded(effect_rows, hidden, index, "W_hidden"))
            fit(f"G_hidden_L{layer}", layer, hidden_features_from_loaded(effect_rows, hidden, index, "G_hidden"))
            fit(f"W_minus_G_L{layer}", layer, x_w_minus_g)
            fit(f"W_minus_neutral_L{layer}", layer, hidden_features_from_loaded(effect_rows, hidden, index, "W_minus_neutral"))
            fit(f"W_minus_G_centered_L{layer}", layer, centered_features(x_w_minus_g, effect_rows))

    reg = pd.DataFrame(regression_rows)
    binary = pd.DataFrame(binary_rows)
    preds = pd.DataFrame(prediction_rows)
    reg.to_csv(output_dir / "role_swapped_regression_summary.csv", index=False)
    binary.to_csv(output_dir / "role_swapped_binary_summary.csv", index=False)
    # Full float precision so the AUROC recomputed from pred_b reproduces the summary exactly
    # (the default float writer can merge 1e-17-apart scores into ties).
    preds.to_csv(output_dir / "role_swapped_predictions.csv", index=False, float_format="%.17g")
    if effect_all:
        pd.concat(effect_all, ignore_index=True).to_csv(output_dir / "role_swapped_effect_rows_all.csv", index=False)
    if not reg.empty:
        print("REGRESSION", flush=True)
        print(reg.sort_values(["model", "pearson"], ascending=[True, False]).head(60).round(4).to_string(index=False), flush=True)
    if not binary.empty:
        print("BINARY", flush=True)
        print(binary.sort_values(["model", "auroc"], ascending=[True, False]).head(60).round(4).to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="data/prompts/")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", default=["mistral_7b", "gemma_9b", "llama_8b"])
    parser.add_argument("--model-shorts", nargs="+", default=["mistral", "gemma", "llama"])
    parser.add_argument("--rules", nargs="+", default=["weighted6", "two_stage", "nested"])
    parser.add_argument("--strengths", nargs="+", default=["weak", "assertive"])
    parser.add_argument("--repeats-per-correct", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--answer-prefix", default="\nFinal answer:")
    parser.add_argument("--hidden-batch-size", type=int, default=16)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--use-hf-token", action="store_true", help="pass HF_TOKEN from the environment (gated models)")
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    if args.build_manifest:
        build_manifest(args)
    if args.run:
        for model_key in args.models:
            run_model(args, model_key)
    if args.analyze:
        analyze(args)
    if not args.build_manifest and not args.run and not args.analyze:
        parser.error("Specify --build-manifest, --run, and/or --analyze")


if __name__ == "__main__":
    main()
