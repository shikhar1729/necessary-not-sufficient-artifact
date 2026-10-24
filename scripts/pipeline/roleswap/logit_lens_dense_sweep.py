"""Role-swap benchmark (app:roleswap): parameter-free logit-lens readout and a dense layer sweep of the W-G probe.
--extract (GPU): states at every slot and lens log-probabilities of the option letters, at the prompt end and the
answer position. --analyze (CPU): lens AUROC per slot, fixed-C and nested (layer, C) probes, cluster bootstrap over
base_id. Input: the role-swap run (--source-dir). Output (--output-dir): logit_lens_summary.csv and
dense_probe_oof_predictions.csv.gz, released in data/scores/roleswap/ as logit_lens_summary.csv and
dense_probe_oof.csv.gz (without swap_pair_id and pair_centered_effect, which are in effect_rows.csv.gz).
Recorded run: --extract, then --analyze --n-jobs 32, under scikit-learn 1.9.1 (its folds need 1.9.x). The lens
files of that run stored ids as object arrays, which np.load without pickling cannot read: re-run --extract with
this version (it writes fixed-width strings) before --analyze.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.pipeline.common.config import load_project_config
from scripts.pipeline.common.model_loader import DTYPE_MAP, format_chat, check_hf_access
from scripts.pipeline.roleswap.generate_benchmark import LETTERS

C_GRID = [0.01, 0.1, 1.0, 10.0]
FIXED_C = 1.0
POSITIONS = ["prompt_end", "answer_pos"]
FOUR_LAYERS = {  # the four layers swept in tab:roleswap-main
    "mistral": [8, 16, 24, 31],
    "gemma": [8, 16, 24, 31],
    "llama": [8, 16, 24, 31],
    "qwen3_14b": [8, 16, 24, 39],
    "qwen3_32b": [8, 24, 48, 63],
}


# --------------------------------------------------------------------------
# GPU stage
# --------------------------------------------------------------------------
def letter_token_ids(tokenizer: Any) -> dict[str, list[int]]:
    """Single-token ids for each option letter, with and without a leading space."""
    out: dict[str, list[int]] = {}
    for letter in LETTERS:
        ids: list[int] = []
        for text in [f" {letter}", letter]:
            encoded = tokenizer(text, add_special_tokens=False).input_ids
            if len(encoded) == 1 and int(encoded[0]) not in ids:
                ids.append(int(encoded[0]))
        if not ids:
            encoded = tokenizer(f" {letter}", add_special_tokens=False).input_ids
            ids = [int(encoded[-1])]
        out[letter] = ids
    return out


def get_final_norm(model: Any) -> Any:
    for path in ["model.norm", "model.final_layernorm", "transformer.ln_f"]:
        obj = model
        ok = True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            return obj
    raise RuntimeError("could not locate the final norm module")


@torch.inference_mode()
def forward_states_and_lens(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    letter_ids: dict[str, list[int]],
    final_norm: Any,
    lm_head: Any,
    softcap: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (states[b, slot, d] float16, lens_logp[b, slot, 4] float32)."""
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    last_positions = (encoded["attention_mask"].sum(dim=1) - 1)
    out = model(**encoded, output_hidden_states=True, use_cache=False)
    n_slots = len(out.hidden_states)
    row_idx = torch.arange(len(prompts), device=out.hidden_states[0].device)
    pos = last_positions.to(out.hidden_states[0].device)

    states = []
    lens = np.zeros((len(prompts), n_slots, len(LETTERS)), dtype=np.float32)
    for slot in range(n_slots):
        vec = out.hidden_states[slot][row_idx, pos]  # [b, d]
        states.append(vec.detach().float().cpu().numpy().astype(np.float16))
        # hidden_states[-1] is already post-final-norm in HF; earlier slots are not.
        normed = vec if slot == n_slots - 1 else final_norm(vec)
        logits = lm_head(normed).float()
        if softcap:
            logits = torch.tanh(logits / float(softcap)) * float(softcap)
        logp = torch.log_softmax(logits, dim=-1)
        for j, letter in enumerate(LETTERS):
            ids = torch.tensor(letter_ids[letter], device=logp.device)
            lens[:, slot, j] = logp[:, ids].max(dim=-1).values.float().cpu().numpy()
        del logits, logp, normed
    del out
    return np.stack(states, axis=1), lens


def extract_model(args: argparse.Namespace, model_key: str) -> None:
    cfg = load_project_config(Path(args.config))
    model_cfg = cfg.models[model_key]
    short = model_cfg.short_name
    source_dir = Path(args.source_dir)
    out_dir = Path(args.output_dir) / short
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(source_dir / "role_swapped_manifest.csv")
    if args.limit:
        manifest = manifest.head(args.limit).copy()
    records = manifest.to_dict("records")

    hf_token = os.environ.get("HF_TOKEN") if args.use_hf_token else None
    check_hf_access(model_cfg.model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_id, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_id,
        torch_dtype=DTYPE_MAP.get(model_cfg.dtype, torch.bfloat16),
        device_map="auto",
        token=hf_token,
    )
    model.eval()

    final_norm = get_final_norm(model)
    lm_head = model.get_output_embeddings()
    softcap = getattr(model.config, "final_logit_softcapping", None)
    letter_ids = letter_token_ids(tokenizer)
    print(f"[{short}] letter token ids: {letter_ids}", flush=True)
    print(f"[{short}] final_logit_softcapping={softcap}", flush=True)

    base_prompts = [format_chat(tokenizer, model_cfg, str(row["prompt"])) for row in records]
    stats_path = out_dir / "extract_stats.json"
    stats: dict[str, Any] = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    stats.update(
        {
            "model": short,
            "model_id": model_cfg.model_id,
            "n_rows": len(records),
            "letter_token_ids": letter_ids,
            "final_logit_softcapping": softcap,
            "transformers_note": "hidden_states[-1] is post-final-norm; earlier slots get final_norm applied for the lens",
        }
    )
    stats.setdefault("positions", {})

    for position in POSITIONS:
        marker = out_dir / f"DONE_{position}.txt"
        if marker.exists() and not args.force:
            print(f"[{short}] {position} already done", flush=True)
            continue
        prompts = base_prompts if position == "prompt_end" else [p + args.answer_prefix for p in base_prompts]
        t0 = time.time()
        state_parts: list[np.ndarray] = []
        lens_parts: list[np.ndarray] = []
        for start in tqdm(
            range(0, len(prompts), args.batch_size),
            desc=f"{short}/{position}",
            mininterval=30.0,
        ):
            s, l = forward_states_and_lens(
                model,
                tokenizer,
                prompts[start : start + args.batch_size],
                letter_ids,
                final_norm,
                lm_head,
                softcap,
            )
            state_parts.append(s)
            lens_parts.append(l)
        states = np.concatenate(state_parts, axis=0)
        lens = np.concatenate(lens_parts, axis=0)
        elapsed = time.time() - t0

        np.savez(
            out_dir / f"lens_logp_{position}.npz",
            lens=lens,
            row_id=manifest["row_id"].astype(str).to_numpy(dtype="<U80"),
            orientation_id=manifest["orientation_id"].astype(str).to_numpy(dtype="<U80"),
            condition=manifest["condition"].astype(str).to_numpy(dtype="<U16"),
            letters=np.asarray(LETTERS),
        )
        np.save(out_dir / f"dense_states_{position}.npy", states)
        (out_dir / f"dense_states_{position}_rows.csv").write_text(
            "\n".join(["row_id"] + manifest["row_id"].astype(str).tolist()) + "\n"
        )
        stats["positions"][position] = {
            "n_slots": int(states.shape[1]),
            "hidden_dim": int(states.shape[2]),
            "seconds": round(elapsed, 1),
            "rows_per_second": round(len(prompts) / max(elapsed, 1e-6), 2),
            "non_finite_lens": int((~np.isfinite(lens)).sum()),
            "state_dtype": str(states.dtype),
        }
        marker.write_text(f"{position} ok {elapsed:.1f}s\n")
        print(f"[{short}] {position}: states {states.shape} in {elapsed:.0f}s", flush=True)
        del states, lens, state_parts, lens_parts
        gc.collect()
        stats_path.write_text(json.dumps(stats, indent=2))

    stats_path.write_text(json.dumps(stats, indent=2))
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# CPU stage helpers
# --------------------------------------------------------------------------
def cluster_bootstrap_auroc(
    y: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    mask = np.isfinite(scores)
    y, scores, groups = y[mask], scores[mask], groups[mask]
    if len(np.unique(y)) < 2:
        return {"auroc": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": int(len(y))}
    point = float(roc_auc_score(y, scores))
    if n_boot <= 0:
        return {"auroc": point, "ci_low": math.nan, "ci_high": math.nan, "n": int(len(y))}
    codes, uniques = pd.factorize(groups)
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(len(uniques)), side="left")
    ends = np.searchsorted(codes[order], np.arange(len(uniques)), side="right")
    index = [order[s:e] for s, e in zip(starts, ends)]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sampled = rng.integers(0, len(uniques), size=len(uniques))
        idx = np.concatenate([index[g] for g in sampled])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(float(roc_auc_score(y[idx], scores[idx])))
    if not vals:
        return {"auroc": point, "ci_low": point, "ci_high": point, "n": int(len(y))}
    arr = np.asarray(vals)
    return {
        "auroc": point,
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "n": int(len(y)),
    }


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return math.nan
    return float(pearsonr(a[mask], b[mask]).statistic)


def fit_predict(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, c_value: float) -> np.ndarray:
    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_te_s = scaler.transform(x_te)
    # Exactly the estimator the role-swap benchmark used in `--fast-analyze`
    # (`grouped_logistic_oof_fixed`): primal liblinear, balanced classes,
    # max_iter=1000. A dual-liblinear variant was 12x slower for the same AUROC
    # to three decimals, and lbfgs agreed to within 0.002.
    clf = LogisticRegression(
        C=c_value, class_weight="balanced", max_iter=1000, solver="liblinear"
    )
    clf.fit(x_tr_s, y_tr)
    return clf.predict_proba(x_te_s)[:, 1]


def layer_fixed_c_oof(
    states: np.ndarray,
    layer: int,
    diff_with: np.ndarray | None,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    c_value: float,
) -> np.ndarray:
    """OOF probabilities for one layer. `states` and `diff_with` are row-aligned."""
    x = states[:, layer, :].astype(np.float32)
    if diff_with is not None:
        x = x - diff_with[:, layer, :].astype(np.float32)
    pred = np.full(len(y), np.nan)
    for train_idx, test_idx in folds:
        if len(np.unique(y[train_idx])) < 2:
            continue
        pred[test_idx] = fit_predict(x[train_idx], y[train_idx], x[test_idx], c_value)
    return pred


def inner_layer_c_score(
    states: np.ndarray,
    diff_with: np.ndarray | None,
    layer: int,
    c_value: float,
    y: np.ndarray,
    train_idx: np.ndarray,
    inner: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, int, float]:
    x = states[:, layer, :].astype(np.float32)
    if diff_with is not None:
        x = x - diff_with[:, layer, :].astype(np.float32)
    vals = []
    for tr_rel, va_rel in inner:
        tr = train_idx[tr_rel]
        va = train_idx[va_rel]
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            continue
        probs = fit_predict(x[tr], y[tr], x[va], c_value)
        vals.append(float(roc_auc_score(y[va], probs)))
    return (float(np.mean(vals)) if vals else 0.5, layer, c_value)


def nested_dense_oof(
    states: np.ndarray,
    diff_with: np.ndarray | None,
    y: np.ndarray,
    groups: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    layers: list[int],
    n_jobs: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Fold-local selection of (layer, C) by inner GroupKFold AUROC."""
    pred = np.full(len(y), np.nan)
    rows: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(folds):
        if len(np.unique(y[train_idx])) < 2:
            continue
        inner_groups = np.unique(groups[train_idx])
        n_inner = max(2, min(3, len(inner_groups)))
        inner = list(
            GroupKFold(n_splits=n_inner).split(
                np.zeros(len(train_idx)), y[train_idx], groups[train_idx]
            )
        )
        combos = [(layer, c) for layer in layers for c in C_GRID]
        results = Parallel(n_jobs=n_jobs, backend="loky", max_nbytes="4M")(
            delayed(inner_layer_c_score)(states, diff_with, layer, c, y, train_idx, inner)
            for layer, c in combos
        )
        inner_auroc, layer, c_value = max(results, key=lambda r: r[0])
        x = states[:, layer, :].astype(np.float32)
        if diff_with is not None:
            x = x - diff_with[:, layer, :].astype(np.float32)
        pred[test_idx] = fit_predict(x[train_idx], y[train_idx], x[test_idx], c_value)
        rows.append(
            {
                "fold": fold,
                "selected_layer": layer,
                "C": c_value,
                "inner_auroc": inner_auroc,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
        )
    return pred, rows


def pair_center(values: np.ndarray, pair_codes: np.ndarray, n_pairs: int) -> np.ndarray:
    """Subtract the swap-pair mean. O(n) via bincount; `pair_codes` is factorized once."""
    values = np.asarray(values, dtype=float)
    sums = np.bincount(pair_codes, weights=values, minlength=n_pairs)
    counts = np.bincount(pair_codes, minlength=n_pairs).astype(float)
    counts[counts == 0] = 1.0
    return values - (sums / counts)[pair_codes]


# --------------------------------------------------------------------------
# CPU stage
# --------------------------------------------------------------------------
def analyze_model(args: argparse.Namespace, short: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_dir = Path(args.source_dir)
    data_dir = Path(args.output_dir) / short
    effect_all = pd.read_csv(source_dir / "role_swapped_effect_rows_all.csv")
    effect = effect_all[effect_all["model"] == short].reset_index(drop=True)
    if effect.empty:
        raise RuntimeError(f"no effect rows for {short}")

    y = effect["binary_label"].to_numpy(dtype=int)
    y_cont = effect["pair_centered_effect"].to_numpy(dtype=float)
    groups = effect["base_id"].astype(str).to_numpy()
    pair_ids = effect["swap_pair_id"].astype(str).to_numpy()
    pair_codes, pair_uniques = pd.factorize(pair_ids)
    n_pairs = len(pair_uniques)
    orientations = effect["orientation_id"].astype(str).to_numpy()
    h1 = effect["h1"].astype(str).str.upper().to_numpy()
    h2 = effect["h2"].astype(str).str.upper().to_numpy()
    li = {letter: i for i, letter in enumerate(LETTERS)}
    h1_cols = np.array([li[c] for c in h1])
    h2_cols = np.array([li[c] for c in h2])
    rows_arange = np.arange(len(h1))

    folds = list(
        GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(np.zeros(len(y)), y, groups)
    )
    print(f"[{short}] n={len(y)} pos={int(y.sum())} groups={len(np.unique(groups))}", flush=True)

    lens_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []

    for position in POSITIONS:
        lens_path = data_dir / f"lens_logp_{position}.npz"
        state_path = data_dir / f"dense_states_{position}.npy"
        if not lens_path.exists():
            print(f"[{short}] missing {lens_path}; skipping position", flush=True)
            continue
        with np.load(lens_path) as z:
            lens = z["lens"]
            row_orientation = z["orientation_id"].astype(str)
            row_condition = z["condition"].astype(str)
        key_to_row = {(o, c): i for i, (o, c) in enumerate(zip(row_orientation, row_condition))}
        idx_w = np.array([key_to_row[(o, "W")] for o in orientations])
        idx_g = np.array([key_to_row[(o, "G")] for o in orientations])
        n_slots = lens.shape[1]
        four_layers = [l for l in FOUR_LAYERS.get(short, []) if l < n_slots]

        # ---- logit lens: no learned probe at all -------------------------
        for slot in range(n_slots):
            lw = lens[idx_w, slot, :]
            lg = lens[idx_g, slot, :]
            lw_h1 = lw[rows_arange, h1_cols]
            lw_h2 = lw[rows_arange, h2_cols]
            lg_h1 = lg[rows_arange, h1_cols]
            lg_h2 = lg[rows_arange, h2_cols]
            variants = {
                "lens_effect_centered": pair_center(lw_h1 - lg_h1, pair_codes, n_pairs),
                "lens_margin_WmG": (lw_h1 - lw_h2) - (lg_h1 - lg_h2),
                "lens_sum_WmG": (lw_h1 + lw_h2) - (lg_h1 + lg_h2),
                "lens_W_margin": lw_h1 - lw_h2,
            }
            headline = slot in four_layers or slot == n_slots - 1
            for name, score in variants.items():
                n_boot = args.n_bootstrap if headline else 0
                boot = cluster_bootstrap_auroc(y, score, groups, n_boot, args.seed)
                lens_rows.append(
                    {
                        "model": short,
                        "position": position,
                        "slot": slot,
                        "is_final_norm_slot": slot == n_slots - 1,
                        "in_four_layer_sweep": slot in four_layers,
                        "lens_variant": name,
                        "n": boot["n"],
                        "auroc": boot["auroc"],
                        "auroc_ci_low": boot["ci_low"],
                        "auroc_ci_high": boot["ci_high"],
                        "pearson_vs_target": safe_pearson(y_cont, score),
                    }
                )

        # ---- dense probe sweep -------------------------------------------
        if not state_path.exists():
            print(f"[{short}] missing {state_path}; lens only for {position}", flush=True)
            continue
        states_all = np.load(state_path, mmap_mode="r")
        sw = np.ascontiguousarray(states_all[idx_w])
        sg = np.ascontiguousarray(states_all[idx_g])
        del states_all
        gc.collect()
        feature_sets = {
            "W_minus_G": (sw, sg),
            "W_hidden": (sw, None),
            "G_hidden": (sg, None),
        }
        layers = list(range(n_slots))
        for method, (base, diff) in feature_sets.items():
            t0 = time.time()
            preds = Parallel(n_jobs=args.n_jobs, backend="loky", max_nbytes="4M")(
                delayed(layer_fixed_c_oof)(base, layer, diff, y, folds, FIXED_C)
                for layer in layers
            )
            print(
                f"[{short}/{position}/{method}] dense sweep {n_slots} layers "
                f"in {time.time() - t0:.0f}s",
                flush=True,
            )
            point = [
                float(roc_auc_score(y[np.isfinite(p)], p[np.isfinite(p)]))
                if len(np.unique(y[np.isfinite(p)])) > 1
                else math.nan
                for p in preds
            ]
            best_slot = int(np.nanargmax(point)) if np.any(np.isfinite(point)) else 0
            for layer, pred in zip(layers, preds):
                headline = layer in four_layers or layer == n_slots - 1 or layer == best_slot
                boot = cluster_bootstrap_auroc(
                    y, pred, groups, args.n_bootstrap if headline else 0, args.seed
                )
                probe_rows.append(
                    {
                        "model": short,
                        "position": position,
                        "method": method,
                        "selection": f"fixed_C{FIXED_C}",
                        "slot": layer,
                        "is_final_norm_slot": layer == n_slots - 1,
                        "in_four_layer_sweep": layer in four_layers,
                        "is_dense_best": layer == best_slot,
                        "n": boot["n"],
                        "auroc": boot["auroc"],
                        "auroc_ci_low": boot["ci_low"],
                        "auroc_ci_high": boot["ci_high"],
                    }
                )
                if headline:
                    for i in range(len(y)):
                        pred_rows.append(
                            {
                                "model": short,
                                "position": position,
                                "method": method,
                                "selection": f"fixed_C{FIXED_C}",
                                "slot": layer,
                                "base_id": groups[i],
                                "swap_pair_id": pair_ids[i],
                                "orientation_id": orientations[i],
                                "binary_label": int(y[i]),
                                "pair_centered_effect": float(y_cont[i]),
                                "oof_prob": float(pred[i]) if np.isfinite(pred[i]) else math.nan,
                            }
                        )
            del preds
            gc.collect()

            if method == "W_minus_G" and not args.skip_nested:
                for tag, layer_set in [
                    ("nested_dense_layer_and_C", layers),
                    ("nested_four_layer_and_C", four_layers),
                ]:
                    if len(layer_set) < 2:
                        continue
                    t0 = time.time()
                    pred, fold_rows = nested_dense_oof(
                        base, diff, y, groups, folds, list(layer_set), args.n_jobs
                    )
                    boot = cluster_bootstrap_auroc(y, pred, groups, args.n_bootstrap, args.seed)
                    print(
                        f"[{short}/{position}/{method}/{tag}] auroc={boot['auroc']:.4f} "
                        f"in {time.time() - t0:.0f}s",
                        flush=True,
                    )
                    probe_rows.append(
                        {
                            "model": short,
                            "position": position,
                            "method": method,
                            "selection": tag,
                            "slot": -1,
                            "is_final_norm_slot": False,
                            "in_four_layer_sweep": tag.endswith("four_layer_and_C"),
                            "is_dense_best": False,
                            "n": boot["n"],
                            "auroc": boot["auroc"],
                            "auroc_ci_low": boot["ci_low"],
                            "auroc_ci_high": boot["ci_high"],
                        }
                    )
                    pd.DataFrame(fold_rows).to_csv(
                        data_dir / f"nested_folds_{position}_{method}_{tag}.csv", index=False
                    )
                    for i in range(len(y)):
                        pred_rows.append(
                            {
                                "model": short,
                                "position": position,
                                "method": method,
                                "selection": tag,
                                "slot": -1,
                                "base_id": groups[i],
                                "swap_pair_id": pair_ids[i],
                                "orientation_id": orientations[i],
                                "binary_label": int(y[i]),
                                "pair_centered_effect": float(y_cont[i]),
                                "oof_prob": float(pred[i]) if np.isfinite(pred[i]) else math.nan,
                            }
                        )
        del sw, sg, feature_sets
        gc.collect()

    return pd.DataFrame(lens_rows), pd.DataFrame(probe_rows), pd.DataFrame(pred_rows)


def analyze(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lens_all, probe_all, pred_all = [], [], []
    for short in args.model_shorts:
        if not (out_dir / short).exists():
            print(f"[{short}] no extraction dir; skipping", flush=True)
            continue
        lens_df, probe_df, pred_df = analyze_model(args, short)
        lens_all.append(lens_df)
        probe_all.append(probe_df)
        pred_all.append(pred_df)
        lens_df.to_csv(out_dir / short / "lens_summary.csv", index=False)
        probe_df.to_csv(out_dir / short / "dense_probe_summary.csv", index=False)
        pred_df.to_csv(out_dir / short / "dense_probe_oof_predictions.csv.gz", index=False)

    if lens_all:
        pd.concat(lens_all, ignore_index=True).to_csv(out_dir / "logit_lens_summary.csv", index=False)
    if probe_all:
        combined = pd.concat(probe_all, ignore_index=True)
        combined.to_csv(out_dir / "dense_probe_summary.csv", index=False)
        if "selection" in combined.columns:  # absent when no position had probe states
            print("NESTED / SELECTION ROWS", flush=True)
            print(
                combined[combined["selection"] != f"fixed_C{FIXED_C}"]
                .round(4)
                .to_string(index=False),
                flush=True,
            )
    if pred_all:
        pd.concat(pred_all, ignore_index=True).to_csv(
            out_dir / "dense_probe_oof_predictions.csv.gz", index=False
        )
    print("analysis done", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/prompts/")
    parser.add_argument("--source-dir", required=True,
                        help="role-swap run with role_swapped_manifest.csv and role_swapped_effect_rows_all.csv")
    parser.add_argument("--output-dir", default="outputs/roleswap_logit_lens")
    parser.add_argument("--models", nargs="+", default=["mistral_7b", "gemma_9b", "llama_8b"])
    parser.add_argument("--model-shorts", nargs="+", default=["mistral", "gemma", "llama"])
    parser.add_argument("--answer-prefix", default="\nFinal answer:")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--use-hf-token", action="store_true")
    parser.add_argument("--skip-nested", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    if args.extract:
        for model_key in args.models:
            extract_model(args, model_key)
    if args.analyze:
        analyze(args)
    if not args.extract and not args.analyze:
        parser.error("specify --extract and/or --analyze")


if __name__ == "__main__":
    main()
