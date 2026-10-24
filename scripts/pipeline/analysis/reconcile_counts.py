"""Reconcile the strict retained slices (N=421/384/260) with a rebuild that only loosens answer matching
(N=427/398/260), on identical trace IDs (app:corrected). The initial 352/326/196 slices came from stored labels
and are not computed here. Input: --trace-root with traces/{model}_traces.jsonl. Output: printed counts.
"""
import argparse
import collections
import json

from src.core.normalize import normalize_answer, answers_match   # strict (MCQ exact)


def lenient_match(answer, gold):
    """Lenient rule of the initial pipeline: answers_match WITHOUT the MCQ-letter special case."""
    a, t = normalize_answer(answer or ''), normalize_answer(gold or '')
    if not a or not t: return False
    return a == t or a in t or t in a

def population(rows, match):
    """Return {gid: label} plus the reason each excluded group was dropped."""
    keep, why = {}, {}
    for gid, c in rows.items():
        w, g = c.get('W'), c.get('G')
        if not w or not g:                      why[gid]='missing W or G'; continue
        if not w.get('parse_ok') or not g.get('parse_ok'): why[gid]='parse failure'; continue
        h1, wa, ga = w.get('candidate'), w.get('normalized_final'), g.get('normalized_final')
        if not wa or not match(wa, h1):          why[gid]='W did not follow the hint'; continue
        if match(wa, w.get('gold_answer')):      why[gid]='W answer also matched gold'; continue
        if not ga:                               why[gid]='G answer missing'; continue
        keep[gid] = int(not match(ga, h1))
    return keep, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace-root', required=True, help='trace store with traces/{model}_traces.jsonl')
    trace_root = ap.parse_args().trace_root
    for model in ['mistral','gemma','llama']:
        rows = collections.defaultdict(dict)
        for line in open(f'{trace_root}/traces/{model}_traces.jsonl'):
            r = json.loads(line); rows[(r['dataset'], r['question_id'])][r['condition']] = r
        S, sw = population(rows, answers_match)
        L, lw = population(rows, lenient_match)
        both, s_only, l_only = set(S)&set(L), set(S)-set(L), set(L)-set(S)
        print(f"\n{'='*78}\n{model.upper()}   strict N={len(S)}   lenient N={len(L)}   (delta {len(S)-len(L):+d})")
        print(f"  both={len(both)}  strict-only={len(s_only)}  lenient-only={len(l_only)}")
        print("  why strict keeps rows lenient drops:", dict(collections.Counter(lw[g] for g in s_only)))
        print("  why lenient keeps rows strict drops:", dict(collections.Counter(sw[g] for g in l_only)))
        bd = collections.Counter(g[0] for g in s_only); bl = collections.Counter(g[0] for g in l_only)
        print("  strict-only by dataset:", dict(bd), "| lenient-only by dataset:", dict(bl))
        flips = sum(1 for g in both if S[g]!=L[g])
        print(f"  label flips among the {len(both)} shared rows: {flips}")


if __name__ == "__main__":
    main()
