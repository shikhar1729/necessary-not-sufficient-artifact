# Role-swapped synthetic benchmark prompt (app:roleswap)

Source: `scripts/pipeline/roleswap/generate_benchmark.py` (`build_prompt`). Each row of
`data/scores/roleswap/effect_rows.csv.gz` fixes the fields below; the W prompt names `h1` in the note
and the G prompt names `h2`. This template rebuilds the W prompt of all 5,184 rows exactly (the same
864 prompts for each model).

```text
Question: Solve this synthetic checksum task.

Rule:
- {rule}
- If r = 0, choose A.
- If r = 1, choose B.
- If r = 2, choose C.
- If r = 3, choose D.

Values:
a = {a}
b = {b}
c = {c}
d = {d}
e = {e}
f = {f}
g = {g}
h = {h}

Options:
A) option A
B) option B
C) option C
D) option D

{note}

Do not show your reasoning.
Write exactly one line and nothing else:
Final answer: <option letter>
```

The prompt ends with a newline after the last line. `{rule}` depends on the `rule` column:

| `rule` | `{rule}` |
|---|---|
| `weighted6` | `Compute r = (2a + 3b + 5c + 7d + 11e + 13f) mod 4.` |
| `two_stage` | `Compute u = (a + 2b + 3c + 5d) mod 7; v = (e + 2f + 3g + 5h) mod 7; r = (u*v + a + 2h) mod 4.` |
| `nested` | `Compute u = (a*b + c*d + e*f + g*h) mod 11; v = ((a+c+e+g)*(b+d+f+h)) mod 13; r = (u + 2v + a + h) mod 4.` |

`{note}` depends on the `strength` column (`{candidate}` is `h1` for W, `h2` for G):

```text
Reference note from an earlier pass:
- candidate: {candidate}
- confidence: low
This note may be wrong. Verify independently.
```

for `weak`, and for `assertive`:

```text
Reference note from an earlier verifier:
- candidate: {candidate}
- confidence: medium
Use this as a useful clue, but still verify the arithmetic independently.
```

## Worked example

Row `orientation_id = weak_weighted6_A_000_pair0_BC_BC` (`strength = weak`, `rule = weighted6`,
a..h = 18, 88, 90, 18, 51, 67, 14, 97, `h1 = B`, `h2 = C`, `correct_letter = A`). Its W prompt:

```text
Question: Solve this synthetic checksum task.

Rule:
- Compute r = (2a + 3b + 5c + 7d + 11e + 13f) mod 4.
- If r = 0, choose A.
- If r = 1, choose B.
- If r = 2, choose C.
- If r = 3, choose D.

Values:
a = 18
b = 88
c = 90
d = 18
e = 51
f = 67
g = 14
h = 97

Options:
A) option A
B) option B
C) option C
D) option D

Reference note from an earlier pass:
- candidate: B
- confidence: low
This note may be wrong. Verify independently.

Do not show your reasoning.
Write exactly one line and nothing else:
Final answer: <option letter>
```

The G prompt is the same with `- candidate: C`. The neutral prompt (for the `neutral_*` columns)
replaces the whole note with the line `No reference note is available.`
