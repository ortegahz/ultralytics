---
name: three-track-gatekeeper
description: |
  Run the three-track gate on a candidate checkpoint (T1 Anti-UAV official-domain source macro /
  T2 Longquan golden pool / T3 blind hard-case sequence) and issue an auditable release verdict
  through the serial three-level funnel: level-1 dual-channel red line -> level-2 Pareto dominance
  -> level-3 same-baseline utility score U. Load this skill whenever the user asks to "给模型过门禁",
  "跑三轨门禁", "出个门禁报告", "这个 checkpoint 能不能上线", "候选和 RC1 比谁好", "三级漏斗判一下",
  "重测一下这个 trial", or asks to judge EXACT_REPRO / BAND_REPRO / NOT_REPRO, or to arbitrate
  candidates by U score. Also load it when a gate_report.json is incomplete or suspected corrupted.
  Do NOT use it to launch or babysit the Optuna hyperparameter search (use hparam-search-operator),
  to train or fine-tune a model, or to curate/label datasets.
descriptions:
  zh-Hans: "对候选权重跑三轨门禁（源域大盘 / 黄金池 / 盲测难例），用一级双通道红线→二级帕累托→三级同基准效用分的串行漏斗给出可审计的上线判决。"
displayNames:
  zh-Hans: "三轨门禁与裁决"
---

# Three Track Gatekeeper

Issue a release verdict on a candidate checkpoint using the project's serial three-level funnel.
Every number in this skill comes from `manu/training/mix_ratio_judgement.py`,
`manu/training/auto_epoch_gate_check.py` and `manu/memory/memory.md` — re-read those instead of
trusting a paraphrase when a value matters.

## Inputs to collect

| Item | Where it comes from | Default |
|:--|:--|:--|
| Candidate weights | user-supplied `.pt`, or `runs/optuna_mix_ratio/trials/trial_XXXX/weights/epochK.pt` | required |
| Candidate gate report | `<trial>/gate/epochK/gate_report.json` | run the gate if absent |
| Reference | `frozen` / `rc1` (RC1/Ep3) / `rc1_epoch5` / another `gate_report.json` | `rc1` |
| Measurement epoch (caliber) | trial log line `[GATE] 三轨门禁（... / epochK）` | 3 |
| Channel | strict only, or strict + relaxed | both |
| Pareto dimension set | `user` / `core` / `full`, or an explicit dict | `full` |
| Free GPU | `nvidia-smi` on the server | required |

The three tracks are fixed and defined in `manu/training/auto_epoch_gate_check.py:52-76`:

- **T1 = `track1_uav`** — Anti-UAV official-domain val subset (31,613 frames / 25,111 GT).
  Frozen baseline F1 0.905736, TP 21624, FP 1014.
- **T2 = `track2_golden`** — Longquan golden pool, 7 sequences (17,187 frames / 16,638 GT).
  Frozen baseline F1 0.988436, FP 26, FN 355. This is the anti-mixing guard.
- **T3 = `track3_blind`** — blind hard-case sequence `VIDEO00007_19700101_001324`
  (11,810 frames / 10,535 GT). Frozen baseline F1 0.837331, FP 590, FN 2523.

If the user has not said which measurement epoch the candidate was gated at, read it from the
trial log before judging — the same weights mean different things at Ep3 and Ep5.

## Procedure

### 1. Pin the caliber before reading any number

Caliber is the product of two independent axes, and conflating them is the single most common
source of wrong verdicts:

- **cosine period** — read `[SCHED] ... cosine 周期 N ep` from `runs/optuna_mix_ratio/logs/trial_XXXX.log`.
  5 ep means the same LR trajectory as RC1; 3 ep is the legacy period whose last-epoch LR is only
  55% of RC1's.
- **measurement epoch** — read `[GATE] 三轨门禁（... / epochK）`. Parse with the regex `epoch(\d+)`;
  the legacy format is `/ epochK` and the current format is `/ epochK（末 epoch）`
  (`manu/training/status_mix_ratio_search.py:152`).

Group names and their RC1 reference systems (`manu/training/status_mix_ratio_search.py:105-114`):

| Group | RC1 reference | Note |
|:--|:--|:--|
| `RC1-Ep3口径` | `RC1_EPOCH3` (U = +0.027579) | current champion |
| `RC1-Ep5口径` | `RC1_EPOCH5` (U = +0.026260) | last epoch of the same 5-epoch run |
| `旧调度口径` | **none** | cosine 3 ep ⇒ U is systematically low; list only, never duel |
| `未知口径` | none | log missing or unparsable |

Three boundary points, keep them separate (`manu/memory/memory.md:184-190`): scheduling caliber from
`trial_0008`, red-line caliber from `trial_0011`, measurement-epoch switch at `trial_0012`.

### 2. Confirm the scripts on the server match the local repo

Every command below targets the Linux server. From a Windows workstation, drive the server over
SSH rather than translating the commands to PowerShell — `md5sum`, `ssh` and the conda activation
have no equivalent here, and the GPU evaluation only exists on the server.

The project rule 〇 ("local first") makes the local repo `/media/manu/1TB-Volume/workspace/ultralytics`
the only place to edit code, and the SSHFS mirror `/home/manu/mnt/pycharm_project_10ae9e2e` writes
straight through to the server `/tmp/pycharm_project_10ae9e2e`. Synchronisation can only be proven
by a three-side md5 comparison, never by "I copied it".

```bash
# 【本地】
md5sum /media/manu/1TB-Volume/workspace/ultralytics/manu/training/mix_ratio_judgement.py \
       /media/manu/1TB-Volume/workspace/ultralytics/manu/training/auto_epoch_gate_check.py
# 【本地】SSHFS 镜像
md5sum /home/manu/mnt/pycharm_project_10ae9e2e/manu/training/mix_ratio_judgement.py \
       /home/manu/mnt/pycharm_project_10ae9e2e/manu/training/auto_epoch_gate_check.py
# 【服务器】
ssh -p 32222 huangzhe@192.168.99.40 "md5sum /tmp/pycharm_project_10ae9e2e/manu/training/mix_ratio_judgement.py /tmp/pycharm_project_10ae9e2e/manu/training/auto_epoch_gate_check.py"
```

If the three hashes disagree, sync first (copy local → SSHFS mirror, which lands on the server)
and re-compare. A running process keeps executing the code it loaded at start, so a code change
never reaches an already-running dispatcher or worker until it is restarted.

### 3. Run the gate (only if no usable report exists yet)

Never share a GPU between a probe/evaluation task and a running training trial — this caused a
real CUDA OOM that nulled out `tracks.track1_uav.finetune_peak.f1` and wasted a 27-minute train.
One card may host "training" or "evaluation", never both.

```bash
# 【服务器】cd /tmp/pycharm_project_10ae9e2e && source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
# 注意服务器上 `python` 没有 3 后缀
python manu/training/auto_epoch_gate_check.py \
  --weights runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt \
  --out-dir  runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3 \
  --gpus 1 --device 3 --infer-gpu 3 --cache-gpus 3 \
  --baseline-cache runs/mixed_ft_route_b/baseline/uav_cache_deP0.pkl \
  --infer-batch 4
```

CLI of `manu/training/auto_epoch_gate_check.py:147-190`: `--weights --out-dir --strict-gate --gpus
--track1-strict --track3-strict --min-track1-f1 --min-track2-f1 --max-track2-fp --skip-track1
--skip-longquan --infer-batch --cache-batch-size --cache-chunk-size --baseline-cache
--baseline-weights --device --infer-gpu --cache-gpus`.

**`--max-track2-fp` and `--min-track1-f1` are absolute thresholds, not increments**
(defaults 36 and 0.905736). Passing an increment there silently mis-judges the candidate.
To express "baseline 26 plus an increment of 10", pass `36`; to express "baseline 0.905736 minus
0.005", pass `0.900736`.

Without `--strict-gate` the script always exits 0, but `gate_report.json` is still written
unconditionally (`manu/training/auto_epoch_gate_check.py:493-504`). Read
`report["verdict"]` / `report["veto_failures"]` from the JSON; never use the process exit code
alone to decide pass/fail in automation.

### 4. Repair an incomplete report instead of discarding it

If `tracks.track1_uav.finetune_peak.f1` is null, the gate infrastructure failed (usually a Track1
OOM) — that is not a metric miss. `metrics_from_gate_report()` raises `GateReportIncomplete`
naming the missing fields (`manu/training/mix_ratio_judgement.py:707-726`), which prevents a
`float(None)` TypeError from destroying the run's diagnostics.

Re-run Track1 alone on an idle card, then merge it back:

```bash
# 【服务器】
python manu/training/auto_epoch_gate_check.py \
  --weights runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt \
  --out-dir  runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3_retry \
  --gpus 1 --device 3 --infer-gpu 3 --cache-gpus 3 \
  --baseline-cache runs/mixed_ft_route_b/baseline/uav_cache_deP0.pkl \
  --skip-longquan --infer-batch 4
python manu/training/merge_gate_report.py \
  --main  runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/gate_report.json \
  --retry runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3_retry/gate_report.json
```

`merge_gate_report.py --main --retry [--no-backup]` keeps a `*.incomplete.json` copy of the broken
original (audit trail) and exits non-zero without touching the main report on failure.

### 5. Level 1 — dual-channel red line (one-vote veto)

Evaluate against the **frozen baseline**, never against the previous trial. Thresholds live in
`manu/training/mix_ratio_judgement.py:80-82` (strict) and `:323-328` (relaxed).

**Strict channel** — all three must hold:

| Rule | Threshold |
|:--|:--|
| T2 golden pool F1 | ≥ 0.985 |
| ΔFP_T2 vs frozen baseline | ≤ +10 |
| ΔF1_T1 vs frozen baseline | ≥ −0.005 |

**Controlled relaxed channel** — only reachable when the strict channel fails, and only if the
golden pool and the hard case both gained materially:

| Rule | Threshold |
|:--|:--|
| T2 golden pool F1 | ≥ 0.985 |
| ΔFP_T2 | ≤ +15 |
| ΔF1_T2 | ≥ +0.008 |
| ΔF1_T3 | ≥ +0.020 |
| ΔF1_T1 | ≥ −0.008 |

A candidate that clears the relaxed channel must be labelled `PASS_RELAXED` explicitly — never
silently admitted. `allow_relaxed=False` in the API and `--strict-only` on the CLI fall back to
the strict wording in one switch.

**Why the relaxed channel is defensible** (calibration evidence, all from real trials):
mixed training leaves T2 FP at 32/37/37/37, i.e. Δ +6~+11, and the single-trial noise on T2 FP is
about ±5 — so the original +10 line sits in the middle of the noise band and wrongly rejected 3 of
4 strong candidates. In the same batch every T2 F1 was ≥ 0.9987 (Δ ≥ +0.0103; FN 355 → 6),
ΔF1_T1 ∈ [−0.005417, −0.003281] (RC1 itself −0.003818), and ΔF1_T3 ∈ [+0.023727, +0.028133].
The relaxed thresholds therefore still demand real gains on both target tracks while tolerating
the source-domain cost that is empirically normal.

Failing both channels ⇒ verdict `FAIL_TRACK2_SAFETY` or `FAIL_TRACK1_RETREAT`, and the candidate
is pruned — it never reaches level 2.

### 6. Level 2 — Pareto dominance with an explicit dimension set

`PARETO_PRESETS` in `manu/training/mix_ratio_judgement.py:273-288`:

| Preset | Dimensions |
|:--|:--|
| `core` | T1 F1↑, T2 F1↑, T3 F1↑ |
| `full` (default) | the three F1↑ plus T1 FP↓, T2 FP↓, T3 FP↓ |
| `user` | T1 F1↑, T2 F1↑, T3 FP↓ (the dimension set implied by the user's Ep3-vs-Ep4 example) |

The dimension set decides whether an "uncontested win" exists, so it must be stated in the report
rather than left implicit. Empirically, under `full` the claim "Ep3 ≻ Ep4" does not hold (Ep3 T3
F1 0.871684 is below Ep4 0.872648), yet the utility score also picks Ep3 — RC1/Ep3 being optimal
is confirmed by two independent paths.

If level 2 holds, declare the win and stop; do not re-open the utility score.

### 7. Level 3 — same-baseline utility score

Only for candidates that are mutually attacking (level 2 did not settle it). Recompute from the
raw metrics with `utility()` (`manu/training/mix_ratio_judgement.py:434-442`):

```
U = ΔF1_T3 + 0.5·ΔF1_T2 − 2.5·max(0, −ΔF1_T1) − 0.0005·max(0, ΔFP_T2)
```

Coefficients: α = 0.5 golden-pool consolidation, β = 2.5 source-domain penalty multiplier,
γ = 0.0005 per new golden-pool false positive. All deltas are against the **frozen baseline**.

Two non-negotiable implementation rules:

1. **Compare same-baseline U values.** The `max(0, ·)` penalty terms make U non-antisymmetric:
   measured `U(Ep3 vs Ep4) = −0.000903` while `U(Ep4 vs Ep3) = −0.001097` — same sign. Pairwise
   comparison therefore contradicts the objective; comparing `U(Ep3) = +0.027579` against
   `U(Ep4) = +0.026260` on the same baseline is the correct decision
   (`manu/training/mix_ratio_judgement.py:579-584`).
2. **Never collapse the three tracks into one weighted sum.** The funnel is serial: veto, then
   Pareto, then same-baseline U. A weighted sum would let a source-domain collapse be "averaged
   away" by a target-domain jump, which this project explicitly forbids.

**Recompute U; do not trust the `utility` column in `optuna_summary.csv`.** A real mismatch
between the stored column and the recomputed value has already been found, and PRUNED rows store
the string `"None"`.

Benchmarks for the same-baseline U: RC1/Ep3 = **+0.027579** (the pass line), RC1/Ep5 =
**+0.026260** (the same 5-epoch run's last epoch). To call a win "significant" against the
champion, the project uses U ≳ 0.045 (+2σ).

CLI for a two-way comparison (`manu/training/mix_ratio_judgement.py:772-787`):

```bash
# 【服务器】cd /tmp/pycharm_project_10ae9e2e && conda activate uav
python manu/training/mix_ratio_judgement.py \
  --candidate runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/gate_report.json \
  --baseline  rc1 \
  --out-json  runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/verdict.json
# 加 --strict-only 可复现"仅严格红线"的旧口径结论
```

### 8. (When the question is "did it reproduce?") — reproduction verdict

Use `reproduction_check(cand, epoch)` (`manu/training/mix_ratio_judgement.py:211-258`) plus
`rc1_epoch_band()` (`:186-199`):

| Verdict | Condition |
|:--|:--|
| `EXACT_REPRO` | every metric within `EXACT_TOL` of RC1[epoch] (T1 F1 ±2e-3, T1 FP ±30, T2 F1 ±1e-3, T2 FP ±4, T3 F1 ±3e-3, T3 FP ±40) |
| `BAND_REPRO` | outside the tolerances but inside RC1's own epoch-1..5 envelope ⇒ normal fluctuation |
| `NOT_REPRO` | outside that envelope ⇒ a genuine anomaly |

The envelope exists because RC1's own adjacent epochs already swing: T3 F1 0.8469~0.8749 (span
0.028), T1 FP 941~1232 (span 291), T2 FP 31~35. Treating one point as "the value it should
reproduce to" manufactures false failures.

Batch form (`manu/training/check_control_repro.py --out-root --trials --epoch --all`):

```bash
# 【服务器】
python manu/training/check_control_repro.py --all
```

### 9. Report the verdict

State the verdict chain, the caliber, the reference, the dimension set, and the artifact paths
with the full command and the execution machine (project rule ⑨). If the candidate only cleared
the relaxed channel, say so in the headline.

## Output contract

Report in this shape (order matters — it mirrors the funnel):

```
Verdict            : PASS | PASS_RELAXED | FAIL
Caliber            : cosine 5ep / measured epoch 3  → reference RC1/Ep3 (U=+0.027579)
Level 1 (red line) : PASS (strict) | PASS_RELAXED (relaxed; ΔFP_T2=+7, ΔF1_T2=+0.010513, ΔF1_T3=+0.060363, ΔF1_T1=-0.003826) | FAIL_TRACK2_SAFETY / FAIL_TRACK1_RETREAT
Level 2 (Pareto)   : dims=full → A_DOMINATES | mutually attacking
Level 3 (utility)  : U(candidate)=+0.052554 (same baseline) vs U(reference)=+0.027579 → UTILITY_A
Reproduction       : EXACT_REPRO | BAND_REPRO | NOT_REPRO   (only when reproduction was the question)
Artifacts          : <gate_report.json path>, <verdict.json path>
Command            : <full copy-pasteable command>  (execution machine: server / local)
```

If any level was skipped, say which and why. Do not emit a bare "PASS".

## Failure handling

- **Gate report missing a required field** → treat it as an infrastructure failure, not a metric
  miss; repair via `merge_gate_report.py`. Never let a `TypeError` stand in for a verdict.
- **`--strict-gate` absent** → exit code is always 0; read `verdict` and `veto_failures` from
  `gate_report.json` instead.
- **No idle GPU** → wait or pick another card; do not run an evaluation on a card that is training.
  Report the wait to the user rather than forcing an OOM.
- **Candidate from the legacy scheduling caliber** → it has no same-scale RC1 reference; list it
  and label it `旧调度口径`, but exclude it from the duel. Running a duel on a group with a
  `None` reference has crashed before.
- **Log lines unparsable** → parse `[GATE]` with the regex `epoch(\d+)`; treat a trial with no
  `[SCHED]`/`[GATE]` line as `未知口径` instead of guessing.
- **Candidate was gated with an absolute threshold mistake** (increment passed where absolute was
  expected) → re-run the gate with the absolute values 36 / 0.900736; do not patch the numbers
  by hand.
- **Verdict flips after a caliber switch** → report both calipers explicitly (e.g. "vs RC1/Ep5
  same caliber" and "vs RC1/Ep3 user goal") rather than silently switching the reference.
- **Threshold or caliber change requested by the user** → stop and report; editing
  `DEFAULT_RELAXED_GATE` or the frozen baseline changes the meaning of every historical verdict and
  requires explicit authorisation.
- **Any change to code** → make it locally first, then the three-side md5 comparison, then report
  "synced". Never edit the server copy directly.

## Examples

### Example 1 — a strictly-clearing candidate vs the champion

Input: `trial_0080` (mix `w2.5`, measured 1:2.524, lr_scale 2.2702), gated at epoch 3, compared
with RC1/Ep3.

Measured metrics (T1 F1/FP, T2 F1/FP, T3 F1/FP): `0.901910 / 1051`, `0.998949 / 33`,
`0.897694 / 482`.

- Level 1 strict: T2 F1 0.998949 ≥ 0.985 ✔ ; ΔFP_T2 = 33 − 26 = +7 ≤ +10 ✔ ;
  ΔF1_T1 = 0.901910 − 0.905736 = −0.003826 ≥ −0.005 ✔ ⇒ `PASS` (strict, no relaxation needed).
- Level 2 with `dims=full`: RC1/Ep3 still holds T2 F1 0.998979 and T3 F1 0.871684, so the two
  candidates mutually attack ⇒ no uncontested Pareto win; the funnel proceeds to level 3.
- Level 3 same-baseline U:
  `ΔF1_T3 = 0.897694 − 0.837331 = +0.060363`,
  `0.5·ΔF1_T2 = 0.5 × (0.998949 − 0.988436) = +0.005257`,
  `−2.5·max(0, 0.003826) = −0.009566`,
  `−0.0005·max(0, 7) = −0.003500`
  ⇒ U = 0.060363 + 0.005257 − 0.009566 − 0.003500 = **+0.052554** vs RC1/Ep3 **+0.027579**
  ⇒ beats the champion by +0.024975, above the +0.045 significance band.

Output: `Verdict: PASS`, level 1 strict, level 2 mutually attacking, level 3 `UTILITY_A`,
U +0.052554, and the note that the gain came from lr_scale rather than the mixing ratio (a
separate attribution finding).

### Example 2 — the caliber switch, and why Ep5 is a different champion

Input: "the same 5-epoch run's last epoch, can it be released?"

RC1/Ep5 metrics: `0.900501 / 1023`, `0.998949 / 33`, `0.874924 / 502`.

- ΔF1_T1 = 0.900501 − 0.905736 = **−0.005235**, which is below the strict line −0.005 ⇒ strict
  `FAIL_TRACK1_RETREAT`.
- Relaxed channel: T2 F1 0.998949 ≥ 0.985 ✔ ; ΔFP_T2 = +7 ≤ +15 ✔ ; ΔF1_T2 = +0.010513 ≥ +0.008 ✔ ;
  ΔF1_T3 = 0.874924 − 0.837331 = +0.037593 ≥ +0.020 ✔ ; ΔF1_T1 = −0.005235 ≥ −0.008 ✔
  ⇒ `PASS_RELAXED`, which must be labelled explicitly.
- Same-baseline U = +0.026260, **below** RC1/Ep3's +0.027579.

Output: the verdict is `PASS_RELAXED`, but the reference for any 5-epoch trial must be
`RC1_EPOCH5` (+0.026260), not `RC1_EPOCH3` — comparing a 5-epoch trial against the Ep3 champion
mixes measurement scales and is not a valid duel.
