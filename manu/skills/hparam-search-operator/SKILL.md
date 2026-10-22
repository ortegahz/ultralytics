---
name: hparam-search-operator
description: |
  Operate the Optuna multi-GPU hyperparameter search for the mixed-training ratio study: start it
  as a detached dispatcher, keep it alive across SSH disconnects, resume it from SQLite, inject
  control/enqueue trials, watch its progress, and harvest its logs and result files. Load this
  skill whenever the user asks to "开超参搜索", "启动 Optuna 搜索", "搜索跑到哪了", "把搜索停掉",
  "断点续跑", "投放一个对照 trial", "复现一下某个 trial", "扩搜索空间", or wants to change the
  search protocol (epochs, gate epoch, workers) without restarting. Also load it when a trial
  crashed, is stuck in RUNNING, or when the dispatcher was killed and must be resumed. Do NOT use
  it to judge a candidate's release verdict (use three-track-gatekeeper), to train a single
  final model, or to curate datasets.
descriptions:
  zh-Hans: "操作 Optuna 多卡并发超参搜索：后台启动与守护、断点续跑、异常熔断、自愈、协议热改、对照投放与结果收割。"
displayNames:
  zh-Hans: "超参搜索操作规程"
---

# Hparam Search Operator

Operate `manu/training/optuna_mix_ratio_search.py` — one file that is both the dispatcher and the
worker (`--worker` switches roles). The dispatcher owns the study; each trial runs in a fresh
worker process on one GPU.

## Inputs to collect

| Item | Where it comes from | Default |
|:--|:--|:--|
| Authorisation | explicit user approval for this run | required, see step 1 |
| Output root | `--output-root` | `runs/optuna_mix_ratio` |
| Study name | `--study-name` | `mix_ratio_search` |
| GPUs | `--gpus` | `0,1,2,3` |
| Trial budget | `--n-trials` / `--max-hours` | 4096 / 20.0 h |
| Epochs per trial | `--epochs` (train) vs `--sched-epochs` (cosine period, default 5) | 3 / 5 |
| Per-trial overrides | `user_attrs{seed, workers, gate_epoch}` written at enqueue time | none |
| Baseline cache | `--baseline-cache` | `runs/mixed_ft_route_b/baseline/uav_cache_deP0.pkl` |

Execution machine for everything below is the **server**:
`ssh -p 32222 huangzhe@192.168.99.40`, project root `/tmp/pycharm_project_10ae9e2e`,
environment `source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav`.
The server's `python` has **no** `3` suffix; the local machine only has `python3` (3.14.4, no
pytest), so local runs are limited to syntax checks and pure-python helpers.

## Procedure

### 1. Get authorisation before touching the search

Project rule 〇-之三 and rule ①: starting a search, extending the search space, or changing the
protocol are all "heavy experiments" that need explicit approval in the current conversation.
An approved 3-epoch search does not imply approval for a 5-epoch one, and neither implies
approval to re-launch a stopped dispatcher. If the user has not said so, report and ask instead of
starting.

### 2. Sync the code, then prove it with md5

Local is the only place to edit (`/media/manu/1TB-Volume/workspace/ultralytics`); the SSHFS mirror
`/home/manu/mnt/pycharm_project_10ae9e2e` writes straight through to the server
`/tmp/pycharm_project_10ae9e2e`. Synchronisation is provable only by a three-side md5 comparison.

```bash
# 【本地】
md5sum /media/manu/1TB-Volume/workspace/ultralytics/manu/training/optuna_mix_ratio_search.py \
       /media/manu/1TB-Volume/workspace/ultralytics/manu/training/enqueue_control_trials.py
# 【本地】SSHFS 镜像
md5sum /home/manu/mnt/pycharm_project_10ae9e2e/manu/training/optuna_mix_ratio_search.py \
       /home/manu/mnt/pycharm_project_10ae9e2e/manu/training/enqueue_control_trials.py
# 【服务器】
ssh -p 32222 huangzhe@192.168.99.40 "md5sum /tmp/pycharm_project_10ae9e2e/manu/training/optuna_mix_ratio_search.py /tmp/pycharm_project_10ae9e2e/manu/training/enqueue_control_trials.py"
```

### 3. Remember that a running process does not read new code

A dispatcher keeps executing the module it imported at start-up. Editing
`optuna_mix_ratio_search.py` has **no effect** on the running dispatcher or on workers it
spawns later; only a restart loads the new code. This has already burned a real result: a
"bit-level reproduction" was believed to be in effect while the live dispatcher was still the
old build from hours earlier, and the `user_attrs` override logic had been written into the file
after that process started. Whenever you need new behaviour, plan a restart and say so.

Note the asymmetry that made the old run still usable: the dispatcher starts a **fresh worker
process per trial**, so new worker code does take effect for later trials as long as the new
parameters have defaults that preserve the old behaviour.

### 4. Start the search detached

The launcher's `setsid` + `nohup` + `disown` trio and the `ps -eo args | grep -c` probe are
POSIX-only, and the search needs the server's 4 GPUs. From a Windows workstation, run this on
the Linux server (or WSL) instead of translating it to PowerShell.

Use the launcher script; it sets up `setsid` + `nohup` + `disown`, writes the PID file and
guards against a duplicate launch.

```bash
# 【服务器】cd /tmp/pycharm_project_10ae9e2e
bash manu/training/launch_mix_ratio_search.sh              # default 20 h, 4 GPUs, 3 epochs
MAX_HOURS=8 bash manu/training/launch_mix_ratio_search.sh  # custom budget
N_TRIALS=200 EPOCHS=5 bash manu/training/launch_mix_ratio_search.sh
```

Environment knobs read by `manu/training/launch_mix_ratio_search.sh:21-27`: `PROJECT_ROOT`,
`OUT_ROOT`, `GPUS`, `N_TRIALS`, `EPOCHS`, `MAX_HOURS`, `STUDY_NAME`. The launcher calls the
dispatcher with `--output-root --study-name --gpus --n-trials --epochs --max-hours`.

The dispatcher runs in its own session with **PPID = 1**, so a power cut or a closed SSH window
does not stop it. Duplicate launches are refused with `[ABORT] 搜索已在运行（PID …）` because the
launcher finds a live PID in `<out_root>/dispatcher.pid`; a stale PID file is overwritten with a
`[WARN]`.

Verify the dispatcher is alive by counting its processes — **use `ps -eo args | grep -c`, not
`pgrep -f`**, which also matches your own shell and miscounts:

```bash
# 【服务器】
ps -eo args | grep -c "[o]ptuna_mix_ratio_search.py"
cat runs/optuna_mix_ratio/dispatcher.pid
```

### 5. Change the protocol of later trials without restarting

The dispatcher re-reads `<out_root>/runtime_config.json` before every dispatch
(`read_runtime_config()` at `manu/training/optuna_mix_ratio_search.py:212-234`). Recognised keys:
`train_epochs` (int), `gate_epoch` (int, 0/absent = last epoch), `workers` (int). A malformed file
is ignored and the start-up arguments are used instead — it can never kill the dispatcher.

```bash
# 【服务器】switch later trials to 5 epochs while keeping the cosine period at 5
cat > runs/optuna_mix_ratio/runtime_config.json <<'EOF'
{"train_epochs": 5}
EOF
```

`--sched-epochs` (default **5**) is what decouples the cosine period from the number of training
steps; setting it to 3 reproduces the legacy period whose last-epoch LR was only 55% of RC1's.
The trial result records `train_epochs / gate_epoch / sched_epochs`
(`manu/training/optuna_mix_ratio_search.py:633-635`), so the caliber of each trial stays auditable.

### 6. Inject control trials into the running study

Enqueued trials are taken first because the dispatcher uses `study.ask()`; no restart is needed.

```bash
# 【服务器】
python manu/training/enqueue_control_trials.py --list                        # show WAITING trials
python manu/training/enqueue_control_trials.py --specs natural,w4.0 --dry-run
python manu/training/enqueue_control_trials.py --specs natural,w4.0
python manu/training/enqueue_control_trials.py --specs natural --exact        # bit-level reproduction
python manu/training/enqueue_control_trials.py --from-trial trial_0080 --count 2
```

CLI of `manu/training/enqueue_control_trials.py:82-97`: `--storage --study-name --specs --dry-run
--list --exact --from-trial --count`. `--exact` means seed 0 / workers 16 / gate epoch 3 — the
RC1-aligned configuration (`enqueue_control_trials.py:65-67`); `--from-trial <trial> --count N`
copies that trial's exact hyper-parameters N times, each with a different trial number and
therefore a different seed.

The per-trial override lands in `trial.user_attrs{seed, workers, gate_epoch}`, and the dispatcher
reads it at dispatch time (`optuna_mix_ratio_search.py:901-905`), defaulting the seed to
`args.seed + trial.number`.

### 7. Watch progress

```bash
# 【服务器】
python manu/training/status_mix_ratio_search.py --top 20
python manu/training/status_mix_ratio_search.py --strict-only     # strict red line only
python manu/training/status_mix_ratio_search.py --all-calibers    # include legacy-caliber trials
```

CLI of `manu/training/status_mix_ratio_search.py:180-190`: `--out-root --study-name --top
--tail-log --out-json --strict-only --all-calibers`. The report groups trials by caliber
(cosine period × measured epoch) and duels each group against its own RC1 reference.

With `--all-calibers`, the legacy scheduling group has **no** reference system, so it must be
listed only and excluded from the duel — duelling it has crashed before.

### 8. Stop, resume, and harvest

```bash
# 【服务器】graceful: stop dispatching, let running trials finish (up to 30 min)
bash manu/training/stop_mix_ratio_search.sh
# 【服务器】immediate: kill the whole process group; trials become FAIL and self-heal on restart
bash manu/training/stop_mix_ratio_search.sh --force
# 【服务器】resume from the SQLite study (completed trials are skipped)
bash manu/training/launch_mix_ratio_search.sh
```

Do not use `pkill -f <pattern>`: it matches the command line of the shell running it and kills
your own SSH session. Kill by PID from `<out_root>/dispatcher.pid`, in a separate command from
the one that launched it. Do not use `rm -rf` to clean artefacts; delete the specific files. A
remote `cmd && cmd &` backgrounds the whole chain, so put long-running sequences in a `.sh`
file instead of an inline command line.

On restart the dispatcher heals stale `RUNNING` trials left by a kill or power cut
(`optuna_mix_ratio_search.py:807-828`): it calls `study.tell(t.number, state=FAIL)`. Two traps
make this non-obvious:

- The RDB `trial_id` starts at 1 while `Trial.number` starts at 0 — they differ by 1.
- `study.get_trials(deepcopy=False)` returns **FrozenTrial** objects, which `study.tell()`
  rejects; pass the number, not the object.

Harvest layout under `<out_root>` (`runs/optuna_mix_ratio`):

| Path | Content |
|:--|:--|
| `logs/trial_XXXX.log` | the trial's full stdout/stderr |
| `trials/trial_XXXX/weights/epochK.pt` | every epoch's weights are kept (21.3 MB each) |
| `trials/trial_XXXX/gate/epochK/` | gate output incl. `gate_report.json` |
| `trials/trial_XXXX/logs/` | per-trial training csv/logs |
| `trials/trial_XXXX/trial_result.json` | metrics, `utility`, `train_epochs`, `gate_epoch`, `sched_epochs` |
| `optuna_summary.csv` | one row per trial (see step 9) |
| `status.json` | heartbeat: elapsed hours, done count, running trials, free GPUs |
| `workers.json` | per-worker pid / gpu / trial map |
| `dispatcher.pid` | dispatcher PID |

### 9. Read the results correctly

The summary CSV columns are defined at `optuna_mix_ratio_search.py:715-720`:
`trial,status,mix_spec,mix_mode,target_ratio,measured_ratio,gate_verdict,t1_f1,t1_fp,t2_f1,t2_fp,
t3_f1,t3_fp,utility,utility_vs_rc1,lr_scale,warmup_epochs,weight_decay,focal_beta,offset_weight,
mosaic,scale,translate,clip_max_norm,freeze_bn,duration_min`.

Recompute the utility score from the raw metrics — a mismatch between the stored `utility` column
and the recomputed value has already been found, and PRUNED rows store the literal string
`"None"`. The score is
`U = ΔF1_T3 + 0.5·ΔF1_T2 − 2.5·max(0,−ΔF1_T1) − 0.0005·max(0,ΔFP_T2)` against the frozen
baseline; RC1/Ep3 = **+0.027579** is the pass line.

Trial outcomes are three-valued: `SUCCESS` (cleared the red line), `PRUNED` (touching the red
line — pruned so TPE does not explore it), `CRASHED` (no `trial_result.json`, or an
infrastructure failure such as a Track1 CUDA OOM). A gate-infrastructure failure is deliberately
recorded as FAIL, not as a prune, because pruning would pollute the sampler's "already explored"
statistics.

### 10. Extend the search space (only when the user asks)

An Optuna study cannot change its distribution after creation — the attempt fails with
`CategoricalDistribution does not support dynamic value space.` For an orthogonal experiment
(e.g. `lr_scale` × `mix_spec`), create a **new study** and inject the top-K configurations from
the old one with `enqueue_trial`.

### 11. Validate without a GPU when you only need to check the mechanism

On a machine with no GPU you can still prove that the enqueue path works, using the chain
`enqueue_trial → ask() → trial.user_attrs`: enqueue with `user_attrs{seed, workers, gate_epoch}`,
then `study.ask()` and read the attributes back. This has been done and confirmed that seed,
workers and gate_epoch all land correctly. It validates the plumbing, not the numbers.

## Output contract

Report in this shape:

```
Search state     : dispatcher PID <pid> (PPID=1, independent session) | done <n>/<budget> | elapsed <h> h
Cards            : <gpus> (<k>/<m> busy)
Trial outcomes   : SUCCESS <a> / PRUNED <b> / CRASHED <c>
Protocol         : train_epochs=<n>, gate_epoch=<k>, sched_epochs=5 (cosine period)
Caliber groups   : RC1-Ep3口径 n=<x> | RC1-Ep5口径 n=<y> | 旧调度口径 n=<z> (listed only, not dueled)
Best candidate   : trial_XXXX mix=<spec> U=<+/-0.000000> (recomputed) vs RC1/Ep3 +0.027579
Artifacts        : <paths to optuna_summary.csv / status.json / dispatcher.pid>
Commands         : <full copy-pasteable commands> (execution machine: server)
```

Every reported number needs the script path, the full command, the execution machine and the
artifact path (project rule ⑨).

## Failure handling

- **Dispatcher will not start** → read `<out_root>/logs/dispatcher.log`; `launch_mix_ratio_search.sh`
  prints the last 30 lines on failure. A stale PID file produces `[WARN]` and is overwritten; a
  live PID produces `[ABORT]` — stop the old one first instead of forcing a second dispatcher.
- **Dispatcher dies immediately** → the classic cause is `args.max-hours` being written as
  `args.max - hours` inside an f-string; it must be `args.max_hours`. The symptom is a silently
  empty 20-hour search.
- **Trial stuck in `RUNNING` after a crash** → the self-heal handles it on the next dispatcher
  start; confirm the `[HEAL] 自愈 N 个陈旧 RUNNING trial` line appears in the dispatcher log.
- **Trial crashed** → open `trials/trial_XXXX/trial_result.json`; if it is missing, the worker died
  before writing it, and the real cause is in `logs/trial_XXXX.log` plus the gate sub-logs
  (`gate/epochK/track1_infer.log`).
- **Gate reports an absolute-threshold mistake** → the dispatcher passes `--max-track2-fp 36`
  and `--min-track1-f1 0.900736` (constants `GATE_TRACK2_FP_ABS_MAX` / `GATE_TRACK1_F1_ABS_MIN`
  at `optuna_mix_ratio_search.py:138,142`) because the gate script takes **absolute** thresholds.
  Passing the increment 10 there makes every trial look like a veto. If you drive the gate script
  by hand, use 36 and 0.900736.
- **`handoff_to_5epoch.sh` (graceful 3→5 epoch switch) is currently unusable** → its SIGTERM path
  waits for the old dispatcher but never harvests the in-flight trials, so restarting loses those
  trials' CSV rows and Optuna states. The defect is registered and that path has deliberately not
  been used. Prefer the `runtime_config.json` route (step 5) plus a manual restart, or report the
  limitation instead of using the script.
- **Same study, new parameter** → cannot be done; create a new study (step 10).
- **Cache pressure** → pkl caches must stay around 100 MB (project rule ②); the Longquan val
  heat-map cache rebuild is the heavy one (8 pkl / 28,997 frames / 3 scales, ≈ 977.7 s).
- **Enqueued trial shows empty `trial.params`** → the parameters were pushed as literals instead
  of being read through `suggest_*`; use `enqueue_trial` (which the dispatcher reads with
  `study.ask()` + `suggest_params()`), and pass `skip_if_exists=False` — Optuna 4.x skips
  duplicate parameter combinations when it is `True`, which silently drops the control you wanted.
- **Anything not covered here** → report to the user with the exact failing command and its
  output. Do not start an unauthorised experiment to "find out".

## Examples

### Example 1 — launch, then change the protocol mid-flight

Input: "Start the 4-GPU ratio search for 20 hours; afterwards I want the later trials on 5 epochs."

```bash
# 【服务器】cd /tmp/pycharm_project_10ae9e2e
bash manu/training/launch_mix_ratio_search.sh
# later, without restarting the dispatcher:
cat > runs/optuna_mix_ratio/runtime_config.json <<'EOF'
{"train_epochs": 5}
EOF
```

Output: dispatcher PID in `runs/optuna_mix_ratio/dispatcher.pid` with PPID 1; trials launched
before the config file appears keep `train_epochs=3`, later ones report `train_epochs=5` with
`sched_epochs=5` in `trial_result.json`, and `status.json` shows the elapsed/done heartbeat.

### Example 2 — inject a bit-level reproduction control and read it back

Input: "Re-run RC1's exact configuration with seed 0 / 16 workers / gate at epoch 3, to settle
whether the pipeline reproduces RC1."

```bash
# 【服务器】
python manu/training/enqueue_control_trials.py --specs natural --exact --dry-run
python manu/training/enqueue_control_trials.py --specs natural --exact
python manu/training/status_mix_ratio_search.py --top 20
```

Output: the WAITING queue shows one trial with
`user_attrs={'role': 'RC1_REPRO_CONTROL', 'seed': 0, 'workers': 16, 'gate_epoch': 3}`; the
dispatcher picks it up without a restart; after it settles,
`python manu/training/check_control_repro.py --all` returns the verdict
`EXACT_REPRO` (within `EXACT_TOL`) / `BAND_REPRO` (inside RC1's own epoch envelope) /
`NOT_REPRO` (outside it).
