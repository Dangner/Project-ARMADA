# ARMADA — generated experiment report

> **This file is auto-generated** by `python -m armada.run --config
> configs/main.yaml --stage figures` (Phase 7).  Do not hand-edit result
> numbers below; re-run the pipeline instead.  The section below is filled in
> progressively and currently reflects the **Phase 1** deliverables.

## Phase 1 — data pipeline + classical baselines

- Implemented: repo structure, EMBER loader (memmap, stratified caps,
  source-only scalers, config-hash cache), temporal drift split, metric suite,
  LR/RF/GBDT baselines, 46 unit/pipeline tests.
- Exact commands to reproduce:

  ```bash
  pip install -r requirements.txt
  python -m armada.data.vectorize --data_dir data/ember2018 --with-test
  python -m armada.run --config configs/main.yaml --stage data
  python -m armada.run --config configs/main.yaml --stage eval
  python -m pytest tests/ -q
  ```

- Result tables `results/metrics_all.csv` / `results/metrics_by_window.csv`
  are produced by the commands above **on real EMBER 2018 data**.  At the
  time of this phase no EMBER corpus was present in the development sandbox
  (dataset downloads are blocked there), so no numbers are recorded in this
  report yet — by project rule, only numbers computed from real EMBER
  predictions may appear here.

# Phase 2 — Threat Profiler + grouped self-attention (source-only)

- Implemented: `ThreatProfiler` (9 per-group `Linear→LayerNorm→GELU`
  projections + learnable `[CLS]` token; shared-projection ablation switch;
  group-wise anomaly profile for explainability), pre-LayerNorm grouped
  self-attention encoder (2–4 layers, 4 heads, exposed per-layer attention
  maps), classifier head (BCE, balanced pos_weight from source labels),
  Stage A training (AdamW, cosine/one-cycle, grad clipping, AMP-when-CUDA,
  early stopping on the **source** validation split, checkpoints + training
  curves), and `--stage eval` rows for the source-only grouped attention
  (`GroupedAttn` in the metrics tables).
- Exact commands to reproduce:

  ```bash
  python -m armada.run --config configs/main.yaml --stage data
  python -m armada.run --config configs/main.yaml --stage train
  python -m armada.run --config configs/main.yaml --stage eval
  python -m pytest tests/ -q
  ```

- Tests: 67 passing (profiler shapes/ablation/profile, encoder attention
  invariants, heads/class-weights, training smoke + early stopping +
  checkpoint round-trip, train→eval pipeline incl. `GroupedAttn` rows).
- Real-data numbers: same status as Phase 1 — to be produced in one batch
  once the EMBER corpus is available; nothing is fabricated here.

# Phase 3 — dual-discriminator DANN + TTT

- Implemented:
  - `models/grl.py`: gradient reversal (identity forward, grad × −λ backward),
    schedule `λ(p) = 2/(1+exp(−10p)) − 1`;
  - `models/discriminators.py`: D1 marginal discriminator on `z`; D2
    class-conditional CDAN discriminator on the outer product `softmax(p) ⊗ z`
    (optional fixed randomised multilinear map; optional CDAN+E entropy
    conditioning, config-driven);
  - `models/ttt.py`: masked **group-feature** reconstruction (learned `[MASK]`
    token replaces masked group tokens; per-group decoders reconstruct the
    processed features from the encoder rows; MSE on masked cells only) and
    `TTTAdapter` (per-window reset unless `ttt.online`; classifier frozen
    during TTT; non-finite loss ⇒ reset to source weights);
  - `train/adapt.py`: Stage B joint objective
    `cls + marg(D1) + cond(D2) + recon` on labeled-source + unlabeled-target
    batches (target labels never used), AdamW + cosine + grad clip + AMP-on-CUDA,
    per-component training curves.
- Variants trained per seed and evaluated through the saved-prediction
  pipeline (both TTT modes reported, per spec §3.5):
  `DANN` (naive single-discriminator baseline), `DualDANN` (dual-disc + recon,
  no TTT), `DualDANN+TTT` (reset per window), `DualDANN+TTT-online`.
- Assumption stated: Stage B uses the **unlabeled rows of the target windows**
  (standard transductive DANN protocol; their labels are never read). Evaluated
  window labels remain held out for metrics only.
- Exact commands to reproduce:

  ```bash
  python -m armada.run --config configs/main.yaml --stage data
  python -m armada.run --config configs/main.yaml --stage train   # Stage A + Stage B variants
  python -m armada.run --config configs/main.yaml --stage eval    # all method rows
  python -m pytest tests/ -q
  ```

- Tests: GRL gradient sign flip & λ-schedule, CDAN feature/entropy properties,
  TTT reset / online / classifier-frozen / loss-decrease, Stage B smoke
  (variants, curves, checkpoint reload), full train→eval pipeline rows.
- Real-data numbers: same status as Phase 1-2 (EMBER corpus unavailable in
  the development sandbox); nothing is fabricated.

# Phase 4 — adversarial training + robustness shield

- Implemented:
  - `attacks/constraints.py`: `FeatureSpaceProjector` — attacks act on the
    model's processed inputs and are projected back onto the EMBER-feasible
    region after every step: count-like features non-negative
    (`robustness.clip_min`), per-column upper bound = source-period max
    (source-only, leakage rule), missingness frozen (the adversary cannot
    flip feature presence).  Upper bounds extend to `max(source_max, clean)`
    so clean rows are always feasible.
  - `attacks/fgsm.py` / `attacks/pgd.py`: one-step signed gradient and
    iterative PGD (configurable eps/step size/steps, seeded random start),
    both l∞-ball bounded and projected.
  - Adversarial training in Stage B (toggle `train.use_adversarial_training`,
    `train.adv_attack: fgsm|pgd`, weight `train.adv_weight`): classification
    loss on FGSM/PGD examples of the labeled source batch, curves gain a
    `train_adv` component.  Off by default (component toggles per spec §5).
  - `eval/robustness.py` + `--stage robust`: clean vs FGSM/PGD at every
    `robustness.eval_epsilons`, plus a random-noise baseline (MalGAN-style
    generator attack not implemented — it was an optional item).  Rows for
    `GroupedAttn`, `DANN`, `DualDANN`, `DualDANN+TTT` (TTT runs before
    attacking) → `results/robustness.csv`, all metrics recomputed from saved
    predictions.
- Limitation (also in README §Limitations): feature-space perturbations are
  **not** guaranteed to correspond to valid PE executables; robustness
  numbers measure the model in feature space only.
- Exact commands to reproduce:

  ```bash
  python -m armada.run --config configs/main.yaml --stage data
  python -m armada.run --config configs/main.yaml --stage train
  python -m armada.run --config configs/main.yaml --stage robust
  python -m pytest tests/ -q
  ```

- Tests (110 total): l∞ ball containment, constraint projection (non-negative
  counts, source-max clamp, frozen missingness, zero-eps identity, seeded PGD
  determinism, 1-step PGD ≡ FGSM), Stage B adversarial-loss component on/off,
  robustness CSV end-to-end.
- Real-data numbers: still pending the EMBER corpus; nothing fabricated.

# Phase 5 — immune memory + calibrated decision engine

- Implemented:
  - `models/memory.py`: `MemoryBank` — fixed-size numpy cosine nearest-neighbour
    store of embeddings (malware + optional benign), age/LRU eviction with
    capacity enforcement, high-confidence window-update rule
    (`memory.add_confidence`); `MemoryCalibrator` — **learned** logistic
    regression on `[P(malware), max_sim_to_malware_memory]`, fit on the
    **source validation split only** (never hand-set weights).
  - `models/decision.py`: `DecisionEngine` — SAFE/SUSPICIOUS/MALWARE from two
    data-derived thresholds: MALWARE at FPR ≤ `decision.malware_fpr_target`
    (0.1 %), SUSPICIOUS at the lower threshold (FPR ≤ 5 %), both selected on
    source validation scores only (no hardcoded 30/60).  Reports the 2-class
    confusion (MALWARE-as-positive and alert-as-positive), verdict counts and
    the SUSPICIOUS routing ("sandbox") rate.
  - `eval/neural.py:evaluate_armada_seed` — full deployment order per seed:
    initial memory from source-train confirmed labels → calibrator + thresholds
    on source-val → chronological windows (TTT reset+adapt on unlabeled rows,
    calibrated scores, verdicts, then memory updated with high-confidence
    detections; age clock advances per window).
  - Method rows: **`ARMADA`** (dual-disc + TTT + memory + calibrator + decision)
    and **`ARMADA-noMem`** (calibrator on the classifier probability only) —
    this pair *is* the required "evaluate with and without memory".
    `results/decision_report.csv` carries verdict counts, sandbox rates,
    confusion counts and the chosen thresholds per window.
- Assumptions stated: (1) memory stores source-model embedding signatures —
  TTT shifts the encoder slightly per window and similarities are computed in
  the adapted space (documented approximation); (2) calibrator and thresholds
  share the source validation split (both are source-only; noted in REPORT).
- Exact commands to reproduce:

  ```bash
  python -m armada.run --config configs/main.yaml --stage data
  python -m armada.run --config configs/main.yaml --stage train
  python -m armada.run --config configs/main.yaml --stage eval    # includes ARMADA / ARMADA-noMem
  python -m pytest tests/ -q
  ```

- Tests (129 total): memory cosine/capacity/age/LRU/update-rule, calibrator
  learned weights + no-memory ablation, threshold FPR budgets on source-val,
  data-derived (non-hardcoded) thresholds, verdict partition, confusion and
  sandbox-rate accounting, pipeline end-to-end with `decision_report.csv`.
- Real-data numbers: still pending the EMBER corpus; nothing fabricated.
