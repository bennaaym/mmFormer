# Fold-2 BraTS 2024 reproduction

This entry point reads the NIfTI volumes and fold JSON directly from the adjacent
SwinUNETR repository. It creates no dataset copy.

Data/evaluation protocol:

- fixed validation Fold 2; training folds 0, 1, 3, and 4;
- channels `[T1n, T1c, T2w, T2f]`;
- SwinUNETR's train and validation transform order, 96-cubed training crops, and
  channel-wise nonzero normalization;
- BraTS 2024 regions `TC=(1|3)`, `WT=(1|2|3)`, and `ET=3`; resection cavity label
  4 is excluded from all three regions;
- MONAI `DiceMetric(MEAN_BATCH)` on hard TC/WT/ET predictions;
- final conditions: all, no T1n, no T1c, no T2w, and no T2f.

Method-specific mmFormer behavior is retained: four modality-specific encoders,
intra/inter-modal Transformers, fusion and separate decoders, PRM outputs, all
official weighted-cross-entropy plus Dice auxiliary losses, Adam with AMSGrad
(`lr=2e-4`, `weight_decay=1e-4`), polynomial decay, and 300 epochs. Training
masks are independent Bernoulli(0.5) availability flags conditioned on a nonempty
subset. Checkpoint selection is Fold-2 all-modality mean Dice every 10 epochs.

The only architecture compatibility change parameterizes the bottleneck token
grid: official 128-cubed inputs use 8 cubed tokens, while 96-cubed inputs use 6
cubed tokens. Transformer dimensions, depth, attention, encoders, and decoders
are unchanged. Legacy internal encoder variable names still reflect the official
FLAIR/T1ce/T1/T2 order, but the tensors supplied to those four symmetric branches
are deliberately the TRDP order above.

## Compatibility findings

- The original NIfTI loader ignores its split files and makes a seeded three-way
  case split. This entry point instead consumes the fixed five-fold patient JSON.
- The original loader expects already converted NumPy volumes in
  `[FLAIR,T1ce,T1,T2]`; this entry point reads the source NIfTI files directly in
  `[T1n,T1c,T2w,T2f]`, without creating a second dataset.
- The original preprocessing uses a 128-cubed crop, arbitrary-angle rotations,
  and its preprocessed intensities. The reproduction uses the SwinUNETR 96-cubed
  foreground/random crop, flips, nonzero channel-wise normalization, scale, and
  shift pipeline. Validation has normalization only; neither reference pipeline
  applies orientation or spacing transforms.
- The original code converts label 4 to enhancing tumor and reports legacy
  categorical BraTS 2018 regions. Here label 3 is ET and label 4 is resection
  cavity excluded from TC/WT/ET. A four-class one-hot view (background/1/2/3) is
  used only to retain mmFormer's official softmax training losses.
- The original evaluation tests all 15 subsets and uses a custom categorical Dice
  function (including an ET postprocessing variant). The reproduction uses only
  the five requested masks and SwinUNETR's MONAI per-region Dice reduction.
- The original training loop saves periodic/last checkpoints without validation
  selection. The reproduction validates all-modal Fold 2 every 10 epochs and
  selects the highest `(ET+TC+WT)/3`; the five-condition evaluation is run only
  after loading that selected checkpoint.
- The original NIfTI loader samples uniformly from a literal list of 15 masks;
  the older array loader draws four Bernoulli(0.5) flags and rejects all-zero.
  These induce the same uniform distribution over the 15 nonempty subsets. The
  reproduction uses the latter direct Bernoulli formulation and logs counts.

Run sanity checks (no training):

```bash
/net/cremi/abennabi/espaces/travail/trdp/SwinUNETR/.venv310/bin/python \
  train_brats2024.py \
  --config configs/mmformer_brats2024_fold2.yaml \
  --sanity-only --workers 0
```

Start full training only after those checks pass:

```bash
/net/cremi/abennabi/espaces/travail/trdp/SwinUNETR/.venv310/bin/python \
  train_brats2024.py --config configs/mmformer_brats2024_fold2.yaml
```

All experiment settings live in the YAML file. Explicit command-line options
remain available as one-run overrides. The fully resolved configuration, including
overrides, is written to `<save_dir>/resolved_config.yaml` for every invocation.
Resume only from trusted checkpoints produced by this training script; restoring
the optimizer requires full checkpoint deserialization rather than PyTorch's
weights-only loading mode.
