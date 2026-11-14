# Frozen model weights

## Stage-A detector

`frp_stageA_v1_best.pt` is the `best.pt` file from the supplied `frp_stageA_v1` detector training output.

- File size: 300,992,729 bytes.
- SHA-256: `3cb84395be5b9d981576e2dd44523e57492b14df230eed4f4d7a2f447b9b8655`.
- The PyTorch ZIP container passed its CRC integrity check.

Use this file for the evaluator's `--weights` argument. Do not point `--ddm-ckpt` at the detector weight.

The detector weight exceeds GitHub's ordinary 100 MB file limit and is excluded by `.gitignore`. Keep it out of ordinary Git commits; include it in the Zenodo ZIP or use an explicitly approved large-file release mechanism.

## DDM checkpoints

The two supplied DDM checkpoints are included separately:

| File | SHA-256 | Recorded epoch | Recorded validation AP |
|---|---|---:|---:|
| `ddm_v3_train_20261001_best.pt` | `3251782c6fbb92b0b13851e7e22555c87e1ac1f911c24052f0a664592c43b51a` | 1 | 0.8360939214783871 |
| `ddm_v3_train_20261001_last.pt` | `511a83bdbe09e1926d4b1f22553fb3e27a38fa836adf57ead7535809f5f4b528` | 30 | 0.7613222643112543 |

Both files are 1,660,101 bytes and passed PyTorch ZIP-container CRC checks. Restricted pickle inspection, without executing project code, confirmed the expected `cues`, `branch`, `p3_channels`, `epoch`, and `val_ap` keys, with `p3_channels=512`. Their archive payloads differ; they are not duplicate checkpoints.

Use `ddm_v3_train_20261001_last.pt` for the evaluator's `--ddm-ckpt` argument, matching the original evaluation configuration. The `best.pt` file is retained for checkpoint-trajectory diagnostics and is not silently substituted for `last.pt`.

GPU model loading and evaluation have not been rerun during packaging.
