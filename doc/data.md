# Data Preparation and Format

Motion Mamba 2.0 uses retargeted BONES Seed, HumanML3D and HiPHI motion in a shared G1 representation. Obtain the processed motion data from the [dataset repository](https://huggingface.co/datasets/AIGeeksGroup/MotionMamba2.0) and model resources from the [model repository](https://huggingface.co/AIGeeksGroup/MotionMamba2.0). Access and use remain subject to the original dataset licenses.

Training consumes processed motion arrays, a JSONL index and cached text embeddings. Raw-data conversion and retargeting are performed separately from this repository.

## Training Index

Store one JSON object per line in a UTF-8 `.jsonl` file. Each record describes a continuous motion segment:

```json
{"dataset":"bones","source_family_id":"family_001","sample_id":"segment_001","project_split":"train","feature_path":"features/segment_001.npy","frames":300,"captions":["A person walks forward."]}
```


| Field                | Requirement                                                                      |
| ---------------------- | ---------------------------------------------------------------------------------- |
| `dataset`            | One of `bones`, `humanml3d` or `hiphi`                                           |
| `source_family_id`   | Original source-family identity shared by related segments                       |
| `sample_id`          | Segment identifier                                                               |
| `project_split`      | Training accepts `train`; keep held-out rows in the index for identity exclusion |
| `feature_path`       | Path to the processed `.npy` motion array                                        |
| `frames`             | Number of frames in the segment; must match the array                            |
| `captions`           | Nonempty text descriptions applying to the whole segment                         |
| `source_motion_id`   | Optional original motion identity, used for split isolation                      |
| `caption_time_spans` | Optional time-local annotations, described below                                 |
| `native_start_sec`   | Optional segment time origin for time-local annotations; defaults to `0`         |

A single whole-segment `caption` string is also accepted instead of `captions`. When `caption_time_spans` is present and nonempty, it takes precedence over the whole-segment captions.

Set the dataset locations in `configs/training_motion_mamba2.yaml`:

```yaml
data:
  index_path: /path/to/data/train_index.jsonl
  feature_root: /path/to/data
  text_features_paths:
    - /path/to/data/text_features
  excluded_families: []
```

These are fields within the existing training configuration, not a replacement for its other settings. Relative `feature_path` entries resolve against `feature_root`, or against the JSONL directory when `feature_root` is empty. Prefer absolute paths for the configuration values.

## Motion Arrays

Each motion file is a finite `float32` NumPy array of shape `[T, 414]`, representing a 34-joint G1 skeleton at 25 FPS. Use the physical motion features expected by the supplied G1 frontend; arbitrary joint coordinates or another 414-channel layout are not interchangeable with this representation. Training applies the configured normalization and frozen codec internally.

The loader samples continuous crops from individual segments. It does not stitch unrelated clips. A crop consists of history `H`, a supervised generation window `G=52`, and an optional future interval `F`:


| History H | Generation G | Future F | Total crop | Updates per 20-update cycle |
| ----------- | -------------- | ---------- | ------------ | ----------------------------- |
| 0         | 52           | 0        | 52         | 2                           |
| 52        | 52           | 0        | 104        | 5                           |
| 196       | 52           | 0        | 248        | 4                           |
| 0         | 52           | 196      | 248        | 2                           |
| 52        | 52           | 144      | 248        | 4                           |
| 144       | 52           | 52       | 248        | 3                           |

The history/future schedule is shuffled within each cycle. Each source dataset must have eligible training families for every crop length. A segment may contribute to the shorter buckets even if it is too short for a 248-frame crop. H=0 samples have no history prefix.

Source sampling uses shuffled cycles of 14 BONES, 3 HumanML3D and 3 HiPHI slots, corresponding to nominal proportions of 70%, 15% and 15%. The history and supervised continuation are real GT. Future frames supply selected visible control values, rather than a fully revealed future motion sequence. Text and spatial conditions follow the dropout settings in the training configuration.

## Time-Local Captions

Use time spans when a description applies to only part of a segment:

```json
{
  "dataset": "hiphi",
  "source_family_id": "family_002",
  "sample_id": "segment_002",
  "project_split": "train",
  "feature_path": "features/segment_002.npy",
  "frames": 500,
  "native_start_sec": 10.0,
  "caption_time_spans": [
    {
      "text": "A person waves one hand repeatedly.",
      "native_start_sec": 10.0,
      "native_end_sec": 30.0
    }
  ]
}
```

The example is shown across lines for readability; serialize each record onto one line in the JSONL index. Span timestamps use the original motion timeline. `native_start_sec` on the record identifies the segment's origin. The complete H/G/F crop must fit inside the caption span; a label covering only the generated window is insufficient for this loader.

## Cached Text Features

Every eligible caption needs a frozen text embedding. Store one `.npy` file per exact caption, named using the SHA-256 digest of its UTF-8 text:

```python
import hashlib
from pathlib import Path

import numpy as np

caption = "A person walks forward."
filename = hashlib.sha256(caption.encode("utf-8")).hexdigest() + ".npy"
embedding = np.load(Path("/path/to/text_features") / filename)
assert embedding.dtype == np.float32
assert embedding.size == 4096
assert np.isfinite(embedding).all()
```

Use a `[4096]` array for each embedding. The loader searches the directories in `data.text_features_paths` in order. Preserve caption spelling, capitalization, spacing and punctuation because they determine the cache filename.

Use the same frozen text encoder, adapters and preprocessing as inference. Compatible caches can be supplied separately or generated through `motion_mamba2/model/text_encoder.py`. Cached language features are distinct from motion supervision: the training targets remain real GT actions.

## Benchmark Inputs

The repository includes twelve fixed input files in `benchmarks/benchmark_assets/`, with four files per domain:


| Filename pattern             | Contents                                            |
| ------------------------------ | ----------------------------------------------------- |
| `<domain>_cases.jsonl`       | Case identities, splits, seeds and target metadata  |
| `<domain>_inputs.npz`        | Initial motion history and geometric control inputs |
| `<domain>_text_features.npy` | Cached prompt embeddings                            |
| `<domain>_text_index.json`   | Prompt ordering for the embedding array             |

The domain prefixes are `bones`, `humanml3d` and `hiphi`. This indexed benchmark cache differs from the per-caption cache used for training. Keep the supplied files unchanged: evaluation verifies their fingerprints against `configs/benchmark_assets.json`.

No training index or text encoder download is needed to run the packaged benchmarks. Leave `MM2_BENCHMARK_DATA_PATH` empty to use the included assets, or set it to a directory containing identical copies. Run the input check from the repository root:

```bash
python -m scripts.evaluate --check
```
