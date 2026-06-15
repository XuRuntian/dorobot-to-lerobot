# dorobot-to-lerobot

Merge dorobot single-episode recordings into a clean LeRobot-style dataset.

The tool is designed for batch cleanup:

- skip incomplete or unreadable episodes instead of failing the whole merge
- optionally validate exported subtask timeline annotations
- renumber kept episodes continuously
- update LeRobot metadata after bad episodes are skipped
- write subtask annotations as sidecar files under `annotations/`

## Install

```bash
uv sync
```

Python 3.10+ is recommended. This project uses `pyproject.toml` for dependency management.

## Basic Usage

```bash
uv run dorobot-to-lerobot \
  --input /path/to/raw_dataset_root \
  --output /path/to/merged_dataset
```

`--input` is the parent directory containing task folders. By default, every
task folder under `--input` is processed.

`--output` is the output root. Each task is written to a subdirectory named
from `meta/common_record.json` using `task_name` and `task_id`.

## Process Selected Tasks

```bash
uv run dorobot-to-lerobot \
  --input /path/to/raw_dataset_root \
  --output /path/to/merged_dataset \
  --folders "clean_the_desktop" "pick_the_banana"
```

`--folders` accepts task folder names relative to `--input`.

## Merge With Subtask Annotations

```bash
uv run dorobot-to-lerobot \
  --input /path/to/raw_dataset_root \
  --output /path/to/merged_dataset \
  --folders "clean_the_desktop" \
  --annotation /path/to/exported_labels.json \
  --default-task-id clean_the_desktop
```

When `--annotation` is provided, episodes without a valid matching annotation
are skipped by default. Invalid labels include missing `videoLabels`, non
continuous ranges, overlapping ranges, invalid frame indices, and annotations
whose final frame exceeds the parquet frame count.

Use `--no-require-annotation` if annotations should be written when available
but should not be required for keeping an episode.

## Dry Run

```bash
uv run dorobot-to-lerobot \
  --input /path/to/raw_dataset_root \
  --output /path/to/merged_dataset \
  --dry-run
```

`--dry-run` validates and reports counts without writing merged data.

## Overwrite Existing Output

```bash
uv run dorobot-to-lerobot \
  --input /path/to/raw_dataset_root \
  --output /path/to/merged_dataset \
  --overwrite
```

Without `--overwrite`, an existing output task directory is treated as an error.

## Output Layout

```text
merged_task/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── episode_000001.parquet
├── videos/
├── depth/
├── meta/
│   ├── common_record.json
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   └── info.json
└── annotations/
    ├── episode_mapping.jsonl
    ├── skipped.jsonl
    ├── subtasks.jsonl
    └── summary.json
```

The merge rewrites:

- episode filenames under `data/`, `videos/`, and `depth/`
- parquet columns `episode_index`, `index`, and `frame_index` when present
- `meta/episodes.jsonl`
- `meta/episodes_stats.jsonl` with recomputed per-episode numeric `min`, `max`, `mean`, and `std` for columns declared in `info.json` features
- `meta/info.json` fields `total_episodes`, `total_frames`, `total_videos`, and `splits`

Subtask annotations are not embedded into parquet files. They are stored in
`annotations/subtasks.jsonl` and use the final renumbered `episode_index`.

Skipped source episodes and their reasons are recorded in
`annotations/skipped.jsonl`.

## CLI Reference

```text
--input PATH              Raw task parent directory.
--output PATH             Merged dataset output root.
--folders NAME [NAME ...] Optional task folders under --input.
--annotation PATH         Optional exported timeline annotation JSON.
--default-task-id NAME    Fallback task id for annotation items.
--no-require-annotation   Keep data even if annotation is missing or invalid.
--dry-run                 Validate only, do not write merged output.
--overwrite               Replace existing output task directories.
--log-dir PATH            Also write logs to PATH/merge.log.
```

