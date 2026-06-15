from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from dorobot_to_lerobot.annotation_utils import (
    AnnotationRecord,
    load_annotation_groups,
    safe_filename,
)


LOGGER = logging.getLogger("dorobot_merge")
SUPPORTED_DATA_VERSIONS = {"v1.0"}


@dataclass
class EpisodeCandidate:
    source_task: Path
    source_episode_dir: Path
    source_order: int
    parquet_path: Path
    source_episode_index: int
    frame_count: int
    info: dict[str, Any]
    common_record: dict[str, Any]
    annotation: AnnotationRecord | None = None
    video_files: list[Path] = field(default_factory=list)
    depth_files: list[Path] = field(default_factory=list)
    other_media_files: list[Path] = field(default_factory=list)


@dataclass
class SkipRecord:
    source_path: str
    reason: str
    detail: Any = None


@dataclass
class MergeResult:
    task_folder: str
    output_path: str | None
    merged_count: int
    skipped_count: int
    total_frames: int


def setup_logging(log_dir: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "merge.log", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(record)
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sort_key(path: Path) -> tuple[int, str]:
    tail = path.name.rsplit("_", 1)[-1]
    return (int(tail), path.name) if tail.isdigit() else (0, path.name)


def episode_index_from_name(path: Path) -> int:
    stem = path.stem
    if stem.startswith("episode_"):
        suffix = stem.removeprefix("episode_")
        if suffix.isdigit():
            return int(suffix)
    return 0


def output_name_from_common(common_record: dict[str, Any], fallback: str) -> str:
    task_name = str(common_record.get("task_name") or fallback).strip()
    task_id = str(common_record.get("task_id") or "").strip()
    name = f"{task_name}_{task_id}" if task_id else task_name
    return safe_filename(name)


def feature_names(info: dict[str, Any]) -> list[str]:
    features = info.get("features")
    if not isinstance(features, dict):
        return []
    return list(features)


def expected_media_paths(episode_dir: Path, parquet_stem: str, info: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for name in feature_names(info):
        if not name.startswith("observation.images"):
            continue
        if "depth" in name:
            paths.append(episode_dir / "depth" / "chunk-000" / name / f"{parquet_stem}.avi")
        else:
            paths.append(episode_dir / "videos" / "chunk-000" / name / f"{parquet_stem}.mp4")
    return paths


def find_matching_media(root: Path, parquet_stem: str) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob(f"{parquet_stem}.*") if path.is_file())


def pick_episode_record(records: list[dict[str, Any]], source_episode_index: int) -> dict[str, Any] | None:
    if not records:
        return None
    for record in records:
        if record.get("episode_index") == source_episode_index:
            return dict(record)
    return dict(records[0])


def select_annotation_group(
    groups: dict[str, list[AnnotationRecord]],
    *,
    task_folder: Path,
    common_record: dict[str, Any] | None,
    only_one_task: bool,
) -> list[AnnotationRecord] | None:
    if not groups:
        return None
    if only_one_task and len(groups) == 1:
        return next(iter(groups.values()))

    aliases = {safe_filename(task_folder.name)}
    if common_record:
        for key in ("task_id", "task_name", "task"):
            value = common_record.get(key)
            if value is not None:
                aliases.add(safe_filename(str(value)))
        aliases.add(output_name_from_common(common_record, task_folder.name))

    for alias in aliases:
        if alias in groups:
            return groups[alias]
    return None


def validate_episode_dir(
    source_task: Path,
    episode_dir: Path,
    source_order: int,
    annotation: AnnotationRecord | None,
    *,
    require_annotation: bool,
) -> tuple[EpisodeCandidate | None, SkipRecord | None]:
    try:
        meta_dir = episode_dir / "meta"
        data_dir = episode_dir / "data"
        if not meta_dir.is_dir():
            return None, SkipRecord(str(episode_dir), "missing meta directory")
        if not data_dir.is_dir():
            return None, SkipRecord(str(episode_dir), "missing data directory")

        info_path = meta_dir / "info.json"
        common_path = meta_dir / "common_record.json"
        if not info_path.exists():
            return None, SkipRecord(str(episode_dir), "missing meta/info.json")
        if not common_path.exists():
            return None, SkipRecord(str(episode_dir), "missing meta/common_record.json")

        info = read_json(info_path)
        data_version = info.get("dorobot_dataset_version", "unknown")
        if data_version not in SUPPORTED_DATA_VERSIONS:
            return None, SkipRecord(str(episode_dir), "unsupported dorobot_dataset_version", data_version)

        common_record = read_json(common_path)
        parquet_files = sorted(data_dir.rglob("episode_*.parquet"))
        if len(parquet_files) != 1:
            return None, SkipRecord(str(episode_dir), "expected exactly one parquet episode", len(parquet_files))
        parquet_path = parquet_files[0]

        try:
            df = pd.read_parquet(parquet_path)
        except Exception as exc:
            return None, SkipRecord(str(episode_dir), "unreadable parquet", str(exc))
        if df.empty:
            return None, SkipRecord(str(episode_dir), "empty parquet episode")

        expected_media = expected_media_paths(episode_dir, parquet_path.stem, info)
        missing_media = [str(path.relative_to(episode_dir)) for path in expected_media if not path.exists()]
        if missing_media:
            return None, SkipRecord(str(episode_dir), "missing media files", missing_media)

        video_files = find_matching_media(episode_dir / "videos", parquet_path.stem)
        depth_files = find_matching_media(episode_dir / "depth", parquet_path.stem)
        other_media_files = find_matching_media(episode_dir / "images", parquet_path.stem)

        if require_annotation:
            if annotation is None:
                return None, SkipRecord(str(episode_dir), "missing annotation for episode")
            if not annotation.is_valid:
                return None, SkipRecord(str(episode_dir), "invalid annotation", annotation.validation_reasons)
            last_annotation_frame = annotation.frame_count - 1
            if last_annotation_frame >= len(df):
                return None, SkipRecord(
                    str(episode_dir),
                    "annotation end_frame exceeds parquet frame count",
                    {"annotation_end_frame": last_annotation_frame, "frame_count": len(df)},
                )

        return (
            EpisodeCandidate(
                source_task=source_task,
                source_episode_dir=episode_dir,
                source_order=source_order,
                parquet_path=parquet_path,
                source_episode_index=episode_index_from_name(parquet_path),
                frame_count=len(df),
                info=info,
                common_record=common_record,
                annotation=annotation,
                video_files=video_files,
                depth_files=depth_files,
                other_media_files=other_media_files,
            ),
            None,
        )
    except Exception as exc:
        return None, SkipRecord(str(episode_dir), "validation crashed", str(exc))


def rewrite_parquet(input_path: Path, output_path: Path, *, episode_index: int, global_start: int) -> int:
    df = pd.read_parquet(input_path)
    frame_count = len(df)
    if "episode_index" in df.columns:
        df["episode_index"] = episode_index
    if "index" in df.columns:
        df["index"] = range(global_start, global_start + frame_count)
    if "frame_index" in df.columns:
        df["frame_index"] = range(frame_count)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return frame_count


def copy_media_file(input_path: Path, source_episode_dir: Path, output_dir: Path, new_stem: str) -> Path:
    relative = input_path.relative_to(source_episode_dir)
    output_relative = relative.with_name(f"{new_stem}{input_path.suffix}")
    output_path = output_dir / output_relative
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)
    return output_path


def update_info_json(info: dict[str, Any], *, total_episodes: int, total_frames: int, total_videos: int) -> dict[str, Any]:
    updated = dict(info)
    updated.update(
        {
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_videos": total_videos,
            "splits": {"train": f"0:{total_episodes}"},
        }
    )
    return updated


def merge_meta_files(candidates: list[EpisodeCandidate], output_dir: Path, total_frames: int, total_videos: int) -> None:
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    first = candidates[0]
    first_meta = first.source_episode_dir / "meta"

    write_json(
        meta_dir / "info.json",
        update_info_json(first.info, total_episodes=len(candidates), total_frames=total_frames, total_videos=total_videos),
    )

    for json_file in sorted(first_meta.glob("*.json")):
        if json_file.name == "info.json":
            continue
        shutil.copy2(json_file, meta_dir / json_file.name)

    episodes_records: list[dict[str, Any]] = []
    stats_records: list[dict[str, Any]] = []
    other_jsonl_records: dict[str, list[dict[str, Any]]] = {}

    for new_index, candidate in enumerate(candidates):
        source_meta = candidate.source_episode_dir / "meta"
        episode_record = pick_episode_record(read_jsonl(source_meta / "episodes.jsonl"), candidate.source_episode_index) or {}
        episode_record["episode_index"] = new_index
        episode_record["length"] = candidate.frame_count
        episodes_records.append(episode_record)

        stats_record = pick_episode_record(read_jsonl(source_meta / "episodes_stats.jsonl"), candidate.source_episode_index)
        if stats_record is not None:
            stats_record["episode_index"] = new_index
            stats_records.append(stats_record)

        for jsonl_file in sorted(source_meta.glob("*.jsonl")):
            if jsonl_file.name in {"episodes.jsonl", "episodes_stats.jsonl"}:
                continue
            records = read_jsonl(jsonl_file)
            if not records:
                continue
            selected = pick_episode_record(records, candidate.source_episode_index)
            if selected is None:
                selected = records[0]
            selected = dict(selected)
            if "episode_index" in selected:
                selected["episode_index"] = new_index
            other_jsonl_records.setdefault(jsonl_file.name, []).append(selected)

    write_jsonl(meta_dir / "episodes.jsonl", episodes_records)
    if stats_records:
        write_jsonl(meta_dir / "episodes_stats.jsonl", stats_records)

    for filename, records in other_jsonl_records.items():
        if records and all("episode_index" not in record for record in records):
            deduped: list[dict[str, Any]] = []
            seen: set[str] = set()
            for record in records:
                key = json.dumps(record, sort_keys=True, ensure_ascii=False)
                if key not in seen:
                    seen.add(key)
                    deduped.append(record)
            records = deduped
        write_jsonl(meta_dir / filename, records)


def write_annotations(output_dir: Path, candidates: list[EpisodeCandidate], skipped: list[SkipRecord]) -> None:
    annotations_dir = output_dir / "annotations"
    subtask_records: list[dict[str, Any]] = []
    mapping_records: list[dict[str, Any]] = []

    for new_index, candidate in enumerate(candidates):
        mapping_records.append(
            {
                "episode_index": new_index,
                "source_episode_path": str(candidate.source_episode_dir),
                "source_episode_index": candidate.source_episode_index,
                "source_order": candidate.source_order,
            }
        )
        if candidate.annotation is not None:
            subtask_records.append(
                {
                    "episode_index": new_index,
                    "task_id": candidate.annotation.task_id,
                    "frame_count": candidate.frame_count,
                    "source_annotation_index": candidate.annotation.source_index,
                    "source_annotation_id": candidate.annotation.original_id,
                    "subtasks": candidate.annotation.subtasks,
                }
            )

    write_jsonl(annotations_dir / "episode_mapping.jsonl", mapping_records)
    if subtask_records:
        write_jsonl(annotations_dir / "subtasks.jsonl", subtask_records)
    write_jsonl(
        annotations_dir / "skipped.jsonl",
        [{"source_path": item.source_path, "reason": item.reason, "detail": item.detail} for item in skipped],
    )
    write_json(
        annotations_dir / "summary.json",
        {
            "merged_episodes": len(candidates),
            "skipped_episodes": len(skipped),
            "subtask_annotations": len(subtask_records),
        },
    )


def merge_task(
    task_path: Path,
    output_root: Path,
    *,
    annotations: list[AnnotationRecord] | None,
    require_annotation: bool,
    dry_run: bool,
    overwrite: bool,
) -> MergeResult:
    episode_dirs = sorted([path for path in task_path.iterdir() if path.is_dir()], key=sort_key)
    skipped: list[SkipRecord] = []
    candidates: list[EpisodeCandidate] = []

    common_record_for_name: dict[str, Any] | None = None
    for episode_dir in episode_dirs:
        common_path = episode_dir / "meta" / "common_record.json"
        if common_path.exists():
            try:
                common_record_for_name = read_json(common_path)
                break
            except Exception:
                pass

    for source_order, episode_dir in enumerate(episode_dirs):
        annotation = annotations[source_order] if annotations and source_order < len(annotations) else None
        candidate, skip = validate_episode_dir(
            task_path,
            episode_dir,
            source_order,
            annotation,
            require_annotation=require_annotation,
        )
        if skip is not None:
            skipped.append(skip)
            continue
        if candidate is not None:
            candidates.append(candidate)

    output_name = output_name_from_common(common_record_for_name or {}, task_path.name)
    output_dir = output_root / output_name

    if annotations and len(annotations) > len(episode_dirs):
        for annotation in annotations[len(episode_dirs):]:
            skipped.append(
                SkipRecord(
                    str(task_path),
                    "annotation has no matching source episode",
                    {"source_annotation_index": annotation.source_index, "original_id": annotation.original_id},
                )
            )

    LOGGER.info("task=%s valid=%s skipped=%s output=%s", task_path.name, len(candidates), len(skipped), output_dir)

    if dry_run:
        return MergeResult(task_path.name, str(output_dir), len(candidates), len(skipped), sum(c.frame_count for c in candidates))
    if not candidates:
        write_annotations(output_dir, [], skipped)
        return MergeResult(task_path.name, str(output_dir), 0, len(skipped), 0)

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    copied_media_count = 0
    for new_index, candidate in enumerate(candidates):
        new_stem = f"episode_{new_index:06d}"
        rewrite_parquet(
            candidate.parquet_path,
            output_dir / "data" / "chunk-000" / f"{new_stem}.parquet",
            episode_index=new_index,
            global_start=total_frames,
        )
        total_frames += candidate.frame_count

        media_files = candidate.video_files + candidate.depth_files + candidate.other_media_files
        for media_file in media_files:
            copy_media_file(media_file, candidate.source_episode_dir, output_dir, new_stem)
            copied_media_count += 1

    merge_meta_files(candidates, output_dir, total_frames, copied_media_count)
    write_annotations(output_dir, candidates, skipped)
    return MergeResult(task_path.name, str(output_dir), len(candidates), len(skipped), total_frames)


def discover_task_folders(input_root: Path, folders: list[str] | None) -> list[Path]:
    if folders:
        task_paths = [input_root / folder for folder in folders]
    else:
        task_paths = sorted([path for path in input_root.iterdir() if path.is_dir()], key=sort_key)
    missing = [str(path) for path in task_paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"task folders not found: {missing}")
    return task_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge dorobot single-episode folders into a valid LeRobot-style dataset."
    )
    parser.add_argument("--input", required=True, type=Path, help="raw task parent directory")
    parser.add_argument("--output", required=True, type=Path, help="merged dataset output root")
    parser.add_argument("--folders", nargs="*", help="task folder names under --input; defaults to all folders")
    parser.add_argument("--annotation", type=Path, help="raw exported timeline annotation JSON")
    parser.add_argument(
        "--default-task-id",
        default="default_task",
        help="task id used when an annotation item does not contain one",
    )
    parser.add_argument(
        "--no-require-annotation",
        action="store_true",
        help="do not skip episodes that have no valid annotation when --annotation is provided",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate and report without writing merged data")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output task directory")
    parser.add_argument("--log-dir", type=Path, help="optional directory for merge.log")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_dir)

    if not args.input.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    task_paths = discover_task_folders(args.input, args.folders)
    annotation_groups: dict[str, list[AnnotationRecord]] = {}
    if args.annotation:
        annotation_groups = load_annotation_groups(args.annotation, default_task_id=args.default_task_id)
        LOGGER.info("loaded annotation groups: %s", {task_id: len(records) for task_id, records in annotation_groups.items()})

    results: list[MergeResult] = []
    for task_path in task_paths:
        first_common: dict[str, Any] | None = None
        for episode_dir in sorted([path for path in task_path.iterdir() if path.is_dir()], key=sort_key):
            common_path = episode_dir / "meta" / "common_record.json"
            if common_path.exists():
                try:
                    first_common = read_json(common_path)
                    break
                except Exception:
                    pass

        annotations = select_annotation_group(
            annotation_groups,
            task_folder=task_path,
            common_record=first_common,
            only_one_task=len(task_paths) == 1,
        )
        if args.annotation and annotations is None:
            LOGGER.warning("no annotation group matched task folder %s", task_path.name)

        result = merge_task(
            task_path,
            args.output,
            annotations=annotations,
            require_annotation=bool(args.annotation and not args.no_require_annotation),
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        results.append(result)

    total_merged = sum(result.merged_count for result in results)
    total_skipped = sum(result.skipped_count for result in results)
    total_frames = sum(result.total_frames for result in results)
    LOGGER.info(
        "done: tasks=%s merged_episodes=%s skipped=%s total_frames=%s",
        len(results),
        total_merged,
        total_skipped,
        total_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
