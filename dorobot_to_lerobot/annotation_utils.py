from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TASK_ID = "default_task"
ALLOW_FIRST_START_ONE = True


VALIDATION_REASON_MESSAGES = {
    "missing_video_labels": "videoLabels is missing or empty",
    "invalid_video_label": "videoLabels entry is not an object",
    "invalid_range_count": "ranges must contain exactly one range",
    "invalid_range_object": "ranges[0] must be an object",
    "invalid_label_count": "timelinelabels must contain exactly one label",
    "invalid_label_text": "timelinelabels[0] must be a non-empty string",
    "invalid_frame_type": "range start/end must be an integer",
    "negative_start": "range start must be >= 0",
    "end_before_start": "range end must be >= start",
    "invalid_first_start": "first subtask must start at frame 0 or 1",
    "overlap": "subtask overlaps the previous range",
    "not_continuous": "subtask does not start at previous end + 1",
}


@dataclass(frozen=True)
class AnnotationRecord:
    task_id: str
    source_index: int
    original_id: Any
    subtasks: list[dict[str, Any]]
    validation_reasons: list[dict[str, Any]]

    @property
    def is_valid(self) -> bool:
        return not self.validation_reasons and bool(self.subtasks)

    @property
    def frame_count(self) -> int:
        return max((subtask["end_frame"] for subtask in self.subtasks), default=-1) + 1


def safe_filename(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    name = name.strip("._")
    return name or "task"


def reason(
    code: str,
    *,
    message: str | None = None,
    subtask_index: int | None = None,
    value: Any = None,
    expected: Any = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "code": code,
        "message": message or VALIDATION_REASON_MESSAGES.get(code, code),
    }
    if subtask_index is not None:
        payload["subtask_index"] = subtask_index
    if value is not None:
        payload["value"] = value
    if expected is not None:
        payload["expected"] = expected
    return payload


def load_raw_annotations(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("annotation JSON must be a list")
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"annotation item {index} must be an object")
    return data


def coerce_frame(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("frame must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError("frame must be an integer")


def infer_task_id(item: dict[str, Any], default_task_id: str = DEFAULT_TASK_ID) -> str:
    for key in ("task_id", "taskId", "task", "dataset", "project"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return safe_filename(value)

    images = (item.get("observation") or {}).get("images") or {}
    if isinstance(images, dict):
        for url in images.values():
            if not isinstance(url, str):
                continue
            match = re.search(r"Galbot_G1_([^/]+?)_\d+_\d+", url)
            if match:
                return safe_filename(match.group(1))

    return safe_filename(default_task_id)


def extract_subtasks(item: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    video_labels = item.get("videoLabels")
    validation_reasons: list[dict[str, Any]] = []
    if not isinstance(video_labels, list) or not video_labels:
        return [], [reason("missing_video_labels")]

    subtasks: list[dict[str, Any]] = []
    for label_index, label in enumerate(video_labels):
        if not isinstance(label, dict):
            validation_reasons.append(reason("invalid_video_label", subtask_index=label_index))
            continue

        ranges = label.get("ranges")
        if not isinstance(ranges, list) or len(ranges) != 1:
            validation_reasons.append(reason("invalid_range_count", subtask_index=label_index))
            continue
        frame_range = ranges[0]
        if not isinstance(frame_range, dict):
            validation_reasons.append(reason("invalid_range_object", subtask_index=label_index))
            continue

        labels = label.get("timelinelabels")
        if not isinstance(labels, list) or len(labels) != 1:
            validation_reasons.append(reason("invalid_label_count", subtask_index=label_index))
            continue
        description = labels[0]
        if not isinstance(description, str) or not description.strip():
            validation_reasons.append(reason("invalid_label_text", subtask_index=label_index))
            continue

        try:
            start = coerce_frame(frame_range.get("start"))
            end = coerce_frame(frame_range.get("end"))
        except ValueError:
            validation_reasons.append(
                reason("invalid_frame_type", subtask_index=label_index, value=frame_range)
            )
            continue
        if start < 0:
            validation_reasons.append(reason("negative_start", subtask_index=label_index, value=start))
            continue
        if end < start:
            validation_reasons.append(
                reason("end_before_start", subtask_index=label_index, value={"start": start, "end": end})
            )
            continue

        subtasks.append(
            {
                "description": description.strip(),
                "start_frame": start,
                "end_frame": end,
            }
        )

    subtasks.sort(key=lambda subtask: (subtask["start_frame"], subtask["end_frame"]))
    validation_reasons.extend(validate_subtasks(subtasks))
    return subtasks, validation_reasons


def validate_subtasks(subtasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validation_reasons: list[dict[str, Any]] = []
    if not subtasks:
        return validation_reasons

    first_start = subtasks[0]["start_frame"]
    allowed_starts = {0, 1} if ALLOW_FIRST_START_ONE else {0}
    if first_start not in allowed_starts:
        validation_reasons.append(
            reason(
                "invalid_first_start",
                subtask_index=0,
                value=first_start,
                expected=sorted(allowed_starts),
            )
        )

    previous_end: int | None = None
    for index, subtask in enumerate(subtasks):
        start = subtask["start_frame"]
        end = subtask["end_frame"]
        if previous_end is not None:
            if start <= previous_end:
                validation_reasons.append(
                    reason("overlap", subtask_index=index, value={"start": start, "previous_end": previous_end})
                )
            elif start != previous_end + 1:
                validation_reasons.append(
                    reason("not_continuous", subtask_index=index, value=start, expected=previous_end + 1)
                )
        previous_end = end
    return validation_reasons


def load_annotation_groups(
    path: Path,
    *,
    default_task_id: str = DEFAULT_TASK_ID,
) -> dict[str, list[AnnotationRecord]]:
    raw_items = load_raw_annotations(path)
    groups: dict[str, list[AnnotationRecord]] = defaultdict(list)
    for item_index, item in enumerate(raw_items):
        task_id = infer_task_id(item, default_task_id)
        subtasks, validation_reasons = extract_subtasks(item)
        groups[task_id].append(
            AnnotationRecord(
                task_id=task_id,
                source_index=item_index,
                original_id=item.get("id"),
                subtasks=subtasks,
                validation_reasons=validation_reasons,
            )
        )
    return dict(groups)

