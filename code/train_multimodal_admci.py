#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "dataset_4"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"

GROUPS = ("AD", "MCI", "HC")
EMOTIONS = ("happy", "disgust", "fear", "angry", "neutral", "sad", "surprise")

# Only the six oculomotor features in Table 2 are model inputs.  Switching rate
# and transition entropy belong to the statistical-analysis branch of the paper.
BEHAVIOR_COLUMNS = (
    "Number_of_fixations",
    "Total_duration_of_fixations",
    "Time_to_first_whole_fixation",
    "Number_of_saccades_in_AOI",
    "Peak_velocity_of_exit_saccade",
    "Pupil_CV",
)

RAW_SEQUENCE_COLUMNS = (
    "Recording timestamp [μs]",
    "Pupil diameter left [mm]",
    "Pupil diameter right [mm]",
    "Gaze point X [MCS norm]",
    "Gaze point Y [MCS norm]",
    "Eye movement event duration [ms]",
    "Eye movement type",
    "AOI Type",
)

EVENT_TO_INDEX = {"fixation": 0, "saccade": 1, "unclassified": 2}
AOI_TO_INDEX = {"eyes": 0, "nose": 1, "mouth": 2, "background": 3}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    groups: Tuple[str, ...]
    label_to_index: Mapping[str, int]
    positive_group: Optional[str]

    @property
    def num_classes(self) -> int:
        return len(self.label_to_index)


TASKS: Dict[str, TaskSpec] = {
    "adhc": TaskSpec("adhc", ("AD", "HC"), {"HC": 0, "AD": 1}, "AD"),
    "mcihc": TaskSpec("mcihc", ("MCI", "HC"), {"HC": 0, "MCI": 1}, "MCI"),
    "admcihc": TaskSpec(
        "admcihc", ("AD", "MCI", "HC"), {"AD": 0, "MCI": 1, "HC": 2}, None
    ),
}


@dataclass(frozen=True)
class TrialRecord:
    group: str
    subject_id: str
    emotion: str
    sequence_path: Path
    heatmap_path: Path
    behavior: np.ndarray


@dataclass(frozen=True)
class SubjectRecord:
    group: str
    subject_id: str
    trials: Tuple[TrialRecord, ...]

    @property
    def key(self) -> str:
        return f"{self.group}:{self.subject_id}"


@dataclass(frozen=True)
class StandardizationStats:
    sequence_mean: np.ndarray
    sequence_std: np.ndarray
    behavior_mean: np.ndarray
    behavior_std: np.ndarray


@dataclass
class ModelConfig:
    num_classes: int
    modalities: Tuple[str, ...] = ("heatmap", "sequence", "behavior")
    fusion: str = "proposed"
    image_size: int = 224
    max_len: int = 1250
    d_model: int = 128
    continuous_dim: int = 64
    n_heads: int = 2
    transformer_layers: int = 2
    event_embedding_dim: int = 16
    aoi_embedding_dim: int = 16
    classifier_dropout: float = 0.5
    behavior_dropout: float = 0.5
    pretrained_heatmap: bool = True


@dataclass
class TrainConfig:
    data_root: Path
    output_dir: Path
    task: str
    seed: int = 20261019
    n_splits: int = 10
    validation_fraction: float = 0.10
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    early_stopping_patience: int = 10
    scheduler_patience: int = 10
    min_learning_rate: float = 1e-6
    num_workers: int = 0
    device: str = "auto"
    use_amp: bool = True
    min_valid_fraction: float = 0.70
    strict_paper_cohort: bool = False


@dataclass(frozen=True)
class RawSequence:
    numeric: torch.Tensor
    valid_mask: torch.Tensor
    event_type: torch.Tensor
    aoi_type: torch.Tensor
    padding_mask: torch.Tensor
    valid_fraction: float


def normalize_emotion(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("nueral", "neutral").replace("neural", "neutral")
    text = text.replace("angray", "angry")
    for emotion in EMOTIONS:
        if emotion in text:
            return emotion
    return text


def compact_text(value: object) -> str:
    return re.sub(r"[\s_\-]+", "", str(value).strip().lower())


def normalize_event(value: object) -> int:
    key = compact_text(value)
    if "fixation" in key:
        return EVENT_TO_INDEX["fixation"]
    if "saccade" in key:
        return EVENT_TO_INDEX["saccade"]
    return EVENT_TO_INDEX["unclassified"]


def normalize_aoi(value: object) -> int:
    key = compact_text(value)
    if key in {"eye", "eyes"}:
        return AOI_TO_INDEX["eyes"]
    if key in {"nose", "noose"}:
        return AOI_TO_INDEX["nose"]
    if key in {"mouth", "mouse"}:
        return AOI_TO_INDEX["mouth"]
    return AOI_TO_INDEX["background"]


def parse_trial_name(path: Path) -> Optional[Tuple[str, str]]:
    match = re.match(
        r"^(r\d+)(happy|disgust|fear|angry|angray|neutral|neural|nueral|sad|surprise)",
        path.stem,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return match.group(1).lower(), normalize_emotion(match.group(2))


def recording_name(subject_id: str) -> str:
    return f"Recording{re.sub(r'^r', '', subject_id, flags=re.IGNORECASE)}"


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(jsonable(value), stream, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_behavior_table(data_root: Path, group: str) -> Dict[Tuple[str, str], np.ndarray]:
    path = data_root / group / "feature" / f"{group}_processed_features.xlsx"
    if not path.is_file():
        raise FileNotFoundError(f"Missing behavioral-feature table: {path}")
    table = pd.read_excel(path)
    required = {"Recording", "Media", *BEHAVIOR_COLUMNS}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    table = table.copy()
    table["Recording"] = table["Recording"].astype(str).str.strip()
    table["Media"] = table["Media"].map(normalize_emotion)
    for column in BEHAVIOR_COLUMNS:
        table[column] = pd.to_numeric(table[column], errors="coerce")

    result: Dict[Tuple[str, str], np.ndarray] = {}
    for (recording, emotion), rows in table.groupby(["Recording", "Media"], sort=False):
        result[(recording, emotion)] = rows[list(BEHAVIOR_COLUMNS)].mean(
            skipna=True
        ).to_numpy(dtype=np.float32)
    return result


_SEQUENCE_CACHE: Dict[Tuple[str, int], RawSequence] = {}


def read_raw_sequence(path: Path, max_len: int = 1250) -> RawSequence:
    cache_key = (str(path), int(max_len))
    cached = _SEQUENCE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    table = pd.read_excel(path)
    missing = set(RAW_SEQUENCE_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"{path} is missing sequence columns: {sorted(missing)}")

    timestamps = pd.to_numeric(table[RAW_SEQUENCE_COLUMNS[0]], errors="coerce")
    if timestamps.notna().any():
        first_timestamp = float(timestamps[timestamps.notna()].iloc[0])
        relative_time = (timestamps - first_timestamp) / 1_000_000.0
    else:
        relative_time = pd.Series(np.nan, index=table.index, dtype=np.float64)

    numeric_frame = pd.DataFrame(
        {
            "relative_timestamp": relative_time,
            "gaze_x": pd.to_numeric(table["Gaze point X [MCS norm]"], errors="coerce"),
            "gaze_y": pd.to_numeric(table["Gaze point Y [MCS norm]"], errors="coerce"),
            "pupil_left": pd.to_numeric(table["Pupil diameter left [mm]"], errors="coerce"),
            "pupil_right": pd.to_numeric(table["Pupil diameter right [mm]"], errors="coerce"),
            "event_duration": pd.to_numeric(
                table["Eye movement event duration [ms]"], errors="coerce"
            ),
        }
    )
    numeric_values = numeric_frame.to_numpy(dtype=np.float32)
    numeric_values[~np.isfinite(numeric_values)] = np.nan
    row_count = min(len(table), max_len)

    numeric = np.full((max_len, 6), np.nan, dtype=np.float32)
    valid_mask = np.zeros((max_len, 6), dtype=np.float32)
    event_type = np.full(max_len, EVENT_TO_INDEX["unclassified"], dtype=np.int64)
    aoi_type = np.full(max_len, AOI_TO_INDEX["background"], dtype=np.int64)
    padding_mask = np.zeros(max_len, dtype=bool)

    if row_count:
        selected = numeric_values[:row_count]
        numeric[:row_count] = selected
        valid_mask[:row_count] = np.isfinite(selected).astype(np.float32)
        event_type[:row_count] = np.asarray(
            [normalize_event(value) for value in table["Eye movement type"].iloc[:row_count]],
            dtype=np.int64,
        )
        aoi_type[:row_count] = np.asarray(
            [normalize_aoi(value) for value in table["AOI Type"].iloc[:row_count]],
            dtype=np.int64,
        )
        padding_mask[:row_count] = True

    # The reduced dataset does not contain Tobii's per-eye validity columns.
    # A sample is therefore considered valid when the exported averaged x/y
    # gaze coordinates are both finite.  The denominator is the expected
    # 250 Hz x 5 s = 1,250 samples used by the paper.
    gaze_valid = np.isfinite(numeric_values[:, 1]) & np.isfinite(numeric_values[:, 2])
    valid_fraction = float(int(gaze_valid[:max_len].sum()) / max_len)
    result = RawSequence(
        numeric=torch.from_numpy(numeric),
        valid_mask=torch.from_numpy(valid_mask),
        event_type=torch.from_numpy(event_type),
        aoi_type=torch.from_numpy(aoi_type),
        padding_mask=torch.from_numpy(padding_mask),
        valid_fraction=valid_fraction,
    )
    _SEQUENCE_CACHE[cache_key] = result
    return result


def discover_subjects(
    data_root: Path,
    max_len: int,
    min_valid_fraction: float,
    scan_quality: bool = True,
) -> Tuple[List[SubjectRecord], dict]:
    report = {
        "data_root": str(data_root),
        "required_emotions": list(EMOTIONS),
        "min_valid_fraction": min_valid_fraction,
        "reduced_sequence_quality_scan_enabled": scan_quality,
        "validity_definition_when_enabled": (
            "finite exported averaged gaze X/Y divided by 1250 expected samples"
        ),
        "groups": {},
        "excluded_incomplete": [],
        "excluded_low_validity": [],
        "unparsed_files": [],
        "missing_components": [],
    }
    all_subjects: List[SubjectRecord] = []

    for group in GROUPS:
        behavior = load_behavior_table(data_root, group)
        sequence_paths: Dict[Tuple[str, str], Path] = {}
        heatmap_paths: Dict[Tuple[str, str], Path] = {}
        for path in sorted((data_root / group / "sequence").glob("*.xlsx")):
            key = parse_trial_name(path)
            if key is None:
                report["unparsed_files"].append(str(path))
            else:
                sequence_paths[key] = path
        for path in sorted((data_root / group / "heatmap").glob("*.png")):
            key = parse_trial_name(path)
            if key is None:
                report["unparsed_files"].append(str(path))
            else:
                heatmap_paths[key] = path

        subject_ids = sorted({key[0] for key in sequence_paths})
        kept = 0
        for subject_id in subject_ids:
            trials_by_emotion: Dict[str, TrialRecord] = {}
            missing_components: Dict[str, List[str]] = defaultdict(list)
            for emotion in EMOTIONS:
                key = (subject_id, emotion)
                sequence_path = sequence_paths.get(key)
                heatmap_path = heatmap_paths.get(key)
                behavior_vector = behavior.get((recording_name(subject_id), emotion))
                if sequence_path is None:
                    missing_components[emotion].append("sequence")
                if heatmap_path is None:
                    missing_components[emotion].append("heatmap")
                if behavior_vector is None:
                    missing_components[emotion].append("behavior")
                if sequence_path is not None and heatmap_path is not None and behavior_vector is not None:
                    trials_by_emotion[emotion] = TrialRecord(
                        group=group,
                        subject_id=subject_id,
                        emotion=emotion,
                        sequence_path=sequence_path,
                        heatmap_path=heatmap_path,
                        behavior=behavior_vector,
                    )

            if missing_components:
                detail = {
                    "group": group,
                    "subject_id": subject_id,
                    "missing": dict(missing_components),
                }
                report["excluded_incomplete"].append(detail)
                report["missing_components"].append(detail)
                continue

            ordered_trials = tuple(trials_by_emotion[emotion] for emotion in EMOTIONS)
            if scan_quality:
                fractions = [
                    read_raw_sequence(trial.sequence_path, max_len).valid_fraction
                    for trial in ordered_trials
                ]
                overall_fraction = float(np.mean(fractions))
                if overall_fraction < min_valid_fraction or min(fractions) < min_valid_fraction:
                    report["excluded_low_validity"].append(
                        {
                            "group": group,
                            "subject_id": subject_id,
                            "overall_valid_fraction": overall_fraction,
                            "trial_valid_fractions": dict(zip(EMOTIONS, fractions)),
                        }
                    )
                    continue

            all_subjects.append(SubjectRecord(group, subject_id, ordered_trials))
            kept += 1

        report["groups"][group] = {
            "sequence_files": len(sequence_paths),
            "heatmap_files": len(heatmap_paths),
            "feature_rows": len(behavior),
            "discovered_subjects": len(subject_ids),
            "retained_complete_subjects": kept,
        }

    return all_subjects, report


def filter_task_subjects(subjects: Sequence[SubjectRecord], task: TaskSpec) -> List[SubjectRecord]:
    selected = [subject for subject in subjects if subject.group in task.groups]
    counts = Counter(subject.group for subject in selected)
    missing = [group for group in task.groups if counts[group] == 0]
    if missing:
        raise ValueError(f"Task {task.name} has no retained subjects for groups: {missing}")
    return selected


def validate_paper_cohort(subjects: Sequence[SubjectRecord]) -> None:
    observed = Counter(subject.group for subject in subjects)
    expected = {"AD": 35, "MCI": 39, "HC": 64}
    if dict(observed) != expected:
        raise RuntimeError(
            "The retained cohort does not match the PDF cohort. "
            f"expected={expected}, observed={dict(observed)}. See data_report.json."
        )


def fit_standardization_stats(
    subjects: Sequence[SubjectRecord], max_len: int
) -> StandardizationStats:
    sequence_sum = np.zeros(6, dtype=np.float64)
    sequence_square_sum = np.zeros(6, dtype=np.float64)
    sequence_count = np.zeros(6, dtype=np.float64)
    behavior_rows: List[np.ndarray] = []

    for subject in subjects:
        for trial in subject.trials:
            raw = read_raw_sequence(trial.sequence_path, max_len)
            values = raw.numeric.numpy().astype(np.float64, copy=False)
            valid = raw.valid_mask.numpy().astype(bool, copy=False)
            valid &= raw.padding_mask.numpy()[:, None]
            safe_values = np.where(valid, values, 0.0)
            sequence_sum += safe_values.sum(axis=0)
            sequence_square_sum += np.square(safe_values).sum(axis=0)
            sequence_count += valid.sum(axis=0)
            behavior_rows.append(trial.behavior.astype(np.float64, copy=False))

    if np.any(sequence_count == 0):
        raise RuntimeError(
            "At least one sequence feature has no valid training values: "
            f"counts={sequence_count.tolist()}"
        )
    sequence_mean = sequence_sum / sequence_count
    sequence_variance = sequence_square_sum / sequence_count - np.square(sequence_mean)
    sequence_std = np.sqrt(np.maximum(sequence_variance, 0.0))
    sequence_std[sequence_std < 1e-8] = 1.0

    behavior_values = np.stack(behavior_rows)
    behavior_mean = np.nanmean(behavior_values, axis=0)
    behavior_mean[~np.isfinite(behavior_mean)] = 0.0
    behavior_std = np.nanstd(behavior_values, axis=0, ddof=0)
    behavior_std[~np.isfinite(behavior_std) | (behavior_std < 1e-8)] = 1.0
    return StandardizationStats(
        sequence_mean=sequence_mean.astype(np.float32),
        sequence_std=sequence_std.astype(np.float32),
        behavior_mean=behavior_mean.astype(np.float32),
        behavior_std=behavior_std.astype(np.float32),
    )


@lru_cache(maxsize=2048)
def load_heatmap_uint8(path_text: str, image_size: int) -> torch.Tensor:
    with Image.open(path_text) as image:
        image = image.convert("RGB")
        image = image.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class ParticipantDataset(Dataset):
    """Return all seven emotion trials for one participant."""

    image_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    image_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __init__(
        self,
        subjects: Sequence[SubjectRecord],
        task: TaskSpec,
        stats: StandardizationStats,
        image_size: int = 224,
        max_len: int = 1250,
    ) -> None:
        self.subjects = list(subjects)
        self.task = task
        self.stats = stats
        self.image_size = image_size
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, index: int) -> dict:
        subject = self.subjects[index]
        heatmaps = []
        numeric_parts = []
        valid_parts = []
        event_parts = []
        aoi_parts = []
        padding_parts = []
        behavior_parts = []

        sequence_mean = torch.from_numpy(self.stats.sequence_mean)
        sequence_std = torch.from_numpy(self.stats.sequence_std)
        behavior_mean = torch.from_numpy(self.stats.behavior_mean)
        behavior_std = torch.from_numpy(self.stats.behavior_std)

        for trial in subject.trials:
            image = load_heatmap_uint8(str(trial.heatmap_path), self.image_size).float() / 255.0
            heatmaps.append((image - self.image_mean) / self.image_std)

            raw = read_raw_sequence(trial.sequence_path, self.max_len)
            standardized = (raw.numeric - sequence_mean) / sequence_std
            standardized = torch.where(raw.valid_mask.bool(), standardized, torch.zeros_like(standardized))
            standardized = torch.where(
                raw.padding_mask[:, None], standardized, torch.zeros_like(standardized)
            )
            numeric_parts.append(standardized)
            valid_parts.append(raw.valid_mask)
            event_parts.append(raw.event_type)
            aoi_parts.append(raw.aoi_type)
            padding_parts.append(raw.padding_mask)

            behavior = torch.from_numpy(trial.behavior.astype(np.float32, copy=True))
            behavior_valid = torch.isfinite(behavior)
            behavior = (behavior - behavior_mean) / behavior_std
            behavior_parts.append(torch.where(behavior_valid, behavior, torch.zeros_like(behavior)))

        return {
            "heatmap": torch.stack(heatmaps),
            "sequence_numeric": torch.stack(numeric_parts),
            "sequence_valid_mask": torch.stack(valid_parts),
            "event_type": torch.stack(event_parts),
            "aoi_type": torch.stack(aoi_parts),
            "sequence_padding_mask": torch.stack(padding_parts),
            "behavior": torch.stack(behavior_parts),
            "label": torch.tensor(self.task.label_to_index[subject.group], dtype=torch.long),
            "subject_key": subject.key,
            "group": subject.group,
        }


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.encoding[:, : values.size(1)]


class HeatmapEncoder(nn.Module):
    """ImageNet-pretrained EfficientNet-B0 producing 49 spatial tokens."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        weights = EfficientNet_B0_Weights.DEFAULT if config.pretrained_heatmap else None
        try:
            model = efficientnet_b0(weights=weights)
        except TypeError:  # torchvision versions predating the weights enum API
            model = efficientnet_b0(pretrained=config.pretrained_heatmap)
        except Exception as error:
            if config.pretrained_heatmap:
                raise RuntimeError(
                    "ImageNet-1K EfficientNet-B0 weights could not be loaded. "
                    "Cache/download the weights or use --no-pretrained-heatmap only for tests."
                ) from error
            raise
        self.backbone = model.features
        self.projection = nn.Sequential(
            nn.Linear(1280, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )
        self.position = nn.Parameter(torch.zeros(1, 49, config.d_model))
        nn.init.normal_(self.position, mean=0.0, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(images)
        batch_size, channels, height, width = feature_map.shape
        if (height, width) != (7, 7):
            raise RuntimeError(f"Expected a 7x7 EfficientNet feature map, got {height}x{width}")
        tokens = feature_map.reshape(batch_size, channels, 49).transpose(1, 2)
        return self.projection(tokens) + self.position


class SequenceEncoder(nn.Module):
    """Two-layer, two-head Transformer over 1,250 eye-tracking samples."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.continuous_projection = nn.Linear(6, config.continuous_dim)
        self.event_embedding = nn.Embedding(3, config.event_embedding_dim)
        self.aoi_embedding = nn.Embedding(4, config.aoi_embedding_dim)
        combined_dim = (
            config.continuous_dim
            + 6
            + config.event_embedding_dim
            + config.aoi_embedding_dim
        )
        self.input_projection = nn.Sequential(
            nn.Linear(combined_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )
        self.position = SinusoidalPositionalEncoding(config.d_model, config.max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.transformer_layers)

    def forward(
        self,
        numeric: torch.Tensor,
        valid_mask: torch.Tensor,
        event_type: torch.Tensor,
        aoi_type: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        continuous = self.continuous_projection(numeric)
        values = torch.cat(
            (
                continuous,
                valid_mask,
                self.aoi_embedding(aoi_type.long()),
                self.event_embedding(event_type.long()),
            ),
            dim=-1,
        )
        values = self.position(self.input_projection(values))
        valid_tokens = padding_mask.bool()
        tokens = self.transformer(values, src_key_padding_mask=~valid_tokens)
        tokens = tokens * valid_tokens.unsqueeze(-1).to(tokens.dtype)
        return tokens, valid_tokens


class BehaviorEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(config.behavior_dropout),
            nn.Linear(64, config.d_model),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(tokens.dtype)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class SoftmaxGate(nn.Module):
    def __init__(self, d_model: int, input_count: int) -> None:
        super().__init__()
        self.input_count = input_count
        self.gate = nn.Sequential(
            nn.Linear(d_model * input_count, d_model),
            nn.GELU(),
            nn.Linear(d_model, input_count),
        )

    def forward(self, representations: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(representations) != self.input_count:
            raise ValueError(f"Expected {self.input_count} gate inputs, got {len(representations)}")
        weights = torch.softmax(self.gate(torch.cat(tuple(representations), dim=-1)), dim=-1)
        stacked = torch.stack(tuple(representations), dim=1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return fused, weights


class TrialFusion(nn.Module):
    """Fusion module shared by full-model and modality-ablation experiments."""

    modality_order = ("heatmap", "sequence", "behavior")

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.modalities = tuple(
            modality for modality in self.modality_order if modality in config.modalities
        )
        if not self.modalities:
            raise ValueError("At least one modality must be enabled")
        self.cross_attention = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=0.0, batch_first=True
        )

        proposed_input_count = 1 + int("behavior" in self.modalities)
        self.proposed_gate = (
            SoftmaxGate(config.d_model, proposed_input_count)
            if proposed_input_count > 1
            else None
        )
        self.modality_gate = (
            SoftmaxGate(config.d_model, len(self.modalities))
            if len(self.modalities) > 1
            else None
        )
        self.concat_projection = nn.Sequential(
            nn.Linear(config.d_model * len(self.modalities), config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )
        self.modality_position = nn.Parameter(
            torch.zeros(1, len(self.modalities), config.d_model)
        )
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.modality_transformer = nn.TransformerEncoder(transformer_layer, num_layers=1)
        nn.init.normal_(self.modality_position, mean=0.0, std=0.02)

    def forward(
        self,
        heatmap_tokens: Optional[torch.Tensor],
        sequence_tokens: Optional[torch.Tensor],
        sequence_mask: Optional[torch.Tensor],
        behavior_code: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pooled: Dict[str, torch.Tensor] = {}
        if heatmap_tokens is not None:
            pooled["heatmap"] = heatmap_tokens.mean(dim=1)
        if sequence_tokens is not None and sequence_mask is not None:
            pooled["sequence"] = masked_mean(sequence_tokens, sequence_mask)
        if behavior_code is not None:
            pooled["behavior"] = behavior_code

        if self.config.fusion == "proposed":
            if heatmap_tokens is not None and sequence_tokens is not None and sequence_mask is not None:
                cross_tokens, _ = self.cross_attention(
                    query=sequence_tokens,
                    key=heatmap_tokens,
                    value=heatmap_tokens,
                    need_weights=False,
                )
                cross_code = masked_mean(cross_tokens, sequence_mask)
                inputs = [cross_code]
                if behavior_code is not None:
                    inputs.append(behavior_code)
            else:
                inputs = [pooled[name] for name in self.modalities]
            if len(inputs) == 1:
                return inputs[0], None
            if len(inputs) == getattr(self.proposed_gate, "input_count", -1):
                return self.proposed_gate(inputs)
            return self.modality_gate(inputs)

        representations = [pooled[name] for name in self.modalities]
        if len(representations) == 1:
            return representations[0], None
        if self.config.fusion == "concatenation":
            return self.concat_projection(torch.cat(representations, dim=-1)), None
        if self.config.fusion == "gated":
            return self.modality_gate(representations)
        if self.config.fusion == "transformer":
            tokens = torch.stack(representations, dim=1) + self.modality_position
            return self.modality_transformer(tokens).mean(dim=1), None
        raise ValueError(f"Unknown fusion strategy: {self.config.fusion}")


class ParticipantMultimodalModel(nn.Module):
    """Encode trials, average seven trial representations, and classify a participant."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.heatmap_encoder = HeatmapEncoder(config) if "heatmap" in config.modalities else None
        self.sequence_encoder = SequenceEncoder(config) if "sequence" in config.modalities else None
        self.behavior_encoder = BehaviorEncoder(config) if "behavior" in config.modalities else None
        self.fusion = TrialFusion(config)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.d_model, config.num_classes),
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict:
        batch_size, emotion_count = batch["heatmap"].shape[:2]
        heatmap_tokens = None
        sequence_tokens = None
        sequence_mask = None
        behavior_code = None

        if self.heatmap_encoder is not None:
            images = batch["heatmap"].flatten(0, 1).float()
            heatmap_tokens = self.heatmap_encoder(images)
        if self.sequence_encoder is not None:
            sequence_tokens, sequence_mask = self.sequence_encoder(
                batch["sequence_numeric"].flatten(0, 1).float(),
                batch["sequence_valid_mask"].flatten(0, 1).float(),
                batch["event_type"].flatten(0, 1),
                batch["aoi_type"].flatten(0, 1),
                batch["sequence_padding_mask"].flatten(0, 1),
            )
        if self.behavior_encoder is not None:
            behavior_code = self.behavior_encoder(batch["behavior"].flatten(0, 1).float())

        trial_code, gates = self.fusion(
            heatmap_tokens, sequence_tokens, sequence_mask, behavior_code
        )
        trial_code = trial_code.reshape(batch_size, emotion_count, -1)
        participant_code = trial_code.mean(dim=1)
        output = {
            "logits": self.classifier(participant_code),
            "participant_code": participant_code,
            "trial_code": trial_code,
        }
        if gates is not None:
            output["gates"] = gates.reshape(batch_size, emotion_count, -1)
        return output


def move_batch(batch: dict, device: torch.device) -> dict:
    non_blocking = device.type == "cuda"
    return {
        key: value.to(device, non_blocking=non_blocking) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    task: TaskSpec,
    loss: float,
) -> dict:
    predictions = probabilities.argmax(axis=1)
    if task.num_classes == 2:
        positive_index = task.label_to_index[task.positive_group]
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            pos_label=positive_index,
            zero_division=0,
        )
        try:
            auc = roc_auc_score(labels, probabilities[:, positive_index])
        except ValueError:
            auc = float("nan")
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="macro", zero_division=0
        )
        try:
            auc = roc_auc_score(
                labels,
                probabilities,
                labels=list(range(task.num_classes)),
                multi_class="ovr",
                average="macro",
            )
        except ValueError:
            auc = float("nan")
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=list(range(task.num_classes))
        ).tolist(),
        "n_subjects": int(len(labels)),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    task: TaskSpec,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = True,
    return_predictions: bool = False,
) -> dict:
    training = optimizer is not None
    model.train(training)
    amp_enabled = bool(use_amp and device.type == "cuda")
    loss_sum = 0.0
    subject_count = 0
    label_parts: List[np.ndarray] = []
    probability_parts: List[np.ndarray] = []
    subject_keys: List[str] = []

    with torch.set_grad_enabled(training):
        for batch in loader:
            keys = list(batch["subject_key"])
            batch = move_batch(batch, device)
            labels = batch["label"]
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(batch)
                loss = criterion(output["logits"], labels)
            if training:
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            probabilities = torch.softmax(output["logits"].detach(), dim=-1)
            current_count = int(labels.size(0))
            loss_sum += float(loss.detach().cpu()) * current_count
            subject_count += current_count
            label_parts.append(labels.detach().cpu().numpy())
            probability_parts.append(probabilities.cpu().numpy())
            subject_keys.extend(keys)

    if not label_parts:
        raise RuntimeError("The DataLoader produced no participant batches")
    labels = np.concatenate(label_parts)
    probabilities = np.concatenate(probability_parts)
    metrics = classification_metrics(
        labels, probabilities, task, loss_sum / max(subject_count, 1)
    )
    if return_predictions:
        index_to_label = {index: group for group, index in task.label_to_index.items()}
        predictions = probabilities.argmax(axis=1)
        metrics["predictions"] = [
            {
                "subject_key": subject_key,
                "true_index": int(label),
                "true_label": index_to_label[int(label)],
                "predicted_index": int(prediction),
                "predicted_label": index_to_label[int(prediction)],
                "probabilities": probability.tolist(),
            }
            for subject_key, label, prediction, probability in zip(
                subject_keys, labels, predictions, probabilities
            )
        ]
    return metrics


def make_loader(
    subjects: Sequence[SubjectRecord],
    task: TaskSpec,
    stats: StandardizationStats,
    model_config: ModelConfig,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = ParticipantDataset(
        subjects,
        task,
        stats,
        image_size=model_config.image_size,
        max_len=model_config.max_len,
    )
    arguments = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        arguments["persistent_workers"] = True
    return DataLoader(**arguments)


def class_weights(subjects: Sequence[SubjectRecord], task: TaskSpec) -> torch.Tensor:
    counts = np.bincount(
        [task.label_to_index[subject.group] for subject in subjects],
        minlength=task.num_classes,
    ).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError(f"Training partition is missing a class: {counts.tolist()}")
    proportions = counts / counts.sum()
    return torch.tensor(1.0 / proportions, dtype=torch.float32)


def split_outer_and_validation(
    subjects: Sequence[SubjectRecord],
    task: TaskSpec,
    n_splits: int,
    validation_fraction: float,
    seed: int,
) -> List[Tuple[List[int], List[int], List[int]]]:
    labels = np.asarray([task.label_to_index[subject.group] for subject in subjects])
    indices = np.arange(len(subjects))
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for fold_index, (development_indices, test_indices) in enumerate(outer.split(indices, labels)):
        internal = StratifiedShuffleSplit(
            n_splits=1,
            test_size=validation_fraction,
            random_state=seed + fold_index + 1,
        )
        internal_train, internal_validation = next(
            internal.split(development_indices, labels[development_indices])
        )
        train_indices = development_indices[internal_train]
        validation_indices = development_indices[internal_validation]
        train_keys = {subjects[index].key for index in train_indices}
        validation_keys = {subjects[index].key for index in validation_indices}
        test_keys = {subjects[index].key for index in test_indices}
        if train_keys & validation_keys or train_keys & test_keys or validation_keys & test_keys:
            raise RuntimeError("Participant leakage detected in cross-validation split")
        splits.append(
            (train_indices.tolist(), validation_indices.tolist(), test_indices.tolist())
        )
    return splits


def subset(subjects: Sequence[SubjectRecord], indices: Sequence[int]) -> List[SubjectRecord]:
    return [subjects[index] for index in indices]


def partition_report(subjects: Sequence[SubjectRecord]) -> dict:
    return {
        "n_subjects": len(subjects),
        "class_counts": dict(Counter(subject.group for subject in subjects)),
        "subject_keys": [subject.key for subject in subjects],
    }


def print_epoch(
    fold_index: int,
    epoch: int,
    learning_rate: float,
    train_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, float],
) -> None:
    print(
        f"fold={fold_index:02d} epoch={epoch:03d} lr={learning_rate:.3e} "
        f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['f1']:.4f} "
        f"val_loss={validation_metrics['loss']:.4f} "
        f"val_accuracy={validation_metrics['accuracy']:.4f} "
        f"val_f1={validation_metrics['f1']:.4f} "
        f"val_auc={validation_metrics['auc']:.4f}",
        flush=True,
    )


def train_fold(
    fold_index: int,
    train_subjects: Sequence[SubjectRecord],
    validation_subjects: Sequence[SubjectRecord],
    test_subjects: Sequence[SubjectRecord],
    task: TaskSpec,
    train_config: TrainConfig,
    model_config: ModelConfig,
) -> dict:
    set_seed(train_config.seed + fold_index)
    fold_dir = train_config.output_dir / f"fold_{fold_index:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    stats = fit_standardization_stats(train_subjects, model_config.max_len)
    train_loader = make_loader(
        train_subjects,
        task,
        stats,
        model_config,
        train_config.batch_size,
        train_config.num_workers,
        shuffle=True,
    )
    validation_loader = make_loader(
        validation_subjects,
        task,
        stats,
        model_config,
        train_config.batch_size,
        train_config.num_workers,
        shuffle=False,
    )
    test_loader = make_loader(
        test_subjects,
        task,
        stats,
        model_config,
        train_config.batch_size,
        train_config.num_workers,
        shuffle=False,
    )

    device = resolve_device(train_config.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    model = ParticipantMultimodalModel(model_config).to(device)
    weights = class_weights(train_subjects, task).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=train_config.scheduler_patience,
        min_lr=train_config.min_learning_rate,
    )
    amp_enabled = bool(train_config.use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    split_info = {
        "train": partition_report(train_subjects),
        "validation": partition_report(validation_subjects),
        "test": partition_report(test_subjects),
    }
    write_json(fold_dir / "split.json", split_info)
    history = []
    best_f1 = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = fold_dir / "best_model.pt"

    print(
        f"fold={fold_index:02d} device={device} train={len(train_subjects)} "
        f"validation={len(validation_subjects)} test={len(test_subjects)} "
        f"class_weights={weights.detach().cpu().tolist()}",
        flush=True,
    )
    for epoch in range(1, train_config.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            task,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=amp_enabled,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            criterion,
            task,
            device,
            use_amp=amp_enabled,
        )
        scheduler.step(validation_metrics["loss"])
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        print_epoch(fold_index, epoch, current_lr, train_metrics, validation_metrics)

        if validation_metrics["f1"] > best_f1 + 1e-12:
            best_f1 = validation_metrics["f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": asdict(model_config),
                    "task": task.name,
                    "label_to_index": dict(task.label_to_index),
                    "feature_columns": list(BEHAVIOR_COLUMNS),
                    "emotion_order": list(EMOTIONS),
                    "sequence_mean": stats.sequence_mean,
                    "sequence_std": stats.sequence_std,
                    "behavior_mean": stats.behavior_mean,
                    "behavior_std": stats.behavior_std,
                    "best_epoch": best_epoch,
                    "best_validation_metrics": validation_metrics,
                    "class_weights": weights.detach().cpu(),
                    "split": split_info,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_config.early_stopping_patience:
                print(f"fold={fold_index:02d} early stopping at epoch {epoch}", flush=True)
                break

    if not checkpoint_path.is_file():
        raise RuntimeError(f"No checkpoint was saved for fold {fold_index}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        task,
        device,
        use_amp=amp_enabled,
        return_predictions=True,
    )
    result = {
        "fold": fold_index,
        "best_epoch": best_epoch,
        "best_validation": checkpoint["best_validation_metrics"],
        "test": test_metrics,
        "split": split_info,
    }
    write_json(fold_dir / "history.json", history)
    write_json(fold_dir / "result.json", result)
    print(
        f"fold={fold_index:02d} TEST accuracy={test_metrics['accuracy']:.4f} "
        f"precision={test_metrics['precision']:.4f} recall={test_metrics['recall']:.4f} "
        f"f1={test_metrics['f1']:.4f} auc={test_metrics['auc']:.4f}",
        flush=True,
    )
    return result


def summarize_folds(fold_results: Sequence[dict], task: TaskSpec) -> dict:
    metric_names = ("accuracy", "precision", "recall", "f1", "auc")
    summary = {"test_fold_mean_std": {}, "folds": list(fold_results)}
    for metric in metric_names:
        values = np.asarray([fold["test"][metric] for fold in fold_results], dtype=np.float64)
        summary["test_fold_mean_std"][metric] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values, ddof=1)),
            "values": values.tolist(),
        }

    predictions = []
    for fold in fold_results:
        for row in fold["test"]["predictions"]:
            predictions.append({"fold": fold["fold"], **row})
    subject_keys = [row["subject_key"] for row in predictions]
    duplicates = [key for key, count in Counter(subject_keys).items() if count != 1]
    if duplicates:
        raise RuntimeError(f"Out-of-fold subjects did not occur exactly once: {duplicates}")
    labels = np.asarray([row["true_index"] for row in predictions], dtype=np.int64)
    probabilities = np.asarray([row["probabilities"] for row in predictions], dtype=np.float64)
    summary["pooled_out_of_fold"] = classification_metrics(
        labels, probabilities, task, loss=float("nan")
    )
    summary["out_of_fold_predictions"] = predictions
    return summary


def run_task(
    all_subjects: Sequence[SubjectRecord],
    task: TaskSpec,
    base_output: Path,
    arguments: argparse.Namespace,
) -> dict:
    subjects = filter_task_subjects(all_subjects, task)
    task_output = base_output / task.name
    task_output.mkdir(parents=True, exist_ok=True)
    model_config = ModelConfig(
        num_classes=task.num_classes,
        modalities=tuple(arguments.modalities),
        fusion=arguments.fusion,
        image_size=arguments.image_size,
        max_len=arguments.max_len,
        d_model=arguments.d_model,
        continuous_dim=arguments.continuous_dim,
        n_heads=arguments.n_heads,
        transformer_layers=arguments.transformer_layers,
        classifier_dropout=arguments.dropout,
        behavior_dropout=arguments.dropout,
        pretrained_heatmap=arguments.pretrained_heatmap,
    )
    train_config = TrainConfig(
        data_root=arguments.data_root,
        output_dir=task_output,
        task=task.name,
        seed=arguments.seed,
        n_splits=arguments.folds,
        validation_fraction=arguments.validation_fraction,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        early_stopping_patience=arguments.early_stopping_patience,
        scheduler_patience=arguments.scheduler_patience,
        min_learning_rate=arguments.min_learning_rate,
        num_workers=arguments.num_workers,
        device=arguments.device,
        use_amp=arguments.amp,
        min_valid_fraction=arguments.min_valid_fraction,
        strict_paper_cohort=arguments.strict_paper_cohort,
    )
    write_json(
        task_output / "run_config.json",
        {
            "train": asdict(train_config),
            "model": asdict(model_config),
            "task": asdict(task),
            "cohort": partition_report(subjects),
        },
    )
    splits = split_outer_and_validation(
        subjects,
        task,
        train_config.n_splits,
        train_config.validation_fraction,
        train_config.seed,
    )
    write_json(
        task_output / "fold_assignments.json",
        [
            {
                "fold": fold_index,
                "train": [subjects[index].key for index in train_indices],
                "validation": [subjects[index].key for index in validation_indices],
                "test": [subjects[index].key for index in test_indices],
            }
            for fold_index, (train_indices, validation_indices, test_indices) in enumerate(splits)
        ],
    )
    if arguments.audit_only:
        return {"task": task.name, "cohort": partition_report(subjects), "folds": len(splits)}

    fold_results = []
    for fold_index, (train_indices, validation_indices, test_indices) in enumerate(splits):
        fold_results.append(
            train_fold(
                fold_index,
                subset(subjects, train_indices),
                subset(subjects, validation_indices),
                subset(subjects, test_indices),
                task,
                train_config,
                model_config,
            )
        )
    summary = summarize_folds(fold_results, task)
    write_json(task_output / "cv_summary.json", summary)
    return summary


def parse_modalities(values: Sequence[str]) -> Tuple[str, ...]:
    valid = {"heatmap", "sequence", "behavior"}
    result = []
    for value in values:
        for item in value.split(","):
            normalized = item.strip().lower()
            if normalized and normalized not in result:
                result.append(normalized)
    unknown = set(result) - valid
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown modalities: {sorted(unknown)}")
    if not result:
        raise argparse.ArgumentTypeError("At least one modality is required")
    return tuple(result)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PDF-aligned AD/HC, MCI/HC, and AD/MCI/HC eye-tracking classification"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--task", choices=("adhc", "mcihc", "admcihc", "all"), default="all")
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["heatmap", "sequence", "behavior"],
        help="Any subset of: heatmap sequence behavior (comma-separated is also accepted)",
    )
    parser.add_argument(
        "--fusion",
        choices=("proposed", "concatenation", "gated", "transformer"),
        default="proposed",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-len", type=int, default=1250)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--continuous-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-valid-fraction", type=float, default=0.70)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--recheck-reduced-validity",
        action="store_true",
        help=(
            "Optionally approximate the 70%% Tobii validity QC from reduced sequence files. "
            "Disabled by default because exact per-eye validity flags are not present."
        ),
    )
    parser.add_argument("--strict-paper-cohort", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--pretrained-heatmap", dest="pretrained_heatmap", action="store_true", default=True
    )
    parser.add_argument(
        "--no-pretrained-heatmap", dest="pretrained_heatmap", action="store_false"
    )
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    return parser


def validate_arguments(arguments: argparse.Namespace) -> None:
    arguments.modalities = parse_modalities(arguments.modalities)
    if arguments.max_len != 1250:
        print(
            f"WARNING: --max-len={arguments.max_len} differs from the PDF value 1250.",
            flush=True,
        )
    if arguments.d_model % arguments.n_heads:
        raise ValueError("d-model must be divisible by n-heads")
    if arguments.folds < 2:
        raise ValueError("folds must be at least 2")
    if not 0.0 < arguments.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be between 0 and 1")
    if not 0.0 <= arguments.min_valid_fraction <= 1.0:
        raise ValueError("min-valid-fraction must be between 0 and 1")
    if arguments.fusion == "proposed" and not {
        "heatmap",
        "sequence",
    }.issubset(arguments.modalities):
        print(
            "WARNING: proposed cross-attention requires heatmap+sequence; with this modality "
            "subset the implementation falls back to the PDF-aligned softmax gate/direct encoder.",
            flush=True,
        )


def run_smoke_test(arguments: argparse.Namespace) -> None:
    config = ModelConfig(
        num_classes=3,
        modalities=tuple(arguments.modalities),
        fusion=arguments.fusion,
        image_size=224,
        max_len=32,
        d_model=32,
        continuous_dim=16,
        n_heads=2,
        transformer_layers=2,
        classifier_dropout=0.5,
        behavior_dropout=0.5,
        pretrained_heatmap=False,
    )
    model = ParticipantMultimodalModel(config).eval()
    batch_size, emotion_count, sequence_length = 1, 7, config.max_len
    batch = {
        "heatmap": torch.randn(batch_size, emotion_count, 3, 224, 224),
        "sequence_numeric": torch.randn(batch_size, emotion_count, sequence_length, 6),
        "sequence_valid_mask": torch.ones(batch_size, emotion_count, sequence_length, 6),
        "event_type": torch.zeros(batch_size, emotion_count, sequence_length, dtype=torch.long),
        "aoi_type": torch.zeros(batch_size, emotion_count, sequence_length, dtype=torch.long),
        "sequence_padding_mask": torch.ones(
            batch_size, emotion_count, sequence_length, dtype=torch.bool
        ),
        "behavior": torch.randn(batch_size, emotion_count, 6),
    }
    with torch.no_grad():
        output = model(batch)
    assert output["logits"].shape == (1, 3)
    assert output["trial_code"].shape == (1, 7, config.d_model)
    assert output["participant_code"].shape == (1, config.d_model)
    print(
        "Smoke test passed: "
        f"logits={tuple(output['logits'].shape)} "
        f"trial_code={tuple(output['trial_code'].shape)} "
        f"participant_code={tuple(output['participant_code'].shape)}"
    )


def main() -> None:
    parser = make_parser()
    arguments = parser.parse_args()
    validate_arguments(arguments)
    if arguments.smoke_test:
        run_smoke_test(arguments)
        return

    arguments.data_root = arguments.data_root.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    modality_name = "-".join(arguments.modalities)
    base_output = arguments.output_root.resolve() / (
        f"pdf_method_{arguments.fusion}_{modality_name}_{timestamp}"
    )
    base_output.mkdir(parents=True, exist_ok=False)
    set_seed(arguments.seed)

    subjects, data_report = discover_subjects(
        arguments.data_root,
        arguments.max_len,
        arguments.min_valid_fraction,
        scan_quality=arguments.recheck_reduced_validity,
    )
    write_json(base_output / "data_report.json", data_report)
    print("Retained cohort:", dict(Counter(subject.group for subject in subjects)), flush=True)
    if data_report["excluded_incomplete"]:
        print(
            f"Excluded incomplete participants: {len(data_report['excluded_incomplete'])}; "
            "see data_report.json",
            flush=True,
        )
    if data_report["excluded_low_validity"]:
        print(
            f"Excluded participants below 70% validity: "
            f"{len(data_report['excluded_low_validity'])}; see data_report.json",
            flush=True,
        )
    if arguments.strict_paper_cohort:
        validate_paper_cohort(subjects)

    write_json(
        base_output / "invocation.json",
        {
            "arguments": vars(arguments),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
    )
    selected_tasks = list(TASKS.values()) if arguments.task == "all" else [TASKS[arguments.task]]
    summaries = {}
    for task in selected_tasks:
        summaries[task.name] = run_task(subjects, task, base_output, arguments)
    write_json(base_output / "all_tasks_summary.json", summaries)
    print(f"Completed. Results: {base_output}", flush=True)


if __name__ == "__main__":
    main()
