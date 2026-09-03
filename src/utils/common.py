"""
src/utils/common.py
───────────────────
Shared utilities: config loading, logging setup, path resolution.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml


# ── Project root ──────────────────────────────────────────────────────────────
# Resolved relative to this file: drc_cl_project/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Config ────────────────────────────────────────────────────────────────────
def load_config(config_path: str | Path | None = None) -> dict:
    """Load YAML config.  Defaults to configs/config.yaml under project root."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Resolve all path values to absolute
    for key, val in cfg.get("paths", {}).items():
        cfg["paths"][key] = str(PROJECT_ROOT / val)
    return cfg


def get_path(cfg: dict, key: str) -> Path:
    """Return Path object for a named path in cfg['paths'], creating it if needed."""
    p = Path(cfg["paths"][key])
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Logging ───────────────────────────────────────────────────────────────────
def get_logger(name: str, log_dir: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger that writes to stdout + optional log file.
    Call once per script; subsequent calls with the same name return the same logger.
    """
    logger = logging.getLogger(name)
    if logger.handlers:          # already configured
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # File handler (optional)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ── Quarter helpers ───────────────────────────────────────────────────────────
def quarter_id(year: int, quarter: int) -> str:
    """Return zero-padded window id string, e.g. (2018, 1) → 'D01'."""
    # Map (year, quarter) → sequential index 1..24 for 2018Q1..2023Q4
    base_year = 2018
    idx = (year - base_year) * 4 + quarter   # 1-based
    return f"D{idx:02d}"


def quarter_label(year: int, quarter: int) -> str:
    """Human-readable label: (2018, 1) → '2018_Q1'."""
    return f"{year}_Q{quarter}"


# ── Window list từ config (thay thế hard-code 2018/24) ───────────────────────
def get_window_ids(cfg: dict) -> list[str]:
    """
    Sinh danh sách window_ids từ config temporal.start và temporal.end.
    Ví dụ: start=2020-01-01, end=2025-06-30 → ['D01','D02',...,'D22']
    """
    import pandas as pd
    start = pd.Timestamp(cfg["temporal"]["start"])
    end   = pd.Timestamp(cfg["temporal"]["end"])
    quarters = pd.date_range(start=start, end=end, freq="QS")
    return [f"D{i+1:02d}" for i in range(len(quarters))]


def get_base_year(cfg: dict) -> int:
    """Năm bắt đầu từ config."""
    import pandas as pd
    return pd.Timestamp(cfg["temporal"]["start"]).year


def year_of_window(win_id: str, cfg: dict) -> int:
    """Trả về năm lịch của window_id, tính từ base_year trong config."""
    base = get_base_year(cfg)
    idx  = int(win_id[1:])          # 1-based
    return base + (idx - 1) // 4
