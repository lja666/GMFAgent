# -*- coding: utf-8 -*-
"""File utilities for GMPE model selection: list dir, read CSV."""
import csv
from pathlib import Path

try:
    from config import GMPE_ROOT
except ImportError:
    GMPE_ROOT = Path(__file__).resolve().parent.parent / ".." / "gmpe_root1"


def get_gmpe_root() -> Path:
    """Return GMPE model library root path."""
    return Path(GMPE_ROOT)


def list_gmpe_files(directory: str = None) -> list:
    """List relative paths of all files under GMPE directory."""
    base = Path(directory or GMPE_ROOT)
    if not base.exists():
        return [f"Directory not found: {base}"]
    files = []
    for item in base.rglob("*"):
        if item.is_file():
            files.append(str(item.relative_to(base)))
    return files


def read_gmpe_csv(name: str, encoding: str = 'utf-8', directory: str = None) -> list:
    """Read CSV under GMPE dir. Returns list of dicts."""
    base = Path(directory or GMPE_ROOT)
    path = base / name
    try:
        with open(path, "r", encoding=encoding) as f:
            reader = csv.DictReader(f)
            return [row for row in reader]
    except UnicodeDecodeError as e:
        return [f"Encoding error: {e}. Try another encoding."]
    except FileNotFoundError:
        return [f"File '{name}' not found."]
    except Exception as e:
        return [f"Read error: {e}"]
