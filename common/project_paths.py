"""Path and output helpers for the self-contained AR-SEIR project."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INTERNATIONAL_DATA_DIR = DATA_DIR / "international"
CHINA_DATA_DIR = DATA_DIR / "china"
SOURCE_DATA_DIR = DATA_DIR / "source"
REFERENCE_RESULTS_DIR = PROJECT_ROOT / "reference_results"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_RESULTS_DIR = OUTPUT_DIR / "results"
OUTPUT_FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_TABLES_DIR = OUTPUT_DIR / "tables"
LOG_DIR = PROJECT_ROOT / "logs"


def ensure_output_dirs() -> None:
    """Create all writable output directories."""
    for path in (OUTPUT_RESULTS_DIR, OUTPUT_FIGURES_DIR, OUTPUT_TABLES_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def resolve_result_file(filename: str, source: str = "reference") -> Path:
    """Resolve a result file from bundled reference output or a fresh run.

    Parameters
    ----------
    filename:
        Basename of a CSV/JSON result file.
    source:
        ``reference`` (default), ``outputs``, or ``auto``. ``auto`` uses a
        freshly generated output when present and otherwise falls back to the
        bundled reference result.
    """
    if source not in {"reference", "outputs", "auto"}:
        raise ValueError("source must be 'reference', 'outputs', or 'auto'")
    ref = REFERENCE_RESULTS_DIR / filename
    out = OUTPUT_RESULTS_DIR / filename
    if source == "reference":
        path = ref
    elif source == "outputs":
        path = out
    else:
        path = out if out.exists() and out.stat().st_size > 0 else ref
    if not path.exists():
        raise FileNotFoundError(
            f"Required result file not found: {path}. Run the corresponding "
            "fit script or use --results-source reference."
        )
    return path


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required project files:\n  " + "\n  ".join(missing))


def output_path(kind: str, filename: str, directory: Optional[Path] = None) -> Path:
    """Return an output path and create its parent directory."""
    ensure_output_dirs()
    if directory is not None:
        base = Path(directory).expanduser().resolve()
    elif kind == "results":
        base = OUTPUT_RESULTS_DIR
    elif kind == "figures":
        base = OUTPUT_FIGURES_DIR
    elif kind == "tables":
        base = OUTPUT_TABLES_DIR
    else:
        raise ValueError("kind must be results, figures, or tables")
    base.mkdir(parents=True, exist_ok=True)
    return base / filename
