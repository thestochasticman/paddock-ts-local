"""
PaddockTS configuration management.

Config is loaded from (in order of priority):
1. Environment variables (PADDOCKTS_OUT_DIR, PADDOCKTS_TMP_DIR, PADDOCKTS_SILO_DIR)
2. Config file (~/.paddockts.yaml or ~/.config/paddockts/config.yaml)
3. Built-in defaults

Usage:
    from PaddockTS.config import config

    # Access config values
    print(config.out_dir)
    print(config.tmp_dir)
    print(config.silo_dir)

    # Update config at runtime
    config.out_dir = "/new/path"

    # Save config to file
    config.save()
"""
import os
from pathlib import Path
from typing import Optional

# Try to import yaml, fall back to json if not available
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    import json


# Default paths
DEFAULT_OUT_DIR = os.path.expanduser("~/Documents/PaddockTSLocal")
DEFAULT_TMP_DIR = os.path.expanduser("~/Downloads/PaddockTSLocal")
DEFAULT_SILO_DIR = None  # Will default to {tmp_dir}/SILO if not set

# Config file locations (in order of priority)
CONFIG_PATHS = [
    Path.home() / ".paddockts.yaml",
    Path.home() / ".paddockts.json",
    Path.home() / ".config" / "paddockts" / "config.yaml",
    Path.home() / ".config" / "paddockts" / "config.json",
]


class PaddockTSConfig:
    """PaddockTS configuration singleton."""

    _instance: Optional["PaddockTSConfig"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Set defaults
        self._out_dir = DEFAULT_OUT_DIR
        self._tmp_dir = DEFAULT_TMP_DIR
        self._silo_dir = DEFAULT_SILO_DIR
        self._config_path: Optional[Path] = None

        # Load from file and environment
        self._load_config()

    def _load_config(self):
        """Load config from file and environment variables."""
        # First, try to load from config file
        for path in CONFIG_PATHS:
            if path.exists():
                self._load_from_file(path)
                self._config_path = path
                break

        # Then override with environment variables (higher priority)
        if "PADDOCKTS_OUT_DIR" in os.environ:
            self._out_dir = os.path.expanduser(os.environ["PADDOCKTS_OUT_DIR"])
        if "PADDOCKTS_TMP_DIR" in os.environ:
            self._tmp_dir = os.path.expanduser(os.environ["PADDOCKTS_TMP_DIR"])
        if "PADDOCKTS_SILO_DIR" in os.environ:
            self._silo_dir = os.path.expanduser(os.environ["PADDOCKTS_SILO_DIR"])

    def _load_from_file(self, path: Path):
        """Load config from a YAML or JSON file."""
        try:
            with open(path) as f:
                if path.suffix in (".yaml", ".yml") and HAS_YAML:
                    data = yaml.safe_load(f) or {}
                else:
                    data = json.load(f)

            if "out_dir" in data:
                self._out_dir = os.path.expanduser(data["out_dir"])
            if "tmp_dir" in data:
                self._tmp_dir = os.path.expanduser(data["tmp_dir"])
            if "silo_dir" in data:
                self._silo_dir = os.path.expanduser(data["silo_dir"])

        except Exception as e:
            print(f"Warning: Failed to load config from {path}: {e}")

    @property
    def out_dir(self) -> str:
        """Output directory for results."""
        return self._out_dir

    @out_dir.setter
    def out_dir(self, value: str):
        self._out_dir = os.path.expanduser(value)

    @property
    def tmp_dir(self) -> str:
        """Temporary directory for intermediate files."""
        return self._tmp_dir

    @tmp_dir.setter
    def tmp_dir(self, value: str):
        self._tmp_dir = os.path.expanduser(value)

    @property
    def silo_dir(self) -> str:
        """SILO cache directory. Defaults to {tmp_dir}/SILO."""
        if self._silo_dir is None:
            return os.path.join(self._tmp_dir, "SILO")
        return self._silo_dir

    @silo_dir.setter
    def silo_dir(self, value: str):
        self._silo_dir = os.path.expanduser(value) if value else None

    def save(self, path: Optional[Path] = None):
        """Save current config to file.

        Parameters
        ----------
        path : Path, optional
            Path to save config. Defaults to ~/.paddockts.yaml
        """
        if path is None:
            path = self._config_path or CONFIG_PATHS[0]

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "out_dir": self._out_dir,
            "tmp_dir": self._tmp_dir,
        }
        if self._silo_dir is not None:
            data["silo_dir"] = self._silo_dir

        with open(path, "w") as f:
            if path.suffix in (".yaml", ".yml") and HAS_YAML:
                yaml.dump(data, f, default_flow_style=False)
            else:
                json.dump(data, f, indent=2)

        self._config_path = path
        print(f"Config saved to: {path}")

    def to_dict(self) -> dict:
        """Return config as dictionary."""
        return {
            "out_dir": self.out_dir,
            "tmp_dir": self.tmp_dir,
            "silo_dir": self.silo_dir,
        }

    def __repr__(self) -> str:
        return f"PaddockTSConfig(out_dir={self.out_dir!r}, tmp_dir={self.tmp_dir!r}, silo_dir={self.silo_dir!r})"


# Singleton instance
config = PaddockTSConfig()


def init_config(out_dir: str = None, tmp_dir: str = None, silo_dir: str = None, save: bool = False):
    """Initialize config with custom values.

    Parameters
    ----------
    out_dir : str, optional
        Output directory for results
    tmp_dir : str, optional
        Temporary directory for intermediate files
    silo_dir : str, optional
        SILO cache directory
    save : bool
        If True, save config to file
    """
    if out_dir:
        config.out_dir = out_dir
    if tmp_dir:
        config.tmp_dir = tmp_dir
    if silo_dir:
        config.silo_dir = silo_dir

    if save:
        config.save()

    return config
