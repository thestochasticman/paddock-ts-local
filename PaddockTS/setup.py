"""Post-install setup for PaddockTS."""

import subprocess
import sys
from pathlib import Path


def run_setup():
    """Initialize git submodules after installation."""
    # Find the repo root (where .git is)
    current = Path(__file__).parent.parent

    if not (current / '.git').exists():
        print("Not in a git repository, skipping submodule init")
        return

    print("Initializing git submodules...")
    try:
        subprocess.run(
            ['git', 'submodule', 'update', '--init', '--recursive'],
            cwd=current,
            check=True,
        )
        print("Submodules initialized successfully")
    except subprocess.CalledProcessError as e:
        print(f"Failed to initialize submodules: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    run_setup()
