"""Launch the Streamlit viewer for a query's outputs.

One app, two launch modes:

- **Local** — opens your browser automatically.
- **SSH / headless** — binds a port on the remote host (auto-detected) and
  prints the ``ssh -L`` port-forward command so you can view it in your
  *local* browser. No data leaves your SSH channel.
"""
import os
import sys
import subprocess
from pathlib import Path

from PaddockTS.viewer.scan import _resolve_target


def _is_ssh() -> bool:
    """True when this process looks like it's running over an SSH session."""
    return bool(os.environ.get('SSH_CONNECTION') or os.environ.get('SSH_TTY'))


def serve(target, *, port: int = 8501, headless: bool | None = None,
          address: str | None = None):
    """Launch the Streamlit viewer pointed at a query's output directory.

    Args:
        target: a :class:`PaddockTS.query.Query` or a path to an output
            directory.
        port: TCP port for the Streamlit server. Default ``8501``.
        headless: If ``None`` (default), auto-detect — headless when running
            over SSH, windowed locally. Pass ``True``/``False`` to force.
        address: Bind address. Defaults to ``localhost`` (safe for SSH
            port-forwarding); only override if you knowingly want to expose
            the server on the network.

    Raises:
        ImportError: if Streamlit isn't installed (``pip install
            'PaddockTS[viewer]'``).
    """
    try:
        import streamlit  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The viewer needs Streamlit. Install it with:\n"
            "    pip install 'PaddockTS[viewer]'\n"
            "  or\n    pip install streamlit"
        ) from e

    out_dir, stub = _resolve_target(target)
    out_dir = str(Path(out_dir))

    if headless is None:
        headless = _is_ssh()
    if address is None:
        address = 'localhost'

    app_path = str(Path(__file__).with_name('app.py'))
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', app_path,
        '--server.port', str(port),
        '--server.address', address,
        '--browser.gatherUsageStats', 'false',
    ]
    if headless:
        cmd += ['--server.headless', 'true']
    cmd += ['--', '--out-dir', out_dir, '--stub', stub or '']

    if headless:
        bar = '-' * 64
        print(bar)
        print('PaddockTS viewer - headless (SSH) mode')
        print(f'  outputs : {out_dir}')
        print(f'  listening on {address}:{port} of the remote host')
        print('  From your LOCAL machine, forward the port:')
        print(f'      ssh -L {port}:localhost:{port} <user>@<this-host>')
        print(f'  then open:  http://localhost:{port}')
        print(bar)

    subprocess.run(cmd, check=False)
