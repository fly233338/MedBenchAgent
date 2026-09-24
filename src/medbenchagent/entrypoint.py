"""Select the terminal interface without coupling CLI and TUI modules."""

from __future__ import annotations

from medbenchagent.cli import _parser, run_command
from medbenchagent.runtime.graph_runner import enforce_offline_mode


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.config and not args.config.is_file():
        parser.error(f"Configuration template does not exist: {args.config}")
    enforce_offline_mode()
    if args.command is None:
        from medbenchagent.tui import MBAApp

        MBAApp(config_template=args.config.resolve() if args.config else None).run()
        return
    raise SystemExit(run_command(args))
