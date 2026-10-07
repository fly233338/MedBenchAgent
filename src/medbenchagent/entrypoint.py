"""Select the terminal interface without coupling CLI and TUI modules."""

from __future__ import annotations

import sys

from medbenchagent.cli import _parser, run_command


def _environment_config(args):
    if args.command in {"register"}:
        return None
    if args.command in {"run", "resume"}:
        from medbenchagent.runtime.sessions import config_path

        return config_path(args.experiment)
    return args.config


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.config and not args.config.is_file():
        parser.error(f"Configuration template does not exist: {args.config}")
    if args.command not in {"register"}:
        from medbenchagent.runtime.environment import ensure_environment

        ensure_environment(_environment_config(args))
    from medbenchagent.runtime.graph_runner import enforce_offline_mode

    enforce_offline_mode()
    if args.command is None:
        from medbenchagent.tui import MBAApp

        MBAApp(config_template=args.config.resolve() if args.config else None).run()
        return
    try:
        status = run_command(args)
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted. Use resume for saved work.", file=sys.stderr)
        status = 130
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}\nCheck the experiment/configuration and command arguments (--help).", file=sys.stderr)
        status = 2
    except RuntimeError as exc:
        print(f"Execution failed: {exc}", file=sys.stderr)
        status = 1
    raise SystemExit(status)
