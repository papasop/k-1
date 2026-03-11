#!/usr/bin/env python3
"""
cli.py – Command-line interface for the ChatGPT Codex Connector.

Usage examples
--------------
  python cli.py generate "a Python function that reverses a string"
  python cli.py complete --language javascript --file partial.js
  python cli.py explain --file my_script.py
  python cli.py fix     --file buggy.py
  python cli.py optimize --file slow.py --goal "memory usage"
"""

import argparse
import sys
import os
import logging

from codex_connector import CodexConnector, Config, setup_logging


def _read_source(args: argparse.Namespace) -> str:
    """Return code from --file or --code, or read from stdin."""
    if getattr(args, "file", None):
        with open(args.file, "r", encoding="utf-8") as fh:
            return fh.read()
    if getattr(args, "code", None):
        return args.code
    if not sys.stdin.isatty():
        return sys.stdin.read()
    print("Error: provide --file, --code, or pipe code via stdin.", file=sys.stderr)
    sys.exit(1)


def _build_connector(args: argparse.Namespace) -> CodexConnector:
    config = Config(
        api_key=getattr(args, "api_key", None) or os.environ.get("OPENAI_API_KEY"),
        model=getattr(args, "model", None),
        max_tokens=getattr(args, "max_tokens", None),
        temperature=getattr(args, "temperature", None),
    )
    return CodexConnector(config=config)


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def cmd_generate(args: argparse.Namespace) -> None:
    connector = _build_connector(args)
    result = connector.generate(args.description, language=args.language)
    print(result)


def cmd_complete(args: argparse.Namespace) -> None:
    connector = _build_connector(args)
    code = _read_source(args)
    result = connector.complete(code, language=args.language)
    print(result)


def cmd_explain(args: argparse.Namespace) -> None:
    connector = _build_connector(args)
    code = _read_source(args)
    result = connector.explain(code, language=args.language)
    print(result)


def cmd_fix(args: argparse.Namespace) -> None:
    connector = _build_connector(args)
    code = _read_source(args)
    result = connector.fix_bugs(code, language=args.language)
    print(result)


def cmd_optimize(args: argparse.Namespace) -> None:
    connector = _build_connector(args)
    code = _read_source(args)
    goal = getattr(args, "goal", "performance and readability")
    result = connector.optimize(code, language=args.language, goal=goal)
    print(result)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--language", "-l", default="Python",
        help="Programming language (default: Python)",
    )
    parser.add_argument(
        "--model", default=None,
        help="OpenAI model to use (overrides CODEX_MODEL env var)",
    )
    parser.add_argument(
        "--api-key", dest="api_key", default=None,
        help="OpenAI API key (overrides OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--max-tokens", dest="max_tokens", type=int, default=None,
        help="Maximum tokens in the response",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature (0–2)",
    )


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", "-f", help="Path to source file")
    group.add_argument("--code", "-c", help="Inline code string")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codex",
        description="ChatGPT Codex Connector – AI-powered code tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING)",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # generate
    p_gen = sub.add_parser("generate", help="Generate code from a description")
    p_gen.add_argument("description", help="Natural-language task description")
    _add_common_args(p_gen)
    p_gen.set_defaults(func=cmd_generate)

    # complete
    p_cmp = sub.add_parser("complete", help="Complete an incomplete code snippet")
    _add_source_args(p_cmp)
    _add_common_args(p_cmp)
    p_cmp.set_defaults(func=cmd_complete)

    # explain
    p_exp = sub.add_parser("explain", help="Explain what code does")
    _add_source_args(p_exp)
    _add_common_args(p_exp)
    p_exp.set_defaults(func=cmd_explain)

    # fix
    p_fix = sub.add_parser("fix", help="Identify and fix bugs in code")
    _add_source_args(p_fix)
    _add_common_args(p_fix)
    p_fix.set_defaults(func=cmd_fix)

    # optimize
    p_opt = sub.add_parser("optimize", help="Optimize code for a given goal")
    _add_source_args(p_opt)
    _add_common_args(p_opt)
    p_opt.add_argument(
        "--goal", default="performance and readability",
        help="Optimization objective (default: 'performance and readability')",
    )
    p_opt.set_defaults(func=cmd_optimize)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)
    args.func(args)


if __name__ == "__main__":
    main()
