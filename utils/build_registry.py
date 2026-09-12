"""Build the symbol registry embedded in ``ARCHITECTURE.md``.

The registry is generated from the abstract syntax tree of every tracked source
file rather than maintained by hand, so it cannot drift from the code. Running
this module rewrites the block delimited by the ``REGISTRY`` markers in
``ARCHITECTURE.md`` and reports any symbol name that is defined more than once.

Duplicate detection is the point of the tool. Two independent implementations of
``load_omics`` are what caused the baseline harness and the MOGFormer runs to be
scored on different patient cohorts, so a name collision is treated as a defect
rather than a style issue.

Typical use::

    python -m utils.build_registry              # rewrite ARCHITECTURE.md
    python -m utils.build_registry --check      # exit 1 on drift or duplicates
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHITECTURE_PATH = REPO_ROOT / "ARCHITECTURE.md"

BEGIN_MARKER = "<!-- BEGIN GENERATED REGISTRY -->"
END_MARKER = "<!-- END GENERATED REGISTRY -->"

#: Directories walked for symbols, relative to the repository root.
PACKAGE_ROOTS: tuple[str, ...] = ("src/mogformer", "utils")

#: Scripts not yet folded into the package. Reported so the remaining surface
#: is visible, but excluded from duplicate enforcement because they are
#: standalone and predate the conventions.
LEGACY_SOURCES: tuple[str, ...] = ("scripts",)

#: Names permitted to recur across modules. Dunder methods and a small set of
#: conventional hooks are expected to appear once per class or module.
DUPLICATE_ALLOWLIST: frozenset[str] = frozenset(
    {
        "main",
        "forward",
        "fit",
        "transform",
        "fit_transform",
        "predict",
        "predict_proba",
        "build",
        "run",
        "__init__",
        "__len__",
        "__getitem__",
        "__repr__",
        "__post_init__",
    }
)


@dataclass(frozen=True)
class Symbol:
    """One class, function or method discovered in a source file.

    Attributes:
        module: Dotted module path, e.g. ``mogformer.data.omics``.
        path: Repository-relative POSIX path of the defining file.
        kind: One of ``class``, ``function`` or ``method``.
        name: Bare symbol name.
        owner: Enclosing class name for methods, otherwise the empty string.
        line: 1-indexed line on which the definition starts.
        end_line: 1-indexed line on which the definition ends.
        signature: Parameter list rendered as ``(a, b, *args, **kwargs)``.
        summary: First non-empty line of the docstring, or the empty string.
        is_public: False when the name or its owner begins with an underscore.
    """

    module: str
    path: str
    kind: str
    name: str
    owner: str
    line: int
    end_line: int
    signature: str
    summary: str
    is_public: bool

    @property
    def qualified_name(self) -> str:
        """Return ``Owner.name`` for methods and ``name`` for everything else."""
        return f"{self.owner}.{self.name}" if self.owner else self.name


def _module_path(path: Path) -> str:
    """Convert a file path into a dotted module path.

    Args:
        path: Absolute path to a Python source file.

    Returns:
        Dotted module path relative to the repository root, with the ``.py``
        suffix and any trailing ``__init__`` component removed.
    """
    relative = path.relative_to(REPO_ROOT).with_suffix("")
    parts = list(relative.parts)
    # Under a src-layout the leading directory is not part of the import path.
    if parts and parts[0] == "src":
        parts.pop(0)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _first_docstring_line(
    node: ast.AsyncFunctionDef | ast.ClassDef | ast.FunctionDef | ast.Module,
) -> str:
    """Return the first non-empty line of ``node``'s docstring, or ``""``."""
    docstring = ast.get_docstring(node)
    if not docstring:
        return ""
    for line in docstring.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _render_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Render a function's parameter names as a compact signature string."""
    args = node.args
    names = [param.arg for param in (*args.posonlyargs, *args.args)]
    if args.vararg:
        names.append(f"*{args.vararg.arg}")
    names.extend(param.arg for param in args.kwonlyargs)
    if args.kwarg:
        names.append(f"**{args.kwarg.arg}")
    return f"({', '.join(names)})"


def _iter_source_files(roots: Sequence[str]) -> Iterator[Path]:
    """Yield every Python file under ``roots``, skipping caches and venvs.

    Args:
        roots: Repository-relative directory or file paths.

    Yields:
        Absolute paths to Python source files, in sorted order.
    """
    for root in roots:
        target = REPO_ROOT / root
        if target.is_file() and target.suffix == ".py":
            yield target
            continue
        if not target.is_dir():
            continue
        for path in sorted(target.rglob("*.py")):
            if any(part in {"__pycache__", ".conda", ".git"} for part in path.parts):
                continue
            yield path


def parse_symbols(path: Path) -> list[Symbol]:
    """Extract every top-level and nested definition from one source file.

    Jupyter shell and line magics are neutralised before parsing so that
    aggregated notebook exports remain readable by :mod:`ast`.

    Args:
        path: Absolute path to a Python source file.

    Returns:
        Every class, function and method defined in the file. Returns an empty
        list when the file cannot be parsed.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"^[!%]", "#", text, flags=re.MULTILINE)
    try:
        tree = ast.parse(text)
    except SyntaxError as error:  # pragma: no cover - defensive
        print(f"warning: could not parse {path}: {error}", file=sys.stderr)
        return []

    module = _module_path(path)
    relative = path.relative_to(REPO_ROOT).as_posix()
    symbols: list[Symbol] = []

    def visit(node: ast.AST, owner: str = "") -> None:
        for child in getattr(node, "body", []):
            if isinstance(child, ast.ClassDef):
                symbols.append(
                    Symbol(
                        module=module,
                        path=relative,
                        kind="class",
                        name=child.name,
                        owner=owner,
                        line=child.lineno,
                        end_line=child.end_lineno or child.lineno,
                        signature="",
                        summary=_first_docstring_line(child),
                        is_public=not child.name.startswith("_"),
                    )
                )
                visit(child, child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(
                    Symbol(
                        module=module,
                        path=relative,
                        kind="method" if owner else "function",
                        name=child.name,
                        owner=owner,
                        line=child.lineno,
                        end_line=child.end_lineno or child.lineno,
                        signature=_render_signature(child),
                        summary=_first_docstring_line(child),
                        is_public=not child.name.startswith("_")
                        and not owner.startswith("_"),
                    )
                )

    visit(tree)
    return symbols


def collect(roots: Sequence[str]) -> list[Symbol]:
    """Parse every source file under ``roots`` and return the merged symbols."""
    symbols: list[Symbol] = []
    for path in _iter_source_files(roots):
        symbols.extend(parse_symbols(path))
    return symbols


def find_duplicates(symbols: Iterable[Symbol]) -> dict[str, list[Symbol]]:
    """Group symbols that share a name across different modules.

    Methods are keyed by their bare name deliberately: two classes each owning a
    ``compute_bias`` method is usually fine, but two modules each defining a
    free function ``load_omics`` is the defect this tool exists to catch. Only
    classes and free functions are considered.

    Args:
        symbols: Symbols to inspect.

    Returns:
        Mapping of name to the competing definitions, for names that are defined
        in more than one module and are not in :data:`DUPLICATE_ALLOWLIST`.
    """
    by_name: dict[str, list[Symbol]] = defaultdict(list)
    for symbol in symbols:
        if symbol.kind == "method" or symbol.name in DUPLICATE_ALLOWLIST:
            continue
        by_name[symbol.name].append(symbol)

    duplicates: dict[str, list[Symbol]] = {}
    for name, group in by_name.items():
        if len({symbol.module for symbol in group}) > 1:
            duplicates[name] = sorted(group, key=lambda s: (s.module, s.line))
    return dict(sorted(duplicates.items()))


def _escape(text: str) -> str:
    """Escape pipe characters so a cell cannot break the Markdown table."""
    return text.replace("|", r"\|")


def render_registry(symbols: Sequence[Symbol]) -> str:
    """Render the package symbol registry as Markdown.

    Args:
        symbols: Symbols belonging to the first-party package.

    Returns:
        Markdown text, one table per module, ordered by module path.
    """
    if not symbols:
        return (
            "_No first-party symbols yet. The registry populates as modules land "
            "under `mogformer/`._\n"
        )

    by_module: dict[str, list[Symbol]] = defaultdict(list)
    for symbol in symbols:
        by_module[symbol.module].append(symbol)

    counts = Counter(symbol.module for symbol in symbols)
    lines: list[str] = [
        f"_{len(symbols)} symbols across {len(by_module)} modules._",
        "",
    ]
    for module in sorted(by_module):
        lines.append(f"#### `{module}`")
        lines.append("")
        lines.append("| Symbol | Kind | Lines | Purpose |")
        lines.append("| --- | --- | --- | --- |")
        for symbol in sorted(by_module[module], key=lambda s: s.line):
            span = (
                f"{symbol.line}"
                if symbol.line == symbol.end_line
                else f"{symbol.line}–{symbol.end_line}"
            )
            label = f"`{symbol.qualified_name}{symbol.signature}`"
            summary = _escape(symbol.summary) or "_undocumented_"
            lines.append(f"| {label} | {symbol.kind} | {span} | {summary} |")
        lines.append("")
    lines.append(
        "<sub>Largest modules: "
        + ", ".join(f"`{module}` ({count})" for module, count in counts.most_common(5))
        + "</sub>"
    )
    lines.append("")
    return "\n".join(lines)


def render_migration_table(symbols: Sequence[Symbol]) -> str:
    """Render a per-file count of symbols still awaiting migration."""
    counts = Counter(symbol.path for symbol in symbols)
    if not counts:
        return "_Nothing outstanding — every legacy source has been folded in._\n"
    lines = [
        "| Legacy source | Symbols | Status |",
        "| --- | --- | --- |",
    ]
    for path, count in sorted(counts.items(), key=lambda item: -item[1]):
        lines.append(f"| `{path}` | {count} | not yet migrated |")
    lines.append("")
    return "\n".join(lines)


def build_block() -> str:
    """Assemble the full generated section of ``ARCHITECTURE.md``."""
    package_symbols = collect(PACKAGE_ROOTS)
    legacy_symbols = collect(LEGACY_SOURCES)
    duplicates = find_duplicates(package_symbols)

    parts: list[str] = [
        BEGIN_MARKER,
        "",
        "> Generated by `python -m utils.build_registry`. Do not edit by hand.",
        "",
        "### Package symbols",
        "",
        render_registry(package_symbols),
        "### Duplicate names",
        "",
    ]
    if duplicates:
        parts.append(
            "| Name | Defined in |\n| --- | --- |\n"
            + "\n".join(
                f"| `{name}` | "
                + ", ".join(f"`{s.module}:{s.line}`" for s in group)
                + " |"
                for name, group in duplicates.items()
            )
            + "\n"
        )
    else:
        parts.append("None. Every first-party name resolves to one definition.\n")

    parts.extend(
        [
            "### Legacy sources still in the tree",
            "",
            render_migration_table(legacy_symbols),
            END_MARKER,
        ]
    )
    return "\n".join(parts)


def write_block(check_only: bool = False) -> int:
    """Rewrite the generated block in ``ARCHITECTURE.md``.

    Args:
        check_only: When True, report drift without writing and return a
            non-zero status if the file is stale.

    Returns:
        Process exit status: 0 on success, 1 on drift or duplicate names.
    """
    block = build_block()
    duplicates = find_duplicates(collect(PACKAGE_ROOTS))

    if not ARCHITECTURE_PATH.exists():
        print(f"error: {ARCHITECTURE_PATH} does not exist", file=sys.stderr)
        return 1

    original = ARCHITECTURE_PATH.read_text(encoding="utf-8")
    pattern = re.compile(
        re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER),
        re.DOTALL,
    )
    if not pattern.search(original):
        print(
            f"error: markers {BEGIN_MARKER} / {END_MARKER} not found in "
            f"{ARCHITECTURE_PATH.name}",
            file=sys.stderr,
        )
        return 1

    updated = pattern.sub(lambda _: block, original)
    status = 0

    if duplicates:
        print("Duplicate first-party symbol names:", file=sys.stderr)
        for name, group in duplicates.items():
            locations = ", ".join(f"{s.module}:{s.line}" for s in group)
            print(f"  {name}: {locations}", file=sys.stderr)
        status = 1

    if check_only:
        if updated != original:
            print(
                "ARCHITECTURE.md registry is stale; run "
                "`python -m utils.build_registry`",
                file=sys.stderr,
            )
            status = 1
        return status

    if updated != original:
        ARCHITECTURE_PATH.write_text(updated, encoding="utf-8")
        print(f"updated {ARCHITECTURE_PATH.relative_to(REPO_ROOT)}")
    else:
        print("registry already up to date")
    return status


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m utils.build_registry``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when the registry is stale",
    )
    arguments = parser.parse_args(argv)
    return write_block(check_only=arguments.check)


if __name__ == "__main__":
    raise SystemExit(main())
