"""The ground rule of the week, enforced. This test is PROVIDED and already passes.

`scipy` and `scikit-learn` are oracles: tests, benchmarks and notebooks may import them
to check your work. The package itself may import `numpy` and nothing else — if scipy
could be called from `src/optlab/`, you would not be writing the algorithms.

Parsing the source with `ast` rather than grepping means a mention inside a string or a
comment is not a false positive.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "optlab"

FORBIDDEN = {"scipy", "sklearn", "matplotlib", "pandas", "torch", "jax", "autograd"}


def _imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=lambda p: str(p.name))
def test_no_oracle_import(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = _imported_roots(tree) & FORBIDDEN
    assert not offenders, (
        f"{path.relative_to(SRC.parent.parent)} imports {sorted(offenders)}. "
        "Only numpy is allowed inside src/optlab/ — oracles belong in tests/."
    )
