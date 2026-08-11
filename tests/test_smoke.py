"""The package and each of its modules import cleanly."""

import importlib

import pytest


@pytest.mark.parametrize(
    "name",
    [
        "acgt",
        "acgt.core",
        "acgt.dataset",
        "acgt.convert",
        "acgt.query",
        "acgt.annotate",
        "acgt.tui",
    ],
)
def test_module_imports(name):
    importlib.import_module(name)
