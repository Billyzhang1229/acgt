"""Shared paths to the synthetic fixtures.

Everything under tests/data is invented — see the one rule in AGENTS.md. A
missing fixture skips rather than fails; tests/data/make_fixtures.py
regenerates the lot (needs htslib's bgzip and tabix).
"""

from pathlib import Path

import pytest

_DATA = Path(__file__).parent / "data"


def _fixture(name):
    path = _DATA / name
    if not path.exists():
        pytest.skip(f"{name} missing; run tests/data/make_fixtures.py")
    return path


@pytest.fixture(scope="session")
def sample_vcf():
    """Multi-contig, multiallelic, mixed-ploidy — the awkward one."""
    return _fixture("sample.vcf.gz")


@pytest.fixture(scope="session")
def sample_bcf():
    return _fixture("sample.bcf")


@pytest.fixture(scope="session")
def mini_vcf():
    """Plain-text, single-contig, uniformly diploid."""
    return _fixture("mini.vcf")


@pytest.fixture(scope="session")
def multisample_vcf():
    return _fixture("multisample.vcf")
