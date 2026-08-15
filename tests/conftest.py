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


@pytest.fixture(scope="session")
def triploid_vcf():
    return _fixture("triploid.vcf")


@pytest.fixture(scope="session")
def empty_vcf():
    return _fixture("empty.vcf")


@pytest.fixture(scope="session")
def undeclared_filter_vcf():
    return _fixture("undeclared_filter.vcf")


@pytest.fixture(scope="session")
def undeclared_contig_vcf():
    return _fixture("undeclared_contig.vcf")


@pytest.fixture(scope="session")
def partial_gt_vcf():
    return _fixture("partial_gt.vcf")


@pytest.fixture(scope="session")
def no_gt_vcf():
    return _fixture("no_gt.vcf")


@pytest.fixture(scope="session")
def pl_without_gt_vcf():
    return _fixture("pl_without_gt.vcf")


@pytest.fixture(scope="session")
def unsorted_vcf():
    return _fixture("unsorted.vcf")


@pytest.fixture(scope="session")
def interleaved_vcf():
    return _fixture("interleaved.vcf")


@pytest.fixture(scope="session")
def dot_and_star_vcf():
    return _fixture("dot_and_star.vcf")


@pytest.fixture(scope="session")
def no_contig_header_vcf():
    return _fixture("no_contig_header.vcf")


@pytest.fixture(scope="session")
def no_contig_length_vcf():
    return _fixture("no_contig_length.vcf")


@pytest.fixture(scope="session")
def undeclared_info_vcf():
    return _fixture("undeclared_info.vcf")
