"""from_vcf: the bio2zarr wrapper.

The converter is the reference implementation, so these tests do not
re-verify VCF semantics. They hold that the wrapper hands the file to it with
the parameters asked for, that the result passes preflight and the reference
tools, and that the one rule the wrapper adds -- one sample -- holds.
"""

import io

import pytest
import vcztools
import zarr
from bio2zarr import vcz_verification

from acgt import convert, dataset

CHUNK = 1000


def test_sample_converts(sample_vcf, tmp_path):
    out = convert.from_vcf(sample_vcf, tmp_path / "s.vcz", chunk_size=CHUNK)
    assert dataset.preflight(out) == []
    ds = dataset.open_dataset(out)
    assert ds.sizes["variants"] == 11


def test_bcf_converts(sample_bcf, tmp_path):
    out = convert.from_vcf(sample_bcf, tmp_path / "bcf.vcz", chunk_size=CHUNK)
    assert dataset.preflight(out) == []
    ds = dataset.open_dataset(out)
    assert ds.sizes["variants"] == 11


def test_chunk_size_is_honoured(mini_vcf, tmp_path):
    out = convert.from_vcf(mini_vcf, tmp_path / "m.vcz", chunk_size=2)
    root = zarr.open_group(out, mode="r")
    position = root["variant_position"]
    assert isinstance(position, zarr.Array)
    assert position.chunks == (2,)
    assert dataset.preflight(out) == []


def test_reference_verifier_accepts_the_store(mini_vcf, tmp_path):
    store = convert.from_vcf(mini_vcf, tmp_path / "mini.vcz", chunk_size=CHUNK)
    vcz_verification.verify(str(mini_vcf), str(store))


def test_vcztools_renders_the_store_back(sample_vcf, tmp_path):
    store = convert.from_vcf(sample_vcf, tmp_path / "s.vcz", chunk_size=CHUNK)
    buf = io.StringIO()
    vcztools.write_vcf(vcztools.VczReader(vcztools.open_zarr(store)), buf)
    lines = [line for line in buf.getvalue().splitlines() if not line.startswith("##")]
    assert lines[0].startswith("#CHROM")
    assert len(lines) == 1 + 11


def test_multisample_is_rejected(multisample_vcf, tmp_path):
    with pytest.raises(ValueError, match="single-sample"):
        convert.from_vcf(multisample_vcf, tmp_path / "out.vcz")
    assert not (tmp_path / "out.vcz").exists()


def test_existing_target_is_bio2zarrs_to_refuse(mini_vcf, tmp_path):
    out = convert.from_vcf(mini_vcf, tmp_path / "t.vcz", chunk_size=CHUNK)
    with pytest.raises(ValueError, match="already exists"):
        convert.from_vcf(mini_vcf, out, chunk_size=CHUNK)
    assert dataset.preflight(out) == []
