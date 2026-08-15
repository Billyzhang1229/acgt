"""from_vcf against the reference implementations.

Three independent referees, per AGENTS.md. bio2zarr converts the same
fixture and every array must come out equal, bit for bit where floats
carry sentinel NaNs. vcztools must render both stores to identical VCF
text. bio2zarr's store-vs-VCF verifier must accept a store we wrote — on
the uniformly diploid fixture only until the release after 0.2.1, which
carries the mixed-ploidy fix already merged upstream. All three are dev
dependencies; the application never sees them.
"""

import io
import logging

import bio2zarr.vcf as b2z
import numpy as np
import pytest
import vcztools
import zarr
from bio2zarr import vcz_verification

from acgt import convert, core, dataset

BOTH_FORMATS = pytest.mark.parametrize("zarr_format", [2, 3])

# Explicit on both sides: region_index is chunk-dependent, so the stores
# are only comparable when their variants chunking agrees.
CHUNK = 1000

LONG_ALT = "T" + "ACGT" * 15  # fixture strings past the inline cap
LONG_STR = "ABCDEFGHIJ" * 6


def oracle_store(vcf, out, zarr_format=2):
    b2z.convert(
        [str(vcf)], str(out), variants_chunk_size=CHUNK, zarr_format=zarr_format
    )
    return out


def store_arrays(path):
    """Every array in the store, materialized and typed for comparison."""
    root = zarr.open_group(path, mode="r")
    out = {}
    for k in root.array_keys():
        node = root[k]
        assert isinstance(node, zarr.Array)
        out[k] = np.asarray(node[:])
    return out


def assert_stores_equal(ours, theirs):
    za = store_arrays(ours)
    zb = store_arrays(theirs)
    assert set(za) == set(zb)
    for k in sorted(za):
        x, y = za[k], zb[k]
        assert x.shape == y.shape, k
        if x.dtype.kind in "OUT" or y.dtype.kind in "OUT":
            same = np.array_equal(
                np.asarray(x, dtype=object), np.asarray(y, dtype=object)
            )
        elif x.dtype.kind == "f":
            # missing and fill are distinct NaN payloads: compare bits
            same = x.dtype == y.dtype and np.array_equal(
                x.view(f"i{x.dtype.itemsize}"), y.view(f"i{y.dtype.itemsize}")
            )
        elif x.dtype.kind in "iu":
            # widths may differ (independent downcasting); values may not
            same = np.array_equal(x.astype(np.int64), y.astype(np.int64))
        else:
            same = np.array_equal(x, y)
        assert same, f"{k} differs from the reference store"


def view(store):
    """The store as VCF text through vcztools, header included.

    Header lines come from array attributes (description) and the group's
    vcf_meta_information; comparing them too holds the round trip to more
    than the record bodies. The two lines that name the writing tool and
    the render time are the only ones that legitimately differ.
    """
    reader = vcztools.VczReader(vcztools.open_zarr(store))
    buf = io.StringIO()
    vcztools.write_vcf(reader, buf)
    return [
        line
        for line in buf.getvalue().splitlines()
        if not line.startswith(("##source=", "##vcztools_viewCommand="))
    ]


@pytest.fixture(scope="module")
def sample_store(sample_vcf, tmp_path_factory):
    out = tmp_path_factory.mktemp("stores") / "sample.vcz"
    return convert.from_vcf(sample_vcf, out, chunk_size=CHUNK)


@BOTH_FORMATS
def test_sample_matches_oracle(sample_vcf, tmp_path, zarr_format):
    ours = convert.from_vcf(
        sample_vcf, tmp_path / "ours.vcz", zarr_format=zarr_format, chunk_size=CHUNK
    )
    theirs = oracle_store(sample_vcf, tmp_path / "oracle.vcz", zarr_format)
    assert dataset.preflight(ours, deep=True) == []
    assert dataset.preflight(theirs, deep=True) == []
    assert_stores_equal(ours, theirs)


@BOTH_FORMATS
def test_mini_matches_oracle(mini_vcf, tmp_path, zarr_format):
    ours = convert.from_vcf(
        mini_vcf, tmp_path / "ours.vcz", zarr_format=zarr_format, chunk_size=CHUNK
    )
    theirs = oracle_store(mini_vcf, tmp_path / "oracle.vcz", zarr_format)
    assert dataset.preflight(ours, deep=True) == []
    assert_stores_equal(ours, theirs)


def test_bcf_matches_oracle(sample_bcf, tmp_path):
    ours = convert.from_vcf(sample_bcf, tmp_path / "ours.vcz", chunk_size=CHUNK)
    theirs = oracle_store(sample_bcf, tmp_path / "oracle.vcz")
    assert dataset.preflight(ours, deep=True) == []
    assert_stores_equal(ours, theirs)


def test_reference_verifier_accepts_our_store(mini_vcf, tmp_path):
    store = convert.from_vcf(mini_vcf, tmp_path / "mini.vcz", chunk_size=CHUNK)
    vcz_verification.verify(str(mini_vcf), str(store))


@BOTH_FORMATS
def test_vcztools_renders_both_stores_identically(sample_vcf, tmp_path, zarr_format):
    ours = convert.from_vcf(
        sample_vcf, tmp_path / "ours.vcz", zarr_format=zarr_format, chunk_size=CHUNK
    )
    theirs = oracle_store(sample_vcf, tmp_path / "oracle.vcz", zarr_format)
    assert view(ours) == view(theirs)


def test_multisample_is_rejected(multisample_vcf, tmp_path):
    with pytest.raises(ValueError, match="single-sample"):
        convert.from_vcf(multisample_vcf, tmp_path / "out.vcz")


def test_triploid_matches_oracle(triploid_vcf, tmp_path):
    # a Number=G field at ploidy 3: the genotype count is the multiset
    # coefficient, which the diploid formula undercounts
    ours = convert.from_vcf(triploid_vcf, tmp_path / "ours.vcz", chunk_size=CHUNK)
    theirs = oracle_store(triploid_vcf, tmp_path / "oracle.vcz")
    assert dataset.preflight(ours, deep=True) == []
    assert_stores_equal(ours, theirs)


def test_empty_vcf_makes_a_valid_store(empty_vcf, tmp_path):
    # bio2zarr cannot convert a zero-record file, so there is no oracle
    # here; preflight and the reader carry the check instead
    store = convert.from_vcf(empty_vcf, tmp_path / "empty.vcz")
    assert dataset.preflight(store) == []
    ds = dataset.open_dataset(store)
    assert ds.sizes[core.DIM_VARIANTS] == 0


def test_undeclared_filter_is_rejected(undeclared_filter_vcf, tmp_path):
    with pytest.raises(ValueError, match="q10"):
        convert.from_vcf(undeclared_filter_vcf, tmp_path / "out.vcz")


def test_undeclared_contig_is_rejected(undeclared_contig_vcf, tmp_path):
    with pytest.raises(ValueError, match="chr9"):
        convert.from_vcf(undeclared_contig_vcf, tmp_path / "out.vcz")


def test_partial_gt_matches_oracle(partial_gt_vcf, tmp_path):
    ours = convert.from_vcf(partial_gt_vcf, tmp_path / "ours.vcz", chunk_size=CHUNK)
    theirs = oracle_store(partial_gt_vcf, tmp_path / "oracle.vcz")
    assert dataset.preflight(ours, deep=True) == []
    assert_stores_equal(ours, theirs)


def test_pl_without_gt_keeps_every_value(pl_without_gt_vcf, tmp_path):
    # GT declared but absent from the record: the formula's ploidy is not
    # the ploidy PL was written for. The reference errors here; we widen.
    store = convert.from_vcf(pl_without_gt_vcf, tmp_path / "ours.vcz", chunk_size=CHUNK)
    arrays = store_arrays(store)
    assert arrays["call_PL"].tolist() == [[[0, 10, 20]]]
    assert arrays["call_genotype"].tolist() == [[[-1]]]
    assert dataset.preflight(store, deep=True) == []


def test_no_gt_file_omits_genotype_arrays(no_gt_vcf, tmp_path):
    ours = convert.from_vcf(no_gt_vcf, tmp_path / "ours.vcz", chunk_size=CHUNK)
    theirs = oracle_store(no_gt_vcf, tmp_path / "oracle.vcz")
    root = zarr.open_group(ours, mode="r")
    assert "call_genotype" not in root
    # without GT the Number=G width comes from the observed values, not
    # from a ploidy formula there is no ploidy for
    assert store_arrays(ours)["call_PL"].shape == (2, 1, 3)
    assert dataset.preflight(ours, deep=True) == []
    assert_stores_equal(ours, theirs)


def test_unsorted_positions_rejected(unsorted_vcf, tmp_path):
    with pytest.raises(ValueError, match="sort the file"):
        convert.from_vcf(unsorted_vcf, tmp_path / "out.vcz")


def test_noncontiguous_contig_rejected(interleaved_vcf, tmp_path):
    with pytest.raises(ValueError, match="contiguous"):
        convert.from_vcf(interleaved_vcf, tmp_path / "out.vcz")


def test_existing_output_refused_up_front(mini_vcf, tmp_path):
    out = convert.from_vcf(mini_vcf, tmp_path / "m.vcz")
    with pytest.raises(ValueError, match="already exists"):
        convert.from_vcf(mini_vcf, out)
    convert.from_vcf(mini_vcf, out, overwrite=True)
    assert dataset.preflight(out) == []


def test_dot_call_and_star_allele_match_oracle(dot_and_star_vcf, tmp_path):
    # a whole-sample "." call and the * spanning-deletion allele
    ours = convert.from_vcf(dot_and_star_vcf, tmp_path / "ours.vcz", chunk_size=CHUNK)
    theirs = oracle_store(dot_and_star_vcf, tmp_path / "oracle.vcz")
    assert dataset.preflight(ours, deep=True) == []
    assert_stores_equal(ours, theirs)


def test_unknown_contig_length_omits_the_array(no_contig_length_vcf, tmp_path):
    ours = convert.from_vcf(
        no_contig_length_vcf, tmp_path / "ours.vcz", chunk_size=CHUNK
    )
    theirs = oracle_store(no_contig_length_vcf, tmp_path / "oracle.vcz")
    root = zarr.open_group(ours, mode="r")
    assert "contig_length" not in root  # cyvcf2 reports unknown as -1
    assert dataset.preflight(ours, deep=True) == []
    assert_stores_equal(ours, theirs)


def test_headerless_contigs_rejected_up_front(no_contig_header_vcf, tmp_path):
    with pytest.raises(ValueError, match="declares no contigs"):
        convert.from_vcf(no_contig_header_vcf, tmp_path / "out.vcz")


def test_undeclared_fields_warn_and_drop(undeclared_info_vcf, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="acgt.convert"):
        ours = convert.from_vcf(
            undeclared_info_vcf, tmp_path / "ours.vcz", chunk_size=CHUNK
        )
    messages = " ".join(record.message for record in caplog.records)
    assert "INFO/XX" in messages
    # FORMAT/DP is undeclared even though INFO/DP is: separate namespaces
    assert "FORMAT/DP" in messages
    theirs = oracle_store(undeclared_info_vcf, tmp_path / "oracle.vcz")
    root = zarr.open_group(ours, mode="r")
    assert "variant_XX" not in root  # dropped, matching the reference
    assert "call_DP" not in root
    assert_stores_equal(ours, theirs)


def test_field_descriptions_reach_the_store(sample_store):
    root = zarr.open_group(sample_store, mode="r")
    assert root["variant_DP"].attrs["description"] == "Total depth"
    assert root["call_FC"].attrs["description"] == "One character per call"


def test_character_fields_are_u1(sample_store):
    root = zarr.open_group(sample_store, mode="r")
    node = root["variant_CH"]
    assert isinstance(node, zarr.Array)
    assert node.dtype == np.dtype("U1")
    assert store_arrays(sample_store)["variant_CH"].tolist()[:2] == ["x", "."]


def test_exclude_drops_only_the_named_field(mini_vcf, tmp_path):
    store = convert.from_vcf(
        mini_vcf, tmp_path / "m.vcz", exclude=("FORMAT/DP",), chunk_size=CHUNK
    )
    root = zarr.open_group(store, mode="r")
    assert "call_DP" not in root
    assert "variant_DP" in root


def test_long_strings_spill_intact(sample_store):
    ds = dataset.open_dataset(sample_store)
    assert str(ds["variant_allele"].values[3, 1]) == LONG_ALT
    assert str(ds["variant_LONGS"].values[3]) == LONG_STR


def test_haploid_calls_pad_with_fill(sample_store):
    ds = dataset.open_dataset(sample_store)
    genotype = ds["call_genotype"].values
    assert genotype[9].tolist() == [[1, core.INT_FILL]]
    assert genotype[10].tolist() == [[0, core.INT_FILL]]
    assert ds["call_genotype_mask"].values[9].tolist() == [[False, True]]


def test_dot_filter_sets_nothing(sample_store):
    ds = dataset.open_dataset(sample_store)
    variant_filter = ds["variant_filter"].values
    assert not variant_filter[2].any()  # FILTER "." is not PASS
    assert variant_filter[0].tolist() == [True, False, False]
    assert variant_filter[4].tolist() == [False, True, True]  # q10;s50


def test_stored_schema_carries_provenance(sample_store):
    root = zarr.open_group(sample_store, mode="r")
    schema = core.Schema.fromdict(root.attrs[core.SCHEMA_ATTR])
    assert schema.dims[core.DIM_VARIANTS] == 11
    assert schema.source == "sample.vcf.gz"
    assert schema.build == "GRCh38"
    assert schema.dims[core.DIM_PLOIDY] == 2
