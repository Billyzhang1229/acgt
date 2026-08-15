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
import shutil
import subprocess

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


# --------------------------------------------------------------------------
# parallel scan and fill
# --------------------------------------------------------------------------
# The fixtures are far below the size at which from_vcf picks the parallel
# path on its own, so every test here forces it with an explicit `workers`.
# What the tests hold is that the parallel path writes the same store the
# serial one does; the serial path's agreement with the oracle then carries
# over.


@pytest.mark.parametrize("fixture", ["sample_vcf", "sample_bcf"])
def test_parallel_matches_serial(fixture, request, tmp_path):
    src = request.getfixturevalue(fixture)
    serial = convert.from_vcf(src, tmp_path / "serial.vcz", chunk_size=CHUNK, workers=1)
    parallel = convert.from_vcf(
        src, tmp_path / "parallel.vcz", chunk_size=CHUNK, workers=2
    )
    assert dataset.preflight(parallel, deep=True) == []
    assert_stores_equal(parallel, serial)


def test_parallel_matches_oracle(sample_vcf, tmp_path):
    ours = convert.from_vcf(
        sample_vcf, tmp_path / "ours.vcz", chunk_size=CHUNK, workers=2
    )
    theirs = oracle_store(sample_vcf, tmp_path / "oracle.vcz")
    assert_stores_equal(ours, theirs)
    assert view(ours) == view(theirs)


def test_partition_boundary_counts_a_spanning_deletion_once(
    sample_vcf, tmp_path, monkeypatch
):
    # chr2:150 AACG>A spans 150-153. A region query returns every record
    # overlapping the window, so with a boundary at 150|151 the deletion is
    # handed to both partitions; assigning by POS must keep it in one.
    monkeypatch.setattr(convert, "PARTITION_WINDOW", 150)
    serial = convert.from_vcf(
        sample_vcf, tmp_path / "serial.vcz", chunk_size=CHUNK, workers=1
    )
    parallel = convert.from_vcf(
        sample_vcf, tmp_path / "parallel.vcz", chunk_size=CHUNK, workers=3
    )
    assert dataset.preflight(parallel, deep=True) == []
    assert_stores_equal(parallel, serial)


def test_unindexed_file_falls_back_to_serial(mini_vcf, tmp_path):
    # no .tbi/.csi: nothing to seek by, so workers=4 still runs serially and
    # the store comes out the same
    assert convert._plan_parallel(mini_vcf, 4, ["chr1"], [1000000]) is None
    ours = convert.from_vcf(
        mini_vcf, tmp_path / "ours.vcz", chunk_size=CHUNK, workers=4
    )
    theirs = oracle_store(mini_vcf, tmp_path / "oracle.vcz")
    assert_stores_equal(ours, theirs)


def test_small_files_stay_serial_unless_asked(sample_vcf, sample_bcf):
    contigs = ["chr1", "chr2", "chrM", "chrEmpty"]
    lengths = [1000000, 800000, 16569, 5000]
    # default workers: the fixture is far below PARALLEL_MIN_BYTES
    assert convert._plan_parallel(sample_vcf, None, contigs, lengths) is None
    # explicit workers: the parallel path is planned; the tabix index names
    # only the contigs that have records, in file order, and chrEmpty is
    # never queried; the BCF's CSI names nothing, so the header order is used
    assert [p[0] for p in convert._plan_parallel(sample_vcf, 2, contigs, lengths)] == [
        "chr1",
        "chr2",
        "chrM",
    ]
    assert [
        p[0] for p in convert._plan_parallel(sample_bcf, 2, contigs, lengths)
    ] == contigs


def test_index_contigs_read_from_tbi_and_csi(sample_vcf, sample_bcf):
    assert convert._index_contigs(convert._index_file(sample_vcf)) == [
        "chr1",
        "chr2",
        "chrM",
    ]
    assert convert._index_contigs(convert._index_file(sample_bcf)) is None


def test_partitions_cover_each_contig_to_the_end_of_coordinates():
    parts = convert._partitions(["a", "b", "c"], [250, 100, -1], window=100)
    # windows are inclusive on both ends and adjacent windows do not overlap;
    # the last window of every contig is open-ended so a wrong header length
    # cannot drop records
    assert parts == [
        ("a", 1, 100),
        ("a", 101, 200),
        ("a", 201, convert._HTS_MAX_POS),
        ("b", 1, convert._HTS_MAX_POS),
        ("c", 1, convert._HTS_MAX_POS),
    ]


def test_index_named_contig_converts_the_same_on_both_paths(
    undeclared_contig_vcf, tmp_path
):
    # Once the file is tabix-indexed, htslib adds any contig the index names
    # but the header does not to its in-memory header (with unknown length),
    # so what the plain-text path rejects converts here -- and both of our
    # paths have to see the same augmented header and write the same store.
    bgzip, tabix = shutil.which("bgzip"), shutil.which("tabix")
    if bgzip is None or tabix is None:
        pytest.skip("bgzip/tabix not installed")
    src = tmp_path / "undeclared.vcf.gz"
    with src.open("wb") as fh:
        subprocess.run([bgzip, "-c", undeclared_contig_vcf], stdout=fh, check=True)
    subprocess.run([tabix, "-p", "vcf", src], check=True)
    assert convert._index_contigs(convert._index_file(src)) == ["chr1", "chr9"]
    serial = convert.from_vcf(src, tmp_path / "serial.vcz", chunk_size=CHUNK, workers=1)
    parallel = convert.from_vcf(
        src, tmp_path / "parallel.vcz", chunk_size=CHUNK, workers=2
    )
    assert store_arrays(serial)["contig_id"].tolist() == ["chr1", "chr9"]
    assert_stores_equal(parallel, serial)
    assert_stores_equal(parallel, oracle_store(src, tmp_path / "oracle.vcz"))


def test_fill_refuses_a_count_that_disagrees_with_the_scan(sample_vcf):
    # The scan sizes every partition's slice; a fill that yields a different
    # count would write over the next partition's rows. Drive one worker
    # in-process with an expected count the region cannot satisfy.
    vcf = convert.VCF(sample_vcf)
    fields, filters, _, has_gt, declared = convert._discover(vcf, ())
    contigs = list(vcf.seqnames)
    n, max_alt, observed, slen, idlen, ploidy = convert._scan(
        sample_vcf, fields, has_gt, declared
    )
    n_alleles = max_alt + 1
    convert._resolve_widths(fields, n_alleles, 6, observed)
    contig_index = {c: i for i, c in enumerate(contigs)}
    filter_index = {f: i for i, f in enumerate(filters)}
    shared = convert._SharedBuffers()
    try:
        convert._allocate(
            n, n_alleles, ploidy, len(filters), len(contigs), fields, slen, idlen,
            48, has_gt, empty=shared.empty,
        )  # fmt: skip
        convert._worker_init(
            sample_vcf, fields, has_gt, declared, contig_index, filter_index, 48
        )
        widths = [f.width for f in fields]
        part = ("chr1", 1, convert._HTS_MAX_POS)  # holds 5 records
        job = (part, 0, 4, shared.meta, widths, n_alleles, ploidy)
        with pytest.raises(ValueError, match="changed underneath"):
            convert._fill_partition(job)
    finally:
        convert._worker.clear()
        shared.close()
