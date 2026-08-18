"""open_dataset and preflight on stores the reference converter wrote.

Nothing here writes a store by hand: bio2zarr converts the synthetic
fixtures, and each test either opens the result or corrupts a copy of it in
one specific way and checks that preflight names the problem. Both Zarr
storage formats are covered, since reading either is the module's job.
"""

import json
import shutil

import bio2zarr.vcf as b2z
import numpy as np
import pytest
import zarr

from acgt import core, dataset

CHUNK = 1000
BOTH_FORMATS = pytest.mark.parametrize("zarr_format", [2, 3])


@pytest.fixture(scope="session")
def reference_stores(sample_vcf, tmp_path_factory):
    """One store per storage format from the rich fixture, converted once."""
    root = tmp_path_factory.mktemp("stores")
    stores = {}
    for zarr_format in (2, 3):
        out = root / f"sample_v{zarr_format}.vcz"
        b2z.convert(
            [str(sample_vcf)],
            str(out),
            variants_chunk_size=CHUNK,
            zarr_format=zarr_format,
        )
        stores[zarr_format] = out
    return stores


@pytest.fixture
def store(reference_stores, tmp_path):
    """A private copy of the format-2 store, safe to corrupt."""
    out = tmp_path / "sample.vcz"
    shutil.copytree(reference_stores[2], out)
    return out


def corrupt(out, change):
    """Apply `change` to the store, opened writable."""
    change(zarr.open_group(out, mode="r+"))


@BOTH_FORMATS
def test_reference_store_passes_and_opens(reference_stores, zarr_format):
    out = reference_stores[zarr_format]
    assert dataset.preflight(out) == []
    root = zarr.open_group(out, mode="r")
    assert root.metadata.zarr_format == zarr_format
    assert root.attrs["vcf_zarr_version"] == core.VCF_ZARR_VERSION

    ds = dataset.open_dataset(out)
    assert ds.sizes[core.DIM_SAMPLES] == 1
    assert ds["variant_position"].values[:4].tolist() == [100, 200, 300, 400]
    assert str(ds["variant_id"].values[3]) == "rs4"
    assert str(ds["variant_LONGS"].values[3]) == "ABCDEFGHIJ" * 6
    quality_bits = ds["variant_quality"].values.view("i4")
    missing_bits = np.array([core.FLOAT32_MISSING], "f4").view("i4")[0]
    assert missing_bits in quality_bits  # a "." QUAL survives bit for bit


@BOTH_FORMATS
def test_dimension_names_readable_either_way(reference_stores, zarr_format):
    root = zarr.open_group(reference_stores[zarr_format], mode="r")
    assert dataset.array_dims(root["call_genotype"]) == [
        core.DIM_VARIANTS,
        core.DIM_SAMPLES,
        core.DIM_PLOIDY,
    ]
    if zarr_format == 3:
        # format 3 carries names in array metadata, not the xarray attribute
        assert "_ARRAY_DIMENSIONS" not in root["call_genotype"].attrs


def test_preflight_flags_missing_required_array(store):
    def drop_sample_id(root):
        del root["sample_id"]

    corrupt(store, drop_sample_id)
    assert any("sample_id" in p for p in dataset.preflight(store))
    with pytest.raises(ValueError, match="sample_id"):
        dataset.open_dataset(store)


def test_preflight_flags_wrong_version(store):
    def downgrade(root):
        root.attrs["vcf_zarr_version"] = "0.4"

    corrupt(store, downgrade)
    assert any("vcf_zarr_version" in p for p in dataset.preflight(store))


def test_preflight_reports_problems_without_raising(store):
    def drop_contig(root):
        del root["variant_contig"]

    corrupt(store, drop_contig)
    problems = dataset.preflight(store)
    assert problems
    assert all(isinstance(p, str) for p in problems)


def test_preflight_flags_wrong_dims_on_a_fixed_array(store):
    def misname(root):
        root["variant_position"].attrs["_ARRAY_DIMENSIONS"] = ["not_variants"]

    corrupt(store, misname)
    assert any(
        "variant_position: dimensions" in p and "not_variants" in p
        for p in dataset.preflight(store)
    )


def test_preflight_flags_wrong_number_of_dimension_names(store):
    def add_one(root):
        root["variant_DP"].attrs["_ARRAY_DIMENSIONS"] = ["variants", "extra"]

    corrupt(store, add_one)
    assert any(
        "variant_DP: 2 dimension names for 1 dims" in p
        for p in dataset.preflight(store)
    )


def test_preflight_flags_missing_dimension_names(store):
    def strip(root):
        del root["variant_DP"].attrs["_ARRAY_DIMENSIONS"]

    corrupt(store, strip)
    assert any("variant_DP: no dimension names" in p for p in dataset.preflight(store))


@pytest.mark.parametrize("bad", [7, "variants", ["variants", 3]])
def test_preflight_flags_malformed_dimension_names(store, bad):
    """The attribute is writer-controlled JSON; anything but a list of names
    is reported as a problem, not raised as a TypeError."""

    def mangle(root):
        root["variant_DP"].attrs["_ARRAY_DIMENSIONS"] = bad

    corrupt(store, mangle)
    assert any("variant_DP: no dimension names" in p for p in dataset.preflight(store))


def test_preflight_reports_a_path_that_is_no_zarr_group(tmp_path):
    (tmp_path / "notes.txt").write_text("not a store\n")
    problems = dataset.preflight(tmp_path)
    assert len(problems) == 1
    assert problems[0].startswith("not a Zarr group")
    with pytest.raises(ValueError, match="not a Zarr group"):
        dataset.open_dataset(tmp_path)


def test_preflight_raises_for_a_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        dataset.preflight(tmp_path / "absent.vcz")


# The metadata document at the group root and at one array, per format.
_METADATA_FILES = {
    2: (".zgroup", "variant_position/.zarray"),
    3: ("zarr.json", "variant_position/zarr.json"),
}


@BOTH_FORMATS
@pytest.mark.parametrize("which", ["root", "array"])
def test_preflight_reports_metadata_that_is_not_json(
    reference_stores, tmp_path, zarr_format, which
):
    """Broken JSON in a metadata document is a problem, not a JSONDecodeError."""
    out = tmp_path / "sample.vcz"
    shutil.copytree(reference_stores[zarr_format], out)
    root_doc, array_doc = _METADATA_FILES[zarr_format]
    (out / (root_doc if which == "root" else array_doc)).write_text("{not json")
    problems = dataset.preflight(out)
    assert len(problems) == 1
    assert problems[0].startswith("store metadata does not parse")
    with pytest.raises(ValueError, match="does not parse"):
        dataset.open_dataset(out)


@BOTH_FORMATS
def test_preflight_reports_metadata_that_parses_but_is_wrong(
    reference_stores, tmp_path, zarr_format
):
    """Valid JSON that zarr rejects (a shape that is not a list of ints)
    goes the same way; zarr raises TypeError for this rather than ValueError."""
    out = tmp_path / "sample.vcz"
    shutil.copytree(reference_stores[zarr_format], out)
    doc = out / _METADATA_FILES[zarr_format][1]
    meta = json.loads(doc.read_text())
    meta["shape"] = "wide"
    doc.write_text(json.dumps(meta))
    problems = dataset.preflight(out)
    assert len(problems) == 1
    assert problems[0].startswith("store metadata does not parse")


def test_preflight_flags_inconsistent_dimension_sizes(store):
    def widen(root):
        del root["contig_id"]
        array = root.create_array("contig_id", shape=(2,), dtype="T", chunks=(2,))
        array.attrs["_ARRAY_DIMENSIONS"] = [core.DIM_CONTIGS]
        array[:] = np.array(["chr1", "chr2"], dtype="T")

    corrupt(store, widen)
    assert any(
        "dim 'contigs' is" in p and "elsewhere" in p for p in dataset.preflight(store)
    )


def test_preflight_refuses_a_multisample_store(multisample_vcf, tmp_path):
    """convert.py refuses cohort files; a store converted elsewhere must not
    get around that by being opened directly."""
    out = tmp_path / "multi.vcz"
    b2z.convert([str(multisample_vcf)], str(out), variants_chunk_size=CHUNK)
    problems = dataset.preflight(out)
    assert any("single-sample" in p for p in problems)
    with pytest.raises(ValueError, match="single-sample"):
        dataset.open_dataset(out)


@pytest.mark.parametrize(
    ("name", "dtype"),
    [
        ("variant_position", "T"),  # text where a query compares integers
        ("variant_contig", "u1"),  # unsigned cannot hold the sentinels
        ("variant_quality", "i4"),
        ("variant_id", "S8"),  # raw bytes are not text
        ("variant_id", "U8"),  # fixed-width unicode truncates
        ("call_genotype_mask", "i1"),
    ],
)
def test_preflight_flags_the_wrong_kind_of_value(store, name, dtype):
    def retype(root):
        node = root[name]
        assert isinstance(node, zarr.Array)
        data = np.asarray(node[:])
        if dtype in ("T", "S8", "U8"):
            data = np.array([str(x) for x in data.reshape(-1)], dtype=dtype).reshape(
                data.shape
            )
        else:
            data = data.astype(dtype)
        dims = dataset.array_dims(node)
        del root[name]
        array = root.create_array(
            name, shape=data.shape, dtype=data.dtype, chunks=data.shape
        )
        array.attrs["_ARRAY_DIMENSIONS"] = dims
        array[:] = data

    corrupt(store, retype)
    assert any(f"{name}: dtype" in p for p in dataset.preflight(store))
