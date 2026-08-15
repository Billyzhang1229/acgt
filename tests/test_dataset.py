"""save, open_dataset, and preflight round-trips on hand-built arrays.

No VCF in sight: arrays and a Schema go in, a store comes out, preflight and
the reader bring it back. Every assertion runs against both Zarr storage
formats, since supporting both is the module's whole reason to exist.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import zarr

from acgt import core, dataset

BOTH_FORMATS = pytest.mark.parametrize("zarr_format", [2, 3])


def tiny():
    """Five variants over two contigs, one spilled long ID."""
    genotype = np.array([[[0, 1]], [[1, 1]], [[-1, -1]], [[0, 0]], [[1, 0]]], "i1")
    arrays = {
        "variant_contig": np.array([0, 0, 0, 1, 1], "i2"),
        "variant_position": np.array([100, 200, 300, 50, 60], "i4"),
        "variant_length": np.array([1, 1, 3, 1, 1], "i4"),
        "variant_id": np.array([b".", b"rs2", b".", b".", b""], "S4"),
        "variant_id_mask": np.array([True, False, True, True, False]),
        "variant_allele": np.array(
            [[b"A", b"G"], [b"C", b"T"], [b"G", b"GAT"], [b"T", b"A"], [b"A", b"C"]],
            "S3",
        ),
        "variant_quality": np.array(
            [10.0, core.FLOAT32_MISSING, 30.0, 40.0, 50.0], "f4"
        ),
        "variant_filter": np.ones((5, 1), "b1"),
        "call_genotype": genotype,
        "call_genotype_mask": genotype < 0,
        "call_genotype_phased": np.zeros((5, 1), "b1"),
        "sample_id": np.array([b"S1"], "S2"),
        "contig_id": np.array([b"chr1", b"chr2"], "S4"),
        "contig_length": np.array([1000, 2000], "i8"),
        "filter_id": np.array([b"PASS"], "S4"),
        "filter_description": np.array([b"All filters passed"], "S18"),
    }
    dims = {
        core.DIM_VARIANTS: 5,
        core.DIM_SAMPLES: 1,
        core.DIM_PLOIDY: 2,
        core.DIM_ALLELES: 2,
        core.DIM_FILTERS: 1,
        core.DIM_CONTIGS: 2,
    }
    fields = tuple(core.FIXED_ARRAYS[name] for name in arrays)
    schema = core.Schema(dims=dims, fields=fields, source="tiny", build="test")
    spill = {("variant_id", (4,)): "rs4242424242"}
    return arrays, schema, spill


def saved(tmp_path, zarr_format, mutate=None):
    arrays, schema, spill = tiny()
    if mutate is not None:
        mutate(arrays, schema)
    return dataset.save(
        arrays, schema, tmp_path / "tiny.vcz", zarr_format=zarr_format, spill=spill
    )


def corrupt(out, change):
    """Apply `change` to an open group, keeping consolidated metadata true."""
    root = zarr.open_group(out, mode="r+")
    change(root)
    if root.metadata.zarr_format == 2:  # format 3 stores carry none
        zarr.consolidate_metadata(root.store)


def rechunk(root, name, chunks, dtype=None):
    """Recreate one array with different chunks or dtype, keeping data."""
    node = root[name]
    assert isinstance(node, zarr.Array)
    data = np.asarray(node[:])
    if dtype is not None:
        data = data.astype(dtype)
    dim_names = dataset.array_dims(node)
    del root[name]
    array = root.create_array(name, shape=data.shape, dtype=data.dtype, chunks=chunks)
    array.attrs["_ARRAY_DIMENSIONS"] = dim_names
    array[:] = data


@BOTH_FORMATS
def test_round_trip(tmp_path, zarr_format):
    out = saved(tmp_path, zarr_format)
    assert dataset.preflight(out) == []
    root = zarr.open_group(out, mode="r")
    assert root.metadata.zarr_format == zarr_format
    assert root.attrs["vcf_zarr_version"] == core.VCF_ZARR_VERSION

    ds = dataset.open_dataset(out)
    assert ds["variant_position"].values.tolist() == [100, 200, 300, 50, 60]
    assert str(ds["variant_allele"].values[2, 1]) == "GAT"
    assert str(ds["variant_id"].values[4]) == "rs4242424242"  # spilled value
    quality_bits = ds["variant_quality"].values.view("i4")
    missing_bits = np.array([core.FLOAT32_MISSING], "f4").view("i4")[0]
    assert quality_bits[1] == missing_bits  # sentinel survives bit for bit


@BOTH_FORMATS
def test_region_index_derived_and_typed(tmp_path, zarr_format):
    out = saved(tmp_path, zarr_format)
    root = zarr.open_group(out, mode="r")
    index = root["region_index"]
    position = root["variant_position"]
    assert isinstance(index, zarr.Array)
    assert isinstance(position, zarr.Array)
    assert index.dtype == position.dtype
    assert np.asarray(index[:], np.int64).tolist() == [
        [0, 0, 100, 300, 302, 3],
        [0, 1, 50, 60, 60, 2],
    ]


@BOTH_FORMATS
def test_dimension_names_readable_either_way(tmp_path, zarr_format):
    out = saved(tmp_path, zarr_format)
    root = zarr.open_group(out, mode="r")
    assert dataset.array_dims(root["call_genotype"]) == [
        core.DIM_VARIANTS,
        core.DIM_SAMPLES,
        core.DIM_PLOIDY,
    ]
    if zarr_format == 3:
        # format 3 carries names in array metadata, not the xarray attribute
        assert "_ARRAY_DIMENSIONS" not in root["call_genotype"].attrs


def test_build_region_index_splits_chunks_and_contigs():
    index = dataset.build_region_index(
        contig=np.array([0, 0, 0, 1, 1]),
        position=np.array([100, 200, 300, 50, 60]),
        length=np.array([1, 1, 3, 1, 1]),
        chunk_size=2,
    )
    assert index.tolist() == [
        [0, 0, 100, 200, 200, 2],
        [1, 0, 300, 300, 302, 1],
        [1, 1, 50, 50, 50, 1],
        [2, 1, 60, 60, 60, 1],
    ]


def test_save_rejects_unknown_format(tmp_path):
    arrays, schema, _ = tiny()
    with pytest.raises(ValueError, match="zarr_format"):
        dataset.save(arrays, schema, tmp_path / "t.vcz", zarr_format=4)


@BOTH_FORMATS
def test_preflight_flags_missing_required_array(tmp_path, zarr_format):
    def drop_sample_id(arrays, schema):
        del arrays["sample_id"]
        schema.fields = tuple(f for f in schema.fields if f.name != "sample_id")

    out = saved(tmp_path, zarr_format, mutate=drop_sample_id)
    assert any("sample_id" in p for p in dataset.preflight(out))
    with pytest.raises(ValueError, match="sample_id"):
        dataset.open_dataset(out)


@BOTH_FORMATS
def test_preflight_flags_wrong_version(tmp_path, zarr_format):
    out = saved(tmp_path, zarr_format)

    def downgrade(root):
        root.attrs["vcf_zarr_version"] = "0.4"

    corrupt(out, downgrade)
    assert any("vcf_zarr_version" in p for p in dataset.preflight(out))


@BOTH_FORMATS
def test_deep_preflight_flags_stale_region_index(tmp_path, zarr_format):
    out = saved(tmp_path, zarr_format)

    def shift_start(root):
        index = root["region_index"][:]
        index[0, 2] += 1
        root["region_index"][:] = index

    corrupt(out, shift_start)
    # metadata cannot see stale values, so the default pass stays clean;
    # the deep pass re-derives the index and catches the drift
    assert dataset.preflight(out) == []
    assert any("region_index" in p for p in dataset.preflight(out, deep=True))


def test_preflight_flags_pass_not_first(tmp_path):
    def rename_filter(arrays, schema):
        arrays["filter_id"] = np.array([b"q10"], "S4")

    out = saved(tmp_path, 2, mutate=rename_filter)
    assert any("PASS" in p for p in dataset.preflight(out))


def test_preflight_flags_schema_disagreement(tmp_path):
    def lie_about_ploidy(arrays, schema):
        schema.dims = {**schema.dims, core.DIM_PLOIDY: 3}

    out = saved(tmp_path, 2, mutate=lie_about_ploidy)
    assert any(core.DIM_PLOIDY in p for p in dataset.preflight(out))


def test_stored_schema_round_trips(tmp_path):
    out = saved(tmp_path, 2)
    root = zarr.open_group(out, mode="r")
    schema = core.Schema.fromdict(root.attrs[core.SCHEMA_ATTR])
    assert schema.source == "tiny"
    assert schema.dims[core.DIM_REGION_INDEX_VALUES] == 2  # added by save
    assert "region_index" in schema.field_map()


def test_schema_fields_needed_for_unknown_arrays(tmp_path):
    arrays, schema, _ = tiny()
    arrays["variant_XX"] = np.zeros(5, "i4")
    with pytest.raises(ValueError, match="variant_XX"):
        dataset.save(arrays, schema, tmp_path / "t.vcz")


def test_save_refuses_then_overwrites(tmp_path):
    arrays, schema, spill = tiny()
    out = dataset.save(arrays, schema, tmp_path / "t.vcz", spill=spill)
    with pytest.raises(ValueError, match="already exists"):
        dataset.save(arrays, schema, out, spill=spill)
    dataset.save(arrays, schema, out, spill=spill, overwrite=True)
    assert dataset.preflight(out) == []


def test_failed_save_never_touches_the_target(tmp_path):
    out = tmp_path / "t.vcz"
    out.mkdir()
    marker = out / "marker.txt"
    marker.write_text("precious")
    arrays, schema, _ = tiny()
    arrays["variant_XX"] = np.zeros(5, "i4")
    with pytest.raises(ValueError, match="variant_XX"):
        dataset.save(arrays, schema, out, overwrite=True)
    assert marker.read_text() == "precious"
    assert not list(tmp_path.glob(".t.vcz.partial-*"))


def test_failed_write_cleans_up_the_partial_directory(tmp_path):
    arrays, schema, _ = tiny()
    # a field lying about its rank makes array creation itself fail
    bad = core.ArraySpec(
        "variant_position", core.KIND_INT, (core.DIM_VARIANTS, "extra_dim")
    )
    schema.fields = tuple(
        bad if f.name == "variant_position" else f for f in schema.fields
    )
    schema.dims = {**schema.dims, "extra_dim": 1}
    with pytest.raises(ValueError, match="dimension_names"):
        dataset.save(arrays, schema, tmp_path / "t.vcz", zarr_format=3)
    assert not (tmp_path / "t.vcz").exists()
    assert not list(tmp_path.glob(".t.vcz.partial-*"))


def test_preflight_flags_wrong_dtype_kind(tmp_path):
    def int_id(arrays, schema):
        arrays["variant_id"] = np.zeros(5, "i4")

    out = saved(tmp_path, 2, mutate=int_id)
    assert any("is not str" in p for p in dataset.preflight(out))


def test_preflight_flags_declared_width_mismatch(tmp_path):
    arrays, schema, spill = tiny()
    arrays["variant_XX"] = np.zeros((5, 3), "i4")
    schema.fields = (
        *schema.fields,
        core.ArraySpec("variant_XX", core.KIND_INT, (core.DIM_VARIANTS, "XX_dim")),
    )
    schema.dims = {**schema.dims, "XX_dim": 2}
    out = dataset.save(arrays, schema, tmp_path / "t.vcz", spill=spill)
    assert any("XX_dim" in p for p in dataset.preflight(out))


def test_preflight_reports_problems_without_raising(tmp_path):
    out = saved(tmp_path, 2)

    def drop_contig(root):
        del root["variant_contig"]

    corrupt(out, drop_contig)
    problems = dataset.preflight(out, deep=True)
    assert any("variant_contig" in p for p in problems)


def test_preflight_accepts_chunks_that_are_multiples_of_the_call_base(tmp_path):
    arrays, schema, spill = tiny()
    out = dataset.save(arrays, schema, tmp_path / "t.vcz", spill=spill, chunk_size=2)
    corrupt(out, lambda root: rechunk(root, "variant_id_mask", (4,)))
    assert dataset.preflight(out, deep=True) == []
    corrupt(out, lambda root: rechunk(root, "variant_id_mask", (3,)))
    assert any("multiples" in p for p in dataset.preflight(out))


def test_call_arrays_must_share_one_chunk_size(tmp_path):
    arrays, schema, spill = tiny()
    out = dataset.save(arrays, schema, tmp_path / "t.vcz", spill=spill, chunk_size=2)
    corrupt(out, lambda root: rechunk(root, "call_genotype_phased", (4, 1)))
    assert any("share one variants chunk" in p for p in dataset.preflight(out))


def test_gcd_is_the_base_without_call_arrays(tmp_path):
    arrays, schema, spill = tiny()
    for name in list(arrays):
        if name.startswith("call_"):
            del arrays[name]
    schema.fields = tuple(f for f in schema.fields if not f.name.startswith("call_"))
    out = dataset.save(arrays, schema, tmp_path / "t.vcz", spill=spill, chunk_size=2)
    # 2 and 3 are coprime; with no call_ arrays the base is gcd = 1 and
    # the reference reader accepts this layout, so preflight must too
    corrupt(out, lambda root: rechunk(root, "variant_id_mask", (3,)))
    assert dataset.preflight(out, deep=True) == []


def test_preflight_rejects_unsigned_integers(tmp_path):
    out = saved(tmp_path, 2)
    corrupt(out, lambda root: rechunk(root, "variant_contig", (5,), dtype="u1"))
    assert any("is not int" in p for p in dataset.preflight(out))


def test_preflight_rejects_bytes_for_strings(tmp_path):
    out = saved(tmp_path, 2)
    corrupt(out, lambda root: rechunk(root, "sample_id", (1,), dtype="S2"))
    assert any("sample_id: dtype" in p for p in dataset.preflight(out))


def test_preflight_requires_u1_for_char(tmp_path):
    arrays, schema, spill = tiny()
    arrays["variant_C"] = np.array([b"a", b"b", b".", b"c", b"d"], "S1")
    schema.fields = (
        *schema.fields,
        core.ArraySpec("variant_C", core.KIND_CHAR, (core.DIM_VARIANTS,)),
    )
    out = dataset.save(arrays, schema, tmp_path / "t.vcz", spill=spill)
    assert dataset.preflight(out) == []
    # a wider unicode array is still kind U but is not the spec's U1
    corrupt(out, lambda root: rechunk(root, "variant_C", (5,), dtype="U3"))
    assert any("not the spec's U1" in p for p in dataset.preflight(out))


def failing_rename_into(target, monkeypatch):
    """Make the first os.replace whose destination is `target` fail.

    Matched by destination path rather than call count (zarr renames temp
    files for every metadata write); later hits into `target`, such as the
    rollback, are let through and recorded.
    """
    real = os.replace
    hits = []

    def flaky(src, dst):
        if Path(dst) == Path(target):
            hits.append(str(src))
            if len(hits) == 1:
                raise OSError("simulated crash at publish")
        return real(src, dst)

    monkeypatch.setattr(dataset.os, "replace", flaky)
    return hits


def test_swap_failure_restores_the_original(tmp_path, monkeypatch):
    arrays, schema, spill = tiny()
    out = dataset.save(arrays, schema, tmp_path / "t.vcz", spill=spill)
    changed, schema2, spill2 = tiny()
    changed["variant_position"] = changed["variant_position"] + 1000
    hits = failing_rename_into(out, monkeypatch)
    with pytest.raises(OSError, match="simulated crash"):
        dataset.save(changed, schema2, out, spill=spill2, overwrite=True)
    assert len(hits) == 2
    assert ".partial-" in hits[0]
    assert ".replaced-" in hits[1]
    monkeypatch.undo()
    assert dataset.preflight(out) == []
    ds = dataset.open_dataset(out)
    assert ds["variant_position"].values.tolist() == [100, 200, 300, 50, 60]
    assert not list(tmp_path.glob(".t.vcz.*"))  # no partial or holding dirs


def test_publish_failure_leaves_no_copy_behind(tmp_path, monkeypatch):
    arrays, schema, spill = tiny()
    out = tmp_path / "new.vcz"
    hits = failing_rename_into(out, monkeypatch)
    with pytest.raises(OSError, match="simulated crash"):
        dataset.save(arrays, schema, out, spill=spill)
    assert len(hits) == 1
    assert not out.exists()
    assert not list(tmp_path.glob(".new.vcz.*"))


def test_target_appearing_mid_write_is_refused(tmp_path, monkeypatch):
    arrays, schema, spill = tiny()
    out = tmp_path / "t.vcz"
    real_write = dataset._write_store

    def racy(path, *args, **kwargs):
        real_write(path, *args, **kwargs)
        out.mkdir()
        (out / "theirs.txt").write_text("someone else")

    monkeypatch.setattr(dataset, "_write_store", racy)
    with pytest.raises(ValueError, match="appeared during conversion"):
        dataset.save(arrays, schema, out, spill=spill)
    assert (out / "theirs.txt").read_text() == "someone else"
    assert not list(tmp_path.glob(".t.vcz.partial-*"))
