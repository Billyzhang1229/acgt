"""Open and save a VCZ store; preflight.

Everything downstream of import goes through here: an xarray Dataset backed by
Zarr, following sgkit's variant/sample dimension conventions. Preflight checks
that a store carries what the schema in core.py says it must, before any query
touches it.

Both Zarr storage formats are supported, and every place the two formats
differ is confined to this module — the same boundary bio2zarr keeps in its
zarr_utils. The differences are small: format 2 names dimensions through the
`_ARRAY_DIMENSIONS` attribute and compresses with numcodecs, format 3 names
them in array metadata and compresses with zarr's own codecs. Readers here
accept either; `save` writes whichever the caller asks for, format 2 by
default because the VCF Zarr spec is still written against it.
"""

import concurrent.futures
import dataclasses
import itertools
import math
import os
import shutil
import tempfile
from pathlib import Path

import numcodecs
import numpy as np
import xarray as xr
import zarr
import zarr.codecs

from acgt import __version__, core

DEFAULT_CHUNK_SIZE = 10_000
_WRITE_THREADS = 8


# --------------------------------------------------------------------------
# format dispatch
# --------------------------------------------------------------------------
def _compressors(zarr_format, level):
    if zarr_format == 2:
        return [numcodecs.Blosc("zstd", clevel=level, shuffle=numcodecs.Blosc.SHUFFLE)]
    return [zarr.codecs.BloscCodec(cname="zstd", clevel=level, shuffle="shuffle")]


def _create_array(root, zarr_format, name, dims, description, **kwargs):
    if zarr_format == 3:
        array = root.create_array(name, dimension_names=dims, **kwargs)
    else:
        array = root.create_array(name, **kwargs)
        array.attrs["_ARRAY_DIMENSIONS"] = dims
    # the reference reader rebuilds VCF header lines from this attribute
    array.attrs["description"] = description
    return array


def array_dims(array):
    """Dimension names of one zarr array, whichever format carries them.

    Format 2 stores them in the `_ARRAY_DIMENSIONS` attribute, format 3 in
    the array metadata itself. Returns None when neither is present.
    """
    dims = array.attrs.get("_ARRAY_DIMENSIONS")
    if dims is None:
        dims = getattr(array.metadata, "dimension_names", None)
    return None if dims is None else list(dims)


def _array(root, name):
    """root[name] as an array. Group access is typed as array-or-subgroup,
    but a VCZ store nests no groups."""
    node = root[name]
    assert isinstance(node, zarr.Array)
    return node


# --------------------------------------------------------------------------
# derived arrays and dtype choices
# --------------------------------------------------------------------------
def build_region_index(contig, position, length, chunk_size) -> np.ndarray:
    """The spec's region index: one row per (variants-chunk, contig) pair,
    with columns [chunk, contig, start, end, max_end, n_records].

    max_end uses variant_length so an interval query cannot miss a long
    deletion whose start lies before the query window. Returned as int64;
    save() narrows it together with variant_position, whose dtype the spec
    requires the index to share.
    """
    n = position.shape[0]
    if n == 0:
        return np.empty((0, len(core.REGION_INDEX_COLUMNS)), np.int64)
    rows = []
    for c0 in range(0, n, chunk_size):
        c1 = min(c0 + chunk_size, n)
        ct, ps, ln = contig[c0:c1], position[c0:c1], length[c0:c1]
        cuts = np.concatenate(
            [[0], np.flatnonzero(np.diff(ct.astype(np.int64))) + 1, [ct.shape[0]]]
        )
        for a, b in itertools.pairwise(cuts):
            rows.append(
                (
                    c0 // chunk_size,
                    int(ct[a]),
                    int(ps[a]),
                    int(ps[b - 1]),
                    int((ps[a:b].astype(np.int64) + ln[a:b] - 1).max()),
                    int(b - a),
                )
            )
    return np.array(rows, dtype=np.int64)


def _smallest_int(a):
    lo, hi = int(a.min()), int(a.max())
    for dt in ("i1", "i2", "i4", "i8"):
        info = np.iinfo(dt)
        if info.min <= lo and hi <= info.max:
            return dt
    return "i8"


def _with_region_index(arrays, schema, chunk_size):
    """Attach region_index to the arrays and account for it in the schema."""
    needed = {"variant_contig", "variant_position", "variant_length"}
    if "region_index" in arrays or not needed <= arrays.keys():
        return schema
    index = build_region_index(
        arrays["variant_contig"],
        arrays["variant_position"],
        arrays["variant_length"],
        chunk_size,
    )
    arrays["region_index"] = index
    dims = dict(schema.dims)
    dims[core.DIM_REGION_INDEX_VALUES] = index.shape[0]
    dims[core.DIM_REGION_INDEX_FIELDS] = index.shape[1]
    fields = schema.fields
    if all(f.name != "region_index" for f in fields):
        fields = (*fields, core.FIXED_ARRAYS["region_index"])
    return dataclasses.replace(schema, dims=dims, fields=fields)


def _forced_dtypes(arrays, downcast):
    """The spec ties region_index's dtype to variant_position's, so the two
    are narrowed together instead of independently."""
    if "variant_position" not in arrays or "region_index" not in arrays:
        return {}
    if not downcast:
        dt = arrays["variant_position"].dtype.str
        return {"region_index": dt}
    both = np.concatenate(
        [
            np.asarray(arrays["variant_position"], dtype=np.int64).reshape(-1),
            arrays["region_index"].reshape(-1),
        ]
    )
    dt = _smallest_int(both) if both.size else "i4"
    return {"variant_position": dt, "region_index": dt}


# --------------------------------------------------------------------------
# save
# --------------------------------------------------------------------------
def save(
    arrays,
    schema,
    out,
    *,
    zarr_format=2,
    chunk_size=DEFAULT_CHUNK_SIZE,
    compress_level=5,
    downcast=True,
    spill=None,
    threads=_WRITE_THREADS,
    overwrite=False,
    meta_information=(),
) -> Path:
    """Write a VCZ store from columnar arrays and the Schema describing them.

    `arrays` maps array names to numpy arrays. Fixed-width bytes arrays
    (dtype kind "S") become variable-length string arrays on disk; `spill`
    carries the values that were too long for their fixed-width buffer, as
    {(array_name, index_tuple): full_string}. region_index is derived here
    when the caller did not provide one. Integer arrays are narrowed to the
    smallest safe dtype unless `downcast` is off. `meta_information` is the
    spec's optional list of (key, value) header pairs the arrays do not
    already carry; it is stored so the header can be rendered back.

    An existing `out` is refused unless `overwrite` says otherwise, and the
    store is built in a hidden sibling directory and renamed into place at
    the end, so a failure part-way never destroys what was there and never
    leaves a half-written store at the destination.
    """
    if zarr_format not in (2, 3):
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format!r}")
    out = Path(out)
    if out.exists() and not overwrite:
        raise ValueError(f"{out} already exists; pass overwrite=True to replace it")
    spill = spill or {}
    arrays = dict(arrays)
    schema = _with_region_index(arrays, schema, chunk_size)
    for name in arrays:
        if name not in schema.field_map() and name not in core.FIXED_ARRAYS:
            raise ValueError(f"no dimensions known for array {name!r}")
    forced = _forced_dtypes(arrays, downcast)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=out.parent))
    try:
        _write_store(
            tmp,
            arrays,
            schema,
            zarr_format,
            chunk_size,
            compress_level,
            forced,
            downcast,
            spill,
            threads,
            meta_information,
        )
        _publish(tmp, out, overwrite)
    finally:
        # never leave a complete partial store behind after a failed publish
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def _publish(tmp, out, overwrite):
    """Move the finished store from `tmp` to `out`.

    Re-checks `overwrite`, since the first check ran before a possibly long
    write. Replacement is rename-aside then rename-in with rollback, so a
    failure at any point leaves either the old or the new store at `out`.
    """
    if not out.exists():
        os.replace(tmp, out)
        return
    if not overwrite:
        raise ValueError(
            f"{out} appeared during conversion; pass overwrite=True to replace it"
        )
    holding = Path(tempfile.mkdtemp(prefix=f".{out.name}.replaced-", dir=out.parent))
    backup = holding / "old"
    os.replace(out, backup)
    try:
        os.replace(tmp, out)
    except BaseException:
        os.replace(backup, out)
        raise
    finally:
        shutil.rmtree(holding, ignore_errors=True)


def _write_store(
    path,
    arrays,
    schema,
    zarr_format,
    chunk_size,
    compress_level,
    forced,
    downcast,
    spill,
    threads,
    meta_information,
):
    specs = {f.name: f for f in schema.fields}
    root = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    root.attrs["vcf_zarr_version"] = core.VCF_ZARR_VERSION
    root.attrs["source"] = f"acgt-{__version__}"
    root.attrs["vcf_meta_information"] = [list(pair) for pair in meta_information]
    root.attrs[core.SCHEMA_ATTR] = schema.asdict()
    compressors = _compressors(zarr_format, compress_level)

    jobs = []
    for name, a in arrays.items():
        spec = specs.get(name) or core.FIXED_ARRAYS[name]
        dims = list(spec.dims)
        is_str = a.dtype.kind == "S"
        # VCF Character is a fixed-width U1 on disk, per the spec and the
        # reference; everything else textual is a variable-length string
        is_char = is_str and spec.kind == core.KIND_CHAR
        src = a
        if not is_str and a.dtype.kind == "i":
            # forced dtypes apply even to empty arrays -- a zero-record
            # store must still keep region_index and variant_position aligned
            want = forced.get(name)
            if want is None and downcast and a.size:
                want = _smallest_int(a)
            if want is not None:
                src = a.astype(want)
        z = _create_array(
            root,
            zarr_format,
            name,
            dims,
            spec.description,
            shape=src.shape,
            dtype="U1" if is_char else str if is_str else src.dtype,
            chunks=(max(1, min(chunk_size, src.shape[0])), *src.shape[1:]),
            compressors=compressors,
            config={"write_empty_chunks": True},
        )
        patches = {k[1]: v for k, v in spill.items() if k[0] == name}
        step = z.chunks[0]
        for c0 in range(0, src.shape[0], step):
            jobs.append(
                (z, src, c0, min(c0 + step, src.shape[0]), is_str, is_char, patches)
            )

    # biggest chunks first so the tail of the thread pool is short
    jobs.sort(key=lambda j: -(j[3] - j[2]) * j[1].dtype.itemsize)

    def run(job):
        z, a, c0, c1, is_str, is_char, patches = job
        if not is_str:
            z[c0:c1] = a[c0:c1]
            return
        block = a[c0:c1]
        if is_char:
            z[c0:c1] = np.char.decode(block, "utf-8").astype("U1")
            return
        # tolist() + bytes.decode goes straight to object dtype; np.char.decode
        # materialises a U-dtype copy (4 bytes/char) and is far slower.
        flat = np.empty(block.size, dtype=object)
        flat[:] = [x.decode("utf-8") for x in block.reshape(-1).tolist()]
        out_block = flat.reshape(block.shape)
        for idx, value in patches.items():
            if c0 <= idx[0] < c1:
                out_block[(idx[0] - c0, *idx[1:])] = value
        z[c0:c1] = out_block

    with concurrent.futures.ThreadPoolExecutor(threads) as ex:  # blosc drops the GIL
        list(ex.map(run, jobs))
    if zarr_format == 2:
        # .zmetadata is standard for format 2; format 3 has no consolidated
        # metadata in its spec, and bio2zarr writes none either
        zarr.consolidate_metadata(root.store)


# --------------------------------------------------------------------------
# preflight and open
# --------------------------------------------------------------------------
# numpy dtype kinds acceptable for each schema kind
_KIND_ACCEPTS = {
    # signed only: the missing and fill sentinels are -1 and -2, so an
    # unsigned array cannot represent them and the spec's dtypes are all i*
    core.KIND_INT: "i",
    core.KIND_FLOAT: "f",
    core.KIND_BOOL: "b",
    # the spec says |O; zarr-python 3 presents that as numpy's StringDType
    # (kind T), and fixed-width unicode is the same text. Raw bytes are not
    # text and no writer of this spec version produces them.
    core.KIND_STR: "OTU",
    core.KIND_CHAR: "U",  # exactly U1 -- the width is checked separately
}
_CHAR_ITEMSIZE = np.dtype("U1").itemsize


def preflight(path, *, deep=False) -> list[str]:
    """Structural checks against VCF Zarr 0.5 and the stored acgt schema.

    Returns the list of problems found; empty means the store passes, and a
    malformed store adds problems rather than raising. Works on any
    conforming store, not only ones this package wrote: the acgt schema
    attribute is checked when present, and fixed arrays are checked against
    the spec regardless. The default checks read metadata only, keeping an
    open cheap however large the store; `deep` additionally re-derives
    region_index and range-checks variant_contig, reading the store one
    variants-chunk at a time.
    """
    z = zarr.open_group(path, mode="r")
    keys = set(z.array_keys())
    problems = []

    version = z.attrs.get("vcf_zarr_version")
    if version != core.VCF_ZARR_VERSION:
        problems.append(
            f"vcf_zarr_version is {version!r}, expected {core.VCF_ZARR_VERSION!r}"
        )

    problems.extend(
        f"missing mandatory array {k}" for k in sorted(core.REQUIRED_ARRAYS - keys)
    )

    # what each array should look like: the stored schema where one exists,
    # the spec's fixed arrays otherwise
    schema = None
    raw = z.attrs.get(core.SCHEMA_ATTR)
    if raw is not None:
        try:
            schema = core.Schema.fromdict(raw)
        except (KeyError, TypeError, ValueError) as e:
            problems.append(f"stored acgt schema does not parse: {e}")
    specs = dict(core.FIXED_ARRAYS)
    declared_sizes = {}
    if schema is not None:
        specs.update(schema.field_map())
        declared_sizes = dict(schema.dims)
        problems.extend(
            f"schema lists {f.name} but the store lacks it"
            for f in schema.fields
            if f.name not in keys
        )
        problems.extend(
            f"{k}: not in the stored schema" for k in sorted(keys) if k not in specs
        )

    arrays = {k: _array(z, k) for k in sorted(keys)}
    dims = {}
    sizes = {}
    for k, array in arrays.items():
        d = array_dims(array)
        if d is None:
            problems.append(f"{k}: no dimension names")
            continue
        if len(d) != array.ndim:
            problems.append(f"{k}: {len(d)} dimension names for {array.ndim} dims")
        dims[k] = d
        spec = specs.get(k)
        if spec is not None:
            if list(spec.dims) != d:
                problems.append(
                    f"{k}: dimensions {d} do not match the schema's {list(spec.dims)}"
                )
            if array.dtype.kind not in _KIND_ACCEPTS[spec.kind]:
                problems.append(f"{k}: dtype {array.dtype} is not {spec.kind}")
            elif spec.kind == core.KIND_CHAR and array.dtype.itemsize != _CHAR_ITEMSIZE:
                problems.append(f"{k}: dtype {array.dtype} is not the spec's U1")
        for name, size in zip(d, array.shape, strict=False):
            prev = sizes.setdefault(name, size)
            if prev != size:
                problems.append(f"{k}: dim {name!r} is {size}, elsewhere {prev}")

    problems.extend(
        f"schema says dim {name!r} is {size}, store has {sizes[name]}"
        for name, size in declared_sizes.items()
        if name in sizes and sizes[name] != size
    )

    # the reference reader's chunking rule: call_ arrays set the base and
    # must agree exactly; without any, the GCD of the rest is the base;
    # every variants-dimension chunk is a positive multiple of the base
    vchunks = {
        k: arrays[k].chunks[0]
        for k, d in dims.items()
        if d and d[0] == core.DIM_VARIANTS
    }
    if vchunks:
        call_sizes = {c for k, c in vchunks.items() if k.startswith("call_")}
        if len(call_sizes) > 1:
            problems.append(
                f"call_ arrays must share one variants chunk size, "
                f"found {sorted(call_sizes)}"
            )
        else:
            base = call_sizes.pop() if call_sizes else math.gcd(*vchunks.values())
            offenders = sorted({c for c in vchunks.values() if c % base})
            if offenders:
                problems.append(
                    f"variants chunk sizes {offenders} are not multiples "
                    f"of the base chunk size ({base})"
                )

    if "region_index" in keys:
        index = arrays["region_index"]
        n_fields = len(core.REGION_INDEX_COLUMNS)
        if index.ndim != 2 or index.shape[1] != n_fields:
            problems.append(
                f"region_index has shape {index.shape}, expected (*, {n_fields})"
            )
        if (
            "variant_position" in keys
            and index.dtype != arrays["variant_position"].dtype
        ):
            problems.append(
                f"region_index dtype {index.dtype} != variant_position "
                f"{arrays['variant_position'].dtype}"
            )
        if "variant_length" not in keys:
            problems.append("region_index present but variant_length missing")

    if (
        "filter_id" in keys
        and arrays["filter_id"].shape[0]
        and str(arrays["filter_id"][0]) != "PASS"
    ):
        problems.append("filter_id[0] must be PASS")

    if deep:
        problems.extend(_deep_check(arrays, keys))
    return problems


def _deep_check(arrays, keys):
    """The checks metadata cannot answer, one variants-chunk at a time."""
    problems = []
    if "variant_contig" in keys and "contig_id" in keys:
        contig = arrays["variant_contig"]
        n_contigs = arrays["contig_id"].shape[0]
        step = contig.chunks[0]
        for c0 in range(0, contig.shape[0], step):
            block = np.asarray(contig[c0 : c0 + step])
            if block.size and (block.min() < 0 or block.max() >= n_contigs):
                problems.append("variant_contig out of range of contig_id")
                break

    needed = {"region_index", "variant_contig", "variant_position", "variant_length"}
    if needed <= keys:
        index = arrays["region_index"]
        position = arrays["variant_position"]
        if index.ndim == 2 and index.shape[1] == len(core.REGION_INDEX_COLUMNS):
            # the index is defined per variant_position chunk; rebuild it
            # chunk by chunk so memory stays bounded by one chunk
            step = position.chunks[0]
            rows = []
            for chunk_id, c0 in enumerate(range(0, position.shape[0], step)):
                block = slice(c0, c0 + step)
                part = build_region_index(
                    np.asarray(arrays["variant_contig"][block]),
                    np.asarray(position[block]),
                    np.asarray(arrays["variant_length"][block]),
                    step,
                )
                part[:, 0] = chunk_id
                rows.append(part)
            want = (
                np.concatenate(rows)
                if rows
                else np.empty((0, len(core.REGION_INDEX_COLUMNS)), np.int64)
            )
            if not np.array_equal(np.asarray(index[:], dtype=np.int64), want):
                problems.append("region_index does not match the data")
    return problems


def open_dataset(path, *, check=True):
    """Open a VCZ store lazily, preflighting it first.

    The Dataset is backed by the store rather than loaded: slice by region,
    never materialize a whole array. The default preflight is metadata-only
    for the same reason; preflight(path, deep=True) exists for the full
    re-derivation. Pass check=False to skip the check when opening a store
    this process just wrote and already checked.
    """
    if check:
        problems = preflight(path)
        if problems:
            raise ValueError("store fails preflight: " + "; ".join(problems))
    # consolidated=False: stores from other writers may carry none, and the
    # fallback warning would be noise; reading per-array metadata is cheap
    return xr.open_zarr(path, chunks=None, consolidated=False, mask_and_scale=False)
