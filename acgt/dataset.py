"""Open and preflight a VCZ store.

Everything downstream of import goes through here: an xarray Dataset backed by
Zarr, following sgkit's variant/sample dimension conventions. Preflight checks
that a store meets the contract in core.py before any query touches it.
Nothing here writes a store; import produces one through the reference
converter (see convert.py), and this module only reads.

Both Zarr storage formats are read. The one place they differ for a reader
is where dimension names live: format 2 in the `_ARRAY_DIMENSIONS`
attribute, format 3 in the array metadata.
"""

import json
from pathlib import Path

import xarray as xr
import zarr
from zarr.errors import BaseZarrError, ContainsArrayError, GroupNotFoundError

from acgt import core


def array_dims(array):
    """Dimension names of one zarr array, or None if it has no usable list."""
    dims = array.attrs.get("_ARRAY_DIMENSIONS")
    if dims is None:
        dims = getattr(array.metadata, "dimension_names", None)
    if not isinstance(dims, list | tuple) or not all(isinstance(n, str) for n in dims):
        return None
    return list(dims)


def preflight(path) -> list[str]:
    """Check a store against the contract ACGT depends on.

    Returns the problems found; empty means the store passes. A malformed
    store is reported, not raised, down to a path that is no Zarr group or
    metadata that does not parse. A path that does not exist raises
    FileNotFoundError. Only metadata is read.
    """
    if not Path(path).exists():
        raise FileNotFoundError(path)
    # Consolidated metadata is skipped here, as it is in open_dataset, so
    # both read the same documents. zarr's own errors and JSONDecodeError
    # are ValueErrors; a metadata field of the wrong shape raises TypeError.
    try:
        z = zarr.open_group(path, mode="r", use_consolidated=False)
        arrays = dict(z.arrays())
    except (GroupNotFoundError, ContainsArrayError) as exc:
        return [f"not a Zarr group: {exc}"]
    except (BaseZarrError, json.JSONDecodeError, TypeError) as exc:
        return [f"store metadata does not parse: {exc}"]
    problems = []

    version = z.attrs.get("vcf_zarr_version")
    if version != core.VCF_ZARR_VERSION:
        problems.append(
            f"vcf_zarr_version is {version!r}, expected {core.VCF_ZARR_VERSION!r}"
        )
    problems.extend(
        f"missing mandatory array {k}"
        for k in sorted(core.REQUIRED_ARRAYS - arrays.keys())
    )

    sizes = {}
    for k in sorted(arrays):
        array = arrays[k]
        d = array_dims(array)
        if d is None:
            problems.append(f"{k}: no dimension names")
            continue
        if len(d) != array.ndim:
            problems.append(f"{k}: {len(d)} dimension names for {array.ndim} dims")
            continue
        fixed = core.FIXED_ARRAYS.get(k)
        if fixed is not None:
            want, kinds = fixed
            if list(want) != d:
                problems.append(
                    f"{k}: dimensions {d} do not match the spec's {list(want)}"
                )
            if array.dtype.kind not in kinds:
                problems.append(
                    f"{k}: dtype {array.dtype} is not the kind a query reads"
                )
        for name, size in zip(d, array.shape, strict=True):
            prev = sizes.setdefault(name, size)
            if prev != size:
                problems.append(f"{k}: dim {name!r} is {size}, elsewhere {prev}")

    n_samples = sizes.get(core.DIM_SAMPLES)
    if n_samples is not None and n_samples != 1:
        problems.append(f"{n_samples} samples; ACGT opens single-sample stores only")
    return problems


def open_dataset(path):
    """Open a VCZ store lazily, preflighting it first.

    The Dataset is backed by the store rather than loaded: slice by region,
    never materialize a whole array.
    """
    problems = preflight(path)
    if problems:
        raise ValueError("store fails preflight: " + "; ".join(problems))
    return xr.open_zarr(path, chunks=None, consolidated=False, mask_and_scale=False)
