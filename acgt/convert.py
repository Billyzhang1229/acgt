"""Convert foreign formats to VCZ.

This is the only module where a foreign format exists. Every conversion path
lives here — VCF/BCF from WGS or WES, and consumer array exports (23andMe,
AncestryDNA) — and the library that parses them is imported here and nowhere
else. New source formats get another function in this file, not another
module. Personal genome data, whatever format it arrives in, comes out as a VCZ store
carrying the schema defined in core.py.

The VCF path is a single-sample fast path. bio2zarr, the reference
converter, pays a per-record Python cost that amortises over the samples
dimension, which makes one-sample files its worst case; this converter
instead resolves the store's exact shape up front and fills preallocated
columnar buffers. Two passes over the file: a scan that counts records and
resolves the widths the header leaves undetermined, then a fill. Sentinel
values follow bio2zarr bit for bit, so output is comparable array for array.

Not handled: multiple samples (rejected explicitly — cohort files are
bio2zarr's territory), symbolic and breakend alleles beyond what cyvcf2
reports verbatim.
"""

import dataclasses
import logging
import math
import re
import time
from pathlib import Path

import numpy as np
from cyvcf2 import VCF

from acgt import core, dataset

DEFAULT_STRING_CAP = 48

log = logging.getLogger(__name__)

_DTYPES = {
    core.KIND_INT: "i4",
    core.KIND_FLOAT: "f4",
    core.KIND_BOOL: "b1",
    core.KIND_STR: "O",
    core.KIND_CHAR: "O",  # parsed like a string; dataset.py stores it as U1
}


@dataclasses.dataclass
class _Field:
    """One INFO or FORMAT field: its array spec plus what parsing needs."""

    spec: core.ArraySpec
    category: str
    name: str
    number: str
    width: int = 1

    @property
    def vector(self):
        base = 2 if self.category == "FORMAT" else 1
        return len(self.spec.dims) > base

    @property
    def dtype(self):
        return _DTYPES[self.spec.kind]


# --------------------------------------------------------------------------
# header discovery
# --------------------------------------------------------------------------
def _header_records(vcf):
    for h in vcf.header_iter():
        try:
            yield h.info()
        except Exception as e:  # noqa: BLE001 -- cyvcf2's error types here are undocumented
            log.debug("skipping header line: %s", e)


def _discover(vcf, exclude):
    fields, filters, descriptions = [], [], {}
    has_genotypes = False
    declared = set()
    for d in _header_records(vcf):
        header_type = d.get("HeaderType")
        if header_type in ("INFO", "FORMAT"):
            # INFO and FORMAT are separate namespaces: INFO/DP declared
            # says nothing about FORMAT/DP
            declared.add((header_type, d["ID"]))
        if header_type == "FILTER":
            filters.append(d["ID"])
            descriptions[d["ID"]] = str(d.get("Description", "")).strip('"')
            continue
        if header_type not in ("INFO", "FORMAT"):
            continue
        name = d["ID"]
        if f"{header_type}/{name}" in exclude or name in exclude:
            continue
        if header_type == "FORMAT" and name == "GT":
            has_genotypes = True
            continue  # becomes call_genotype, not a field array
        number = str(d.get("Number", "1")).strip()
        vcf_type = d.get("Type", "String")
        description = str(d.get("Description", "")).strip('"')
        spec = core.spec_for_field(header_type, name, number, vcf_type, description)
        fields.append(_Field(spec, header_type, name, number))
    filters = ["PASS"] + [f for f in filters if f != "PASS"]
    return fields, filters, descriptions, has_genotypes, declared


_META_LINE = re.compile(r"##([^=]+)=(.*)")
# header lines whose content lives in arrays and would be redundant here
_META_IN_ARRAYS = frozenset({"contig", "FILTER", "INFO", "FORMAT"})


def _meta_information(vcf):
    """The header's remaining ##key=value lines as (key, value) pairs.

    This is the spec's optional vcf_meta_information: everything the arrays
    do not already carry, kept so a store can be rendered back to VCF with
    its ##reference and the rest intact. The selection matches the
    reference implementation's.
    """
    pairs = []
    for line in vcf.raw_header.splitlines():
        match = _META_LINE.fullmatch(line)
        if match and match.group(1) not in _META_IN_ARRAYS:
            pairs.append((match.group(1), match.group(2)))
    return pairs


def _reference_build(meta_information):
    return next((value for key, value in meta_information if key == "reference"), "")


# --------------------------------------------------------------------------
# scan pass
# --------------------------------------------------------------------------
def _observed_width(value):
    if isinstance(value, tuple):
        return len(value)
    if isinstance(value, str):
        return value.count(",") + 1
    if hasattr(value, "dtype") and value.dtype.kind in "SU":
        return max(
            (x.decode() if isinstance(x, bytes) else x).count(",") + 1
            for x in value.reshape(-1).tolist()
        )
    if hasattr(value, "shape"):
        return value.shape[-1]
    return 1


def _scan(path, fields, has_genotypes, declared):
    """Count records and observe everything the header leaves open: the
    number of alternate alleles, ploidy, string lengths, and the width of
    every Number=. field.

    Ordering is enforced here too: positions must not decrease within a
    contig and a contig's records must be contiguous. region_index takes a
    chunk's first and last position as its bounds, so unsorted input would
    not fail — it would silently produce region queries that miss variants.
    """
    open_fields = [f for f in fields if f.vector and f.number not in ("A", "R")]
    widths = dict.fromkeys(((f.category, f.name) for f in open_fields), 1)
    n = max_alt = 0
    max_slen = max_idlen = max_ploidy = 1
    current_chrom = None
    current_pos = 0
    seen_chroms = set()
    vcf = VCF(path)
    for v in vcf:
        n += 1
        if current_chrom != v.CHROM:
            if v.CHROM in seen_chroms:
                raise ValueError(
                    f"records for contig {v.CHROM!r} are not contiguous; "
                    f"sort the file and retry"
                )
            seen_chroms.add(v.CHROM)
            current_chrom = v.CHROM
            current_pos = 0
        if current_pos > v.POS:
            raise ValueError(
                f"unsorted input: {v.CHROM}:{v.POS} follows "
                f"{v.CHROM}:{current_pos}; sort the file and retry"
            )
        current_pos = v.POS
        max_alt = max(max_alt, len(v.ALT))
        longest = max((len(a) for a in v.ALT), default=0)
        max_slen = max(max_slen, len(v.REF), longest)
        if v.ID is not None:
            max_idlen = max(max_idlen, len(v.ID))
        if has_genotypes and "GT" in v.FORMAT:
            genotype = v.genotype
            if genotype is not None:
                max_ploidy = max(max_ploidy, genotype.array().shape[1] - 1)
        for f in open_fields:
            x = v.INFO.get(f.name) if f.category == "INFO" else v.format(f.name)
            if x is not None:
                key = (f.category, f.name)
                widths[key] = max(widths[key], _observed_width(x))
    # htslib adds any field the records use without a declaration to its
    # in-memory header as it parses; comparing afterwards is how we learn
    # about them. The reference implementation drops such fields silently,
    # and output stays comparable to it, but not without saying so.
    undeclared = sorted(
        f"{d['HeaderType']}/{d['ID']}"
        for d in _header_records(vcf)
        if d.get("HeaderType") in ("INFO", "FORMAT")
        and (d["HeaderType"], d["ID"]) not in declared
    )
    if undeclared:
        log.warning(
            "records use INFO/FORMAT fields the header never declared, "
            "and they are not converted: %s",
            ", ".join(undeclared),
        )
    return n, max_alt, widths, max_slen, max_idlen, max_ploidy


def _resolve_widths(fields, n_alleles, max_genotypes, observed):
    """Turn each field's VCF Number into a concrete trailing dimension,
    sized so that fields sharing a reserved dimension agree by construction."""
    for f in fields:
        if f.number == "A":
            f.width = n_alleles - 1
        elif f.number == "R":
            f.width = n_alleles
        elif f.number == "G":
            f.width = max_genotypes
        elif f.number.lstrip("-").isdigit():
            f.width = max(1, int(f.number))
        else:
            f.width = observed.get((f.category, f.name), 1)
        if f.spec.kind == core.KIND_BOOL:
            f.width = 1


# --------------------------------------------------------------------------
# fill pass
# --------------------------------------------------------------------------
def _allocate(
    n, n_alleles, ploidy, n_filters, n_contigs, fields, slen, idlen, cap, has_genotypes
):
    contig_dtype = "i2" if n_contigs <= np.iinfo("i2").max else "i4"
    arrays = {
        "variant_position": np.empty((n,), "i4"),
        "variant_contig": np.empty((n,), contig_dtype),
        "variant_quality": np.empty((n,), "f4"),
        "variant_id": np.empty((n,), f"S{min(idlen, cap)}"),
        "variant_allele": np.empty((n, n_alleles), f"S{min(slen, cap)}"),
        "variant_length": np.empty((n,), "i4"),
        "variant_filter": np.zeros((n, max(1, n_filters)), "b1"),
    }
    if has_genotypes:
        gt_dtype = "i1" if n_alleles <= np.iinfo("i1").max else "i2"
        arrays["call_genotype"] = np.empty((n, 1, ploidy), gt_dtype)
        arrays["call_genotype_phased"] = np.empty((n, 1), "b1")
    for f in fields:
        shape = (n,) if f.category == "INFO" else (n, 1)
        if f.vector:
            shape = (*shape, f.width)
        dtype = f"S{cap}" if f.dtype == "O" else f.dtype
        arrays[f.spec.name] = np.empty(shape, dtype)
    return arrays


def _fill(path, arrays, fields, contig_index, filter_index, n_alleles, ploidy, cap):
    """One pass writing every record into the preallocated buffers.

    The per-field work is flattened into (array, kind, ...) tuples so the
    record loop does no attribute lookups; strings longer than the inline
    cap go to the spill dict instead of widening every row.
    """
    INFO_FLAG, INFO_SCALAR, INFO_VECTOR, FMT_SCALAR, FMT_VECTOR = range(5)
    ops = []
    for f in fields:
        if f.category == "INFO":
            kind = (
                INFO_FLAG
                if f.spec.kind == core.KIND_BOOL
                else INFO_VECTOR
                if f.vector
                else INFO_SCALAR
            )
        else:
            kind = FMT_VECTOR if f.vector else FMT_SCALAR
        ops.append((arrays[f.spec.name], kind, f.name, f.width, f.dtype, f.spec.name))

    spill = {}
    pos = arrays["variant_position"]
    contig = arrays["variant_contig"]
    qual = arrays["variant_quality"]
    vid = arrays["variant_id"]
    allele = arrays["variant_allele"]
    vlen = arrays["variant_length"]
    vfilter = arrays["variant_filter"]
    gt = arrays.get("call_genotype")  # absent when the file declares no GT
    phased = arrays.get("call_genotype_phased")

    def put_str(arr, arr_name, index, value):
        if len(value) > cap:
            spill[(arr_name, index)] = value
            arr[index] = b""
        else:
            arr[index] = value.encode()

    for i, v in enumerate(VCF(path)):
        pos[i] = v.POS
        try:
            contig[i] = contig_index[v.CHROM]
        except KeyError:
            raise ValueError(
                f"record contig {v.CHROM!r} is not declared in the header"
            ) from None
        q = v.QUAL
        qual[i] = core.FLOAT32_MISSING if q is None else q
        put_str(vid, "variant_id", i, v.ID or core.STR_MISSING)
        put_str(allele, "variant_allele", (i, 0), v.REF)
        k = 1
        for a in v.ALT:
            if k < n_alleles:
                put_str(allele, "variant_allele", (i, k), a)
                k += 1
        allele[i, k:] = b""
        vlen[i] = v.end - v.start
        # FILTERS, not FILTER: the plural distinguishes "." (no entry set)
        # from PASS, which the scalar property conflates.
        for name in v.FILTERS:
            j = filter_index.get(name)
            if j is None:
                # dropping it silently would lose data; the reference
                # implementation rejects such files too
                raise ValueError(
                    f"record filter {name!r} is not declared in the header"
                )
            vfilter[i, j] = True
        if gt is not None:
            # a record may legitimately omit GT even when the header
            # declares it; cyvcf2 raises on v.genotype then, so check the
            # record's own FORMAT first, the way bio2zarr does. A call
            # that is not there is missing, matching the reference.
            genotype = v.genotype if "GT" in v.FORMAT else None
            if genotype is None:
                gt[i, 0, :] = core.INT_MISSING
                phased[i, 0] = False
            else:
                # genotype.array() rather than genotypes: the phase bit it
                # reports is the same whether the record came from VCF text
                # or BCF, which the genotypes list's is not -- and it is
                # what bio2zarr reads
                row = genotype.array()[0]
                p = row.shape[0] - 1
                if ploidy == 2 and p == 2:  # diploid fast path
                    gt[i, 0, 0] = row[0]
                    gt[i, 0, 1] = row[1]
                else:
                    for kk in range(p):
                        gt[i, 0, kk] = row[kk]
                    gt[i, 0, p:] = core.INT_FILL
                phased[i, 0] = bool(row[-1])

        info = v.INFO
        for arr, kind, name, width, dtype, arr_name in ops:
            if kind == INFO_FLAG:
                arr[i] = info.get(name) is not None
            elif kind == INFO_SCALAR:
                x = info.get(name)
                if x is None:
                    arr[i] = (
                        core.INT_MISSING
                        if dtype == "i4"
                        else core.FLOAT32_MISSING
                        if dtype == "f4"
                        else b"."
                    )
                else:
                    y = x[0] if type(x) is tuple else x
                    if y is None:  # "." inside a present field
                        arr[i] = (
                            core.INT_MISSING
                            if dtype == "i4"
                            else core.FLOAT32_MISSING
                            if dtype == "f4"
                            else b"."
                        )
                    elif dtype != "O":
                        arr[i] = y
                    else:
                        put_str(arr, arr_name, (i,), str(y))
            elif kind == INFO_VECTOR:
                x = info.get(name)
                is_str = dtype == "O"
                if x is None:
                    # a missing field is a whole row of MISSING, not
                    # MISSING then FILL -- matching bio2zarr
                    arr[i] = (
                        b"."
                        if is_str
                        else core.INT_MISSING
                        if dtype == "i4"
                        else core.FLOAT32_MISSING
                    )
                else:
                    xs = x if type(x) is tuple else (x,)
                    if is_str and type(x) is str:
                        xs = tuple(x.split(","))
                    m = len(xs)
                    pad = (
                        b""
                        if is_str
                        else core.INT_FILL
                        if dtype == "i4"
                        else core.FLOAT32_FILL
                    )
                    # "." inside a present vector is MISSING; only the
                    # positions past the value's own length are FILL
                    absent = core.INT_MISSING if dtype == "i4" else core.FLOAT32_MISSING
                    for j in range(width):
                        if j >= m:
                            arr[i, j] = pad
                        elif is_str:
                            put_str(arr, arr_name, (i, j), str(xs[j]))
                        elif xs[j] is None:
                            arr[i, j] = absent
                        else:
                            arr[i, j] = xs[j]
            elif kind == FMT_SCALAR:
                x = v.format(name)
                if x is None:
                    arr[i, 0] = (
                        b"."
                        if dtype == "O"
                        else core.INT_MISSING
                        if dtype == "i4"
                        else core.FLOAT32_MISSING
                    )
                elif dtype == "O":
                    y = x[0]
                    put_str(
                        arr,
                        arr_name,
                        (i, 0),
                        y.decode() if isinstance(y, bytes) else str(y),
                    )
                else:
                    arr[i, 0] = x[0, 0]
            else:  # FMT_VECTOR
                x = v.format(name)
                if x is None:
                    arr[i, 0] = (
                        b"."
                        if dtype == "O"
                        else core.INT_MISSING
                        if dtype == "i4"
                        else core.FLOAT32_MISSING
                    )
                elif dtype == "O":
                    y = x[0]
                    parts = (y.decode() if isinstance(y, bytes) else str(y)).split(",")
                    for j in range(width):
                        if j < len(parts):
                            put_str(arr, arr_name, (i, 0, j), parts[j])
                        else:
                            arr[i, 0, j] = b""
                else:
                    row = x[0]
                    m = row.shape[0]
                    if m >= width:
                        arr[i, 0, :] = row[:width]
                    else:
                        arr[i, 0, :m] = row
                        arr[i, 0, m:] = (
                            core.INT_FILL if dtype == "i4" else core.FLOAT32_FILL
                        )
    return spill


_VCF_INT_MISSING = np.iinfo(np.int32).min
_VCF_INT_FILL = np.iinfo(np.int32).min + 1


def _remap_sentinels(arrays, fields):
    """htslib's in-band int sentinels become the spec's, vectorised once per
    array -- doing this per record is a large part of what makes the
    reference implementation slow at one sample."""
    for f in fields:
        if f.dtype != "i4":
            continue
        a = arrays[f.spec.name]
        np.putmask(a, a == _VCF_INT_MISSING, core.INT_MISSING)
        np.putmask(a, a == _VCF_INT_FILL, core.INT_FILL)


def _bytes_array(values):
    width = max(1, max(map(len, values), default=1))
    return np.array(values, dtype=f"S{width}")


# --------------------------------------------------------------------------
# public conversion paths
# --------------------------------------------------------------------------
def from_vcf(
    path,
    out,
    *,
    zarr_format=2,
    chunk_size=dataset.DEFAULT_CHUNK_SIZE,
    exclude=(),
    string_cap=DEFAULT_STRING_CAP,
    compress_level=5,
    overwrite=False,
) -> Path:
    """Convert a single-sample VCF or BCF at `path` into a VCZ store at `out`.

    `exclude` drops INFO/FORMAT fields by name ("DP") or qualified name
    ("FORMAT/DP"). Strings longer than `string_cap` are stored out of line,
    so the cap bounds buffer width without truncating anything. The store is
    written in Zarr format 2 unless `zarr_format=3` is asked for; everything
    about that choice lives in dataset.py. An existing `out` is refused
    unless `overwrite` says otherwise — checked here, before the scan, so a
    long conversion cannot fail at the very end on it.
    """
    t0 = time.perf_counter()
    if Path(out).exists() and not overwrite:
        raise ValueError(f"{out} already exists; pass overwrite=True to replace it")
    vcf = VCF(path)
    if len(vcf.samples) != 1:
        raise ValueError(
            f"from_vcf converts single-sample files, got {len(vcf.samples)} "
            f"samples; cohort files are bio2zarr's territory"
        )
    fields, filters, descriptions, has_genotypes, declared = _discover(vcf, exclude)
    contigs = list(vcf.seqnames)
    if not contigs:
        raise ValueError(
            "the header declares no contigs; add ##contig lines "
            "(bcftools reheader --fai) and retry"
        )
    samples = list(vcf.samples)
    meta_information = _meta_information(vcf)
    build = _reference_build(meta_information)
    try:
        contig_lengths = list(vcf.seqlens)
    except AttributeError:
        contig_lengths = []

    n, max_alt, observed, slen, idlen, ploidy = _scan(
        path, fields, has_genotypes, declared
    )
    n_alleles = max(2, max_alt + 1)
    # the genotypes dimension is the multiset coefficient the reference uses,
    # but GT-derived ploidy undercounts records carrying PL without GT, so
    # the widest Number=G value observed sets a floor
    widest_g = max(
        (observed.get((f.category, f.name), 1) for f in fields if f.number == "G"),
        default=1,
    )
    formula = math.comb(n_alleles + ploidy - 1, ploidy) if has_genotypes else 1
    max_genotypes = max(formula, widest_g)
    _resolve_widths(fields, n_alleles, max_genotypes, observed)
    log.info("scanned %s: %d variants, %d fields", path, n, len(fields))

    arrays = _allocate(
        n,
        n_alleles,
        ploidy,
        len(filters),
        len(contigs),
        fields,
        slen,
        idlen,
        string_cap,
        has_genotypes,
    )
    contig_index = {c: i for i, c in enumerate(contigs)}
    filter_index = {f: i for i, f in enumerate(filters)}
    spill = _fill(
        path,
        arrays,
        fields,
        contig_index,
        filter_index,
        n_alleles,
        ploidy,
        string_cap,
    )
    _remap_sentinels(arrays, fields)
    if "call_genotype" in arrays:
        if ploidy == 1:
            # bio2zarr's rule, kept for array-for-array parity: a store
            # whose every call is haploid marks all of them phased
            arrays["call_genotype_phased"][:] = True
        arrays["call_genotype_mask"] = arrays["call_genotype"] < 0
    arrays["variant_id_mask"] = arrays["variant_id"] == b"."
    arrays["sample_id"] = _bytes_array(samples)
    arrays["contig_id"] = _bytes_array(contigs)
    # cyvcf2 reports an unknown contig length as -1; contig_length is an
    # optional array, so any unknown omits it, the way the reference does
    if len(contig_lengths) == len(contigs) and min(contig_lengths, default=0) > 0:
        arrays["contig_length"] = np.array(contig_lengths, dtype="i8")
    arrays["filter_id"] = _bytes_array(filters)
    arrays["filter_description"] = _bytes_array(
        [descriptions.get(f, "") for f in filters]
    )

    dims = {
        core.DIM_VARIANTS: n,
        core.DIM_SAMPLES: 1,
        core.DIM_ALLELES: n_alleles,
        core.DIM_FILTERS: len(filters),
        core.DIM_CONTIGS: len(contigs),
    }
    if "call_genotype" in arrays:
        dims[core.DIM_PLOIDY] = ploidy
    for f in fields:
        for dim in f.spec.dims:
            if dim == core.DIM_ALT_ALLELES:
                dims[dim] = n_alleles - 1
            elif dim == core.DIM_GENOTYPES:
                dims[dim] = max_genotypes
            elif dim not in dims:
                dims[dim] = f.width
    fixed = tuple(spec for name, spec in core.FIXED_ARRAYS.items() if name in arrays)
    schema = core.Schema(
        dims=dims,
        fields=fixed + tuple(f.spec for f in fields),
        source=Path(path).name,
        build=build,
    )

    dataset.save(
        arrays,
        schema,
        out,
        zarr_format=zarr_format,
        chunk_size=chunk_size,
        compress_level=compress_level,
        spill=spill,
        overwrite=overwrite,
        meta_information=meta_information,
    )
    log.info("converted %s -> %s in %.2fs", path, out, time.perf_counter() - t0)
    return Path(out)
