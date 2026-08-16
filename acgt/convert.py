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
resolves the widths the header leaves undetermined, then a fill. Both run
in parallel over genomic windows when the file is indexed and large enough
for that to pay; the exact per-window counts from the scan give each fill
worker its own disjoint slice of shared-memory buffers. Sentinel values
follow bio2zarr bit for bit, so output is comparable array for array.

Not handled: multiple samples (rejected explicitly — cohort files are
bio2zarr's territory), symbolic and breakend alleles beyond what cyvcf2
reports verbatim.
"""

import concurrent.futures
import contextlib
import dataclasses
import gzip
import logging
import math
import multiprocessing
import os
import re
import struct
import time
import warnings
from multiprocessing import shared_memory
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


def _open_fields(fields):
    return [f for f in fields if f.vector and f.number not in ("A", "R")]


def _scan_records(records, open_fields, has_genotypes):
    """Count records and observe everything the header leaves open: the
    number of alternate alleles, ploidy, string lengths, and the width of
    every Number=. field.

    Ordering is enforced here too: positions must not decrease within a
    contig and a contig's records must be contiguous. region_index takes a
    chunk's first and last position as its bounds, so unsorted input would
    not fail — it would silently produce region queries that miss variants.
    """
    widths = dict.fromkeys(((f.category, f.name) for f in open_fields), 1)
    n = max_alt = 0
    max_slen = max_idlen = max_ploidy = 1
    current_chrom = None
    current_pos = 0
    seen_chroms = set()
    for v in records:
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
    return n, max_alt, widths, max_slen, max_idlen, max_ploidy


def _undeclared_fields(vcf, declared):
    """htslib adds any field the records use without a declaration to its
    in-memory header as it parses; comparing afterwards is how we learn
    about them. Only meaningful after the handle has read records."""
    return {
        f"{d['HeaderType']}/{d['ID']}"
        for d in _header_records(vcf)
        if d.get("HeaderType") in ("INFO", "FORMAT")
        and (d["HeaderType"], d["ID"]) not in declared
    }


def _warn_undeclared(undeclared):
    # The reference implementation drops such fields silently, and output
    # stays comparable to it, but not without saying so.
    if undeclared:
        log.warning(
            "records use INFO/FORMAT fields the header never declared, "
            "and they are not converted: %s",
            ", ".join(sorted(undeclared)),
        )


def _scan(path, fields, has_genotypes, declared):
    """The scan pass over a whole file, serially."""
    vcf = VCF(path)
    stats = _scan_records(vcf, _open_fields(fields), has_genotypes)
    _warn_undeclared(_undeclared_fields(vcf, declared))
    return stats


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
def _numpy_empty(name, shape, dtype):
    return np.empty(shape, dtype)


def _allocate(
    n,
    n_alleles,
    ploidy,
    n_filters,
    n_contigs,
    fields,
    slen,
    idlen,
    cap,
    has_genotypes,
    empty=None,
):
    """Preallocate every per-variant buffer at its final shape. `empty` is
    np.empty for the serial path and a shared-memory allocator for the
    parallel one; either way the arrays come back as plain ndarrays."""
    empty = empty or _numpy_empty
    contig_dtype = "i2" if n_contigs <= np.iinfo("i2").max else "i4"
    arrays = {
        "variant_position": empty("variant_position", (n,), "i4"),
        "variant_contig": empty("variant_contig", (n,), contig_dtype),
        "variant_quality": empty("variant_quality", (n,), "f4"),
        "variant_id": empty("variant_id", (n,), f"S{min(idlen, cap)}"),
        "variant_allele": empty("variant_allele", (n, n_alleles), f"S{min(slen, cap)}"),
        "variant_length": empty("variant_length", (n,), "i4"),
        "variant_filter": empty("variant_filter", (n, max(1, n_filters)), "b1"),
    }
    arrays["variant_filter"][:] = False
    if has_genotypes:
        gt_dtype = "i1" if n_alleles <= np.iinfo("i1").max else "i2"
        arrays["call_genotype"] = empty("call_genotype", (n, 1, ploidy), gt_dtype)
        arrays["call_genotype_phased"] = empty("call_genotype_phased", (n, 1), "b1")
    for f in fields:
        shape = (n,) if f.category == "INFO" else (n, 1)
        if f.vector:
            shape = (*shape, f.width)
        dtype = f"S{cap}" if f.dtype == "O" else f.dtype
        arrays[f.spec.name] = empty(f.spec.name, shape, dtype)
    return arrays


def _fill(
    records, start, arrays, fields, contig_index, filter_index, n_alleles, ploidy, cap
):
    """Write `records` into the preallocated buffers from row `start` on.

    The per-field work is flattened into (array, kind, ...) tuples so the
    record loop does no attribute lookups; strings longer than the inline
    cap go to the spill dict instead of widening every row. Returns the
    number of records written and the spill dict, keyed by (array name,
    absolute index tuple) as dataset.save expects.
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

    i = start - 1
    for v in records:
        i += 1
        pos[i] = v.POS
        try:
            contig[i] = contig_index[v.CHROM]
        except KeyError:
            raise ValueError(
                f"record contig {v.CHROM!r} is not declared in the header"
            ) from None
        q = v.QUAL
        qual[i] = core.FLOAT32_MISSING if q is None else q
        # index tuples throughout: the writer patches spill by idx[0]
        put_str(vid, "variant_id", (i,), v.ID or core.STR_MISSING)
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
    return i + 1 - start, spill


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
# parallel scan and fill
# --------------------------------------------------------------------------
# The two passes are embarrassingly parallel once the file is cut into
# genomic windows: the scan reduces per-window counts and maxima, and the
# exact per-window counts give every fill worker its own disjoint slice of
# the preallocated buffers to write into. Workers hold the buffers in
# shared memory, so nothing but the small spill dict crosses a process
# boundary. This needs an index (.tbi or .csi) to seek by region; without
# one the serial path runs.

PARTITION_WINDOW = 10_000_000  # bases per partition
PARALLEL_MIN_BYTES = 2**20  # below ~1 MiB, process start-up outweighs the gain
_HTS_MAX_POS = 2**31 - 1


def _index_file(path):
    """The index the parallel path may seek with, or None.

    Only an index that demonstrably describes the current file is used:
    windows read through an index left over from an earlier version of the
    file see the records it points at and silently miss the rest -- both
    passes use the same index, so their counts agree and the fill-pass check
    cannot notice. Each candidate goes through _index_mismatch; a rejected
    one is logged and the next suffix tried, so a stale .tbi does not hide a
    good .csi. Nothing usable means the serial path, which reads every
    record regardless of any index.
    """
    path = Path(path)
    for suffix in (".tbi", ".csi"):
        candidate = Path(f"{path}{suffix}")
        if not candidate.exists():
            continue
        why = _index_mismatch(path, candidate)
        if why is None:
            return candidate
        log.warning(
            "ignoring %s: %s -- rebuild it (tabix/bcftools index) to convert "
            "in parallel",
            candidate.name,
            why,
        )
    return None


def _index_mismatch(data, index):
    """Why `index` cannot be trusted to describe `data`, or None if it can.

    Two tests. Modification time is the cheap one: an index older than its
    data file was built for something else. It is not identity, though -- a
    copied index is newer than the file it does not describe -- so the index
    is also checked against the file's content. Every chunk in a tabix or
    CSI index ends at a BGZF virtual offset, and the largest of them is
    where the last indexed record ends; in the file the index describes,
    nothing but empty blocks follows that point. If the offset is not a
    block boundary in this file, or uncompressed data remains after it, the
    file has changed since the index was built. This catches appended,
    truncated, and re-compressed files; it cannot tell apart two files of
    identical block layout that differ only in content, which no check
    short of re-reading the file can.
    """
    if index.stat().st_mtime_ns < data.stat().st_mtime_ns:
        return f"it is older than {data.name}"
    try:
        _, end = _read_index(index)
    except (OSError, EOFError, struct.error) as e:
        return f"it does not parse ({e})"
    if end is None:
        return "it is not a tabix or CSI index"
    coff, uoff = end >> 16, end & 0xFFFF
    with data.open("rb") as fh:
        size = fh.seek(0, os.SEEK_END)
        if coff >= size:
            return f"it points past the end of {data.name}"
        block = _bgzf_block(fh, coff)
        if block is None or uoff > block[1]:
            return f"its last record offset is not a block boundary in {data.name}"
        pos, trailing = coff + block[0], block[1] - uoff
        while pos < size and not trailing:
            block = _bgzf_block(fh, pos)
            if block is None:
                return f"{data.name} is not BGZF after the last indexed record"
            pos, trailing = pos + block[0], block[1]
    if trailing:
        return f"{data.name} has data after the last record it indexes"
    return None


def _bgzf_block(fh, offset):
    """(compressed size, uncompressed size) of the BGZF block at `offset`,
    or None when no valid block header starts there."""
    fh.seek(offset)
    header = fh.read(12)
    if len(header) < 12 or header[:4] != b"\x1f\x8b\x08\x04":
        return None
    (xlen,) = struct.unpack("<H", header[10:12])
    extra = fh.read(xlen)
    if len(extra) < xlen:
        return None
    i, bsize = 0, None
    while i + 4 <= xlen:
        si1, si2, slen = (
            extra[i],
            extra[i + 1],
            struct.unpack("<H", extra[i + 2 : i + 4])[0],
        )
        if si1 == 66 and si2 == 67 and slen == 2:
            (bsize,) = struct.unpack("<H", extra[i + 4 : i + 6])
            break
        i += 4 + slen
    if bsize is None:
        return None
    total = bsize + 1
    fh.seek(offset + total - 4)
    tail = fh.read(4)
    if len(tail) < 4:
        return None
    (isize,) = struct.unpack("<I", tail)
    return total, isize


def _read_index(index):
    """(contig names or None, largest chunk end offset or None) from a
    tabix or CSI index; (None, None) for anything else.

    tabix writes contig names into both .tbi and .csi; bcftools' CSI for a
    BCF does not, since BCF records name contigs by header position and
    cannot mention an undeclared one. The pseudo-bin htslib may add per
    contig holds counts, not offsets, and is skipped.
    """
    with gzip.open(index, "rb") as fh:
        read = fh.read

        def ints(n):
            return struct.unpack(f"<{n}i", read(4 * n))

        magic = read(4)
        if magic == b"TBI\1":
            (n_ref, *_) = ints(
                7
            )  # n_ref, format, col_seq, col_beg, col_end, meta, skip
            (l_nm,) = ints(1)
            names = [x.decode() for x in read(l_nm).split(b"\0") if x]
            depth, csi = 5, False
        elif magic == b"CSI\1":
            _, depth, l_aux = ints(3)
            aux = read(l_aux)
            names = None
            if l_aux >= 4 * 7:
                (l_nm,) = struct.unpack("<i", aux[24:28])
                names = [x.decode() for x in aux[28 : 28 + l_nm].split(b"\0") if x]
            (n_ref,) = ints(1)
            csi = True
        else:
            return None, None
        meta_bin = ((1 << (3 * depth + 3)) - 1) // 7 + 1
        end = 0
        for _ in range(n_ref):
            (n_bin,) = ints(1)
            for _ in range(n_bin):
                (bin_id,) = struct.unpack("<I", read(4))
                if csi:
                    read(8)  # loff
                (n_chunk,) = ints(1)
                chunks = struct.unpack(f"<{2 * n_chunk}Q", read(16 * n_chunk))
                if bin_id != meta_bin:
                    end = max(end, *chunks[1::2], 0)
            if not csi:
                (n_intv,) = ints(1)
                read(8 * n_intv)
        return names, end


def _index_contigs(index):
    """The contig names an index carries, in the order they occur in the
    file, or None when it carries none."""
    return _read_index(index)[0]


def _partitions(contigs, lengths, window=None):
    """(contig, start, end) windows covering every listed contig, bounds
    inclusive. The last window of a contig runs to the end of the coordinate
    space rather than to the header length, so a wrong or missing length
    cannot truncate the contig."""
    window = window or PARTITION_WINDOW
    parts = []
    for name, length in zip(contigs, lengths, strict=True):
        starts = range(1, max(length, 1) + 1, window)
        for start in starts:
            end = _HTS_MAX_POS if start == starts[-1] else start + window - 1
            parts.append((name, start, end))
    return parts


def _region(contig, start, end):
    # braces keep htslib from misreading contig names that contain ':'
    return f"{{{contig}}}:{start}-{end}"


def _bounded(records, start, end):
    """Records whose POS lies in [start, end]. A region query returns every
    record *overlapping* the window, so a deletion that begins before it
    would show up again in the next window; keying on POS assigns each
    record to exactly one partition."""
    for v in records:
        if start > v.POS:
            continue
        if end < v.POS:
            break
        yield v


class _SharedBuffers:
    """Named shared-memory blocks the parent allocates and workers attach to.

    `empty` has the signature `_allocate` expects of its allocator; `meta`
    is what a worker needs to map the same blocks by name.
    """

    def __init__(self):
        self.blocks = []
        self.meta = {}

    def empty(self, name, shape, dtype):
        dtype = np.dtype(dtype)
        nbytes = max(1, dtype.itemsize * math.prod(shape))
        shm = shared_memory.SharedMemory(create=True, size=nbytes)
        self.blocks.append(shm)
        self.meta[name] = (shm.name, tuple(shape), dtype.str)
        return np.ndarray(shape, dtype, buffer=shm.buf)

    def close(self):
        # unlink first: the memory stays mapped until every view is gone, and
        # what matters is that the name does not outlive the conversion
        for shm in self.blocks:
            with contextlib.suppress(FileNotFoundError):
                shm.unlink()
            with contextlib.suppress(BufferError):
                shm.close()


# Per-worker state, set once by the pool initializer and reused across
# every partition the worker handles: one cyvcf2 handle, and the shared
# buffers once the fill pass has told the worker their names.
_worker = {}


def _worker_init(
    path, fields, has_genotypes, declared, contig_index, filter_index, cap
):
    warnings.filterwarnings("ignore", message=".*no intervals found.*")
    _worker.update(
        vcf=VCF(path),
        fields=fields,
        open_fields=_open_fields(fields),
        has_genotypes=has_genotypes,
        declared=declared,
        contig_index=contig_index,
        filter_index=filter_index,
        cap=cap,
        handles=[],
        arrays=None,
    )


def _worker_arrays(meta):
    if _worker["arrays"] is None:
        arrays = {}
        for name, (shm_name, shape, dtype) in meta.items():
            shm = shared_memory.SharedMemory(name=shm_name, track=False)
            _worker["handles"].append(shm)  # keeps the mapping alive
            arrays[name] = np.ndarray(shape, np.dtype(dtype), buffer=shm.buf)
        _worker["arrays"] = arrays
    return _worker["arrays"]


def _scan_partition(part):
    contig, start, end = part
    vcf = _worker["vcf"]
    records = _bounded(vcf(_region(contig, start, end)), start, end)
    stats = _scan_records(records, _worker["open_fields"], _worker["has_genotypes"])
    return stats, _undeclared_fields(vcf, _worker["declared"])


def _fill_partition(job):
    part, lo, expected, meta, widths, n_alleles, ploidy = job
    contig, start, end = part
    vcf = _worker["vcf"]
    # the worker's copy of the fields predates the scan; the widths the scan
    # resolved arrive with the job
    for f, width in zip(_worker["fields"], widths, strict=True):
        f.width = width
    records = _bounded(vcf(_region(contig, start, end)), start, end)
    written, spill = _fill(
        records,
        lo,
        _worker_arrays(meta),
        _worker["fields"],
        _worker["contig_index"],
        _worker["filter_index"],
        n_alleles,
        ploidy,
        _worker["cap"],
    )
    if written != expected:
        # the scan sized this slice; writing past it would overwrite the
        # next partition's rows silently
        raise ValueError(
            f"{contig}:{start}-{end} yielded {written} records on the fill "
            f"pass but {expected} on the scan; the file changed underneath us"
        )
    return spill


def _plan_parallel(path, workers, contigs, contig_lengths):
    """Partitions for the parallel path, or None when the serial path should
    run: one worker asked for, no usable index to seek with (none, or one
    that fails _index_mismatch), or -- with `workers` left to default -- a file
    small enough that process start-up would outweigh the gain. An explicit
    `workers` forces the parallel path when an index allows it."""
    if workers == 1:
        return None
    index = _index_file(path)
    if index is None:
        return None
    if workers is None and Path(path).stat().st_size < PARALLEL_MIN_BYTES:
        return None
    lengths = dict(zip(contigs, contig_lengths, strict=False))
    ordered = _index_contigs(index)
    if ordered is None:
        # BCF: the index names nothing, so windows follow the header. A file
        # whose contigs are stored in another order still converts; only the
        # row order differs from the serial path's.
        ordered = contigs
    else:
        unknown = [c for c in ordered if c not in lengths]
        if unknown:
            raise ValueError(
                f"record contig {unknown[0]!r} is not declared in the header"
            )
    return _partitions(ordered, [lengths.get(c, -1) for c in ordered])


def _reduce_scan(results):
    n = max_alt = 0
    max_slen = max_idlen = max_ploidy = 1
    widths = {}
    undeclared = set()
    for (pn, palt, pw, pslen, pidlen, pploidy), pundeclared in results:
        n += pn
        max_alt = max(max_alt, palt)
        max_slen = max(max_slen, pslen)
        max_idlen = max(max_idlen, pidlen)
        max_ploidy = max(max_ploidy, pploidy)
        for k, w in pw.items():
            widths[k] = max(widths.get(k, 1), w)
        undeclared |= pundeclared
    return (n, max_alt, widths, max_slen, max_idlen, max_ploidy), undeclared


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
    workers=None,
) -> Path:
    """Convert a single-sample VCF or BCF at `path` into a VCZ store at `out`.

    `exclude` drops INFO/FORMAT fields by name ("DP") or qualified name
    ("FORMAT/DP"). Strings longer than `string_cap` are stored out of line,
    so the cap bounds buffer width without truncating anything. The store is
    written in Zarr format 2 unless `zarr_format=3` is asked for; everything
    about that choice lives in dataset.py. An existing `out` is refused
    unless `overwrite` says otherwise — checked here, before the scan, so a
    long conversion cannot fail at the very end on it.

    Both passes run in parallel over genomic windows when the file has a
    tabix or CSI index that checks out as describing this file (see
    _index_mismatch) and is large enough for that to pay off; `workers`
    sets the process count (default: up to 8, one per core), 1 forces the
    serial path, and any other explicit value forces the parallel one. The
    output is the same either way for a sorted file; see _plan_parallel for
    the one BCF ordering caveat.
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

    partitions = _plan_parallel(path, workers, contigs, contig_lengths)
    contig_index = {c: i for i, c in enumerate(contigs)}
    filter_index = {f: i for i, f in enumerate(filters)}
    shared = None
    pool = None
    try:
        if partitions is None:
            stats = _scan(path, fields, has_genotypes, declared)
        else:
            pool = concurrent.futures.ProcessPoolExecutor(
                workers or min(8, os.process_cpu_count() or 1),
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_worker_init,
                initargs=(
                    path,
                    fields,
                    has_genotypes,
                    declared,
                    contig_index,
                    filter_index,
                    string_cap,
                ),
            )
            per_partition = list(pool.map(_scan_partition, partitions, chunksize=4))
            stats, undeclared = _reduce_scan(per_partition)
            _warn_undeclared(undeclared)
            counts = [st[0] for st, _ in per_partition]
            shared = _SharedBuffers()
        n, max_alt, observed, slen, idlen, ploidy = stats
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
            empty=shared.empty if shared is not None else _numpy_empty,
        )
        if pool is None or shared is None:  # both set together: the serial path
            _, spill = _fill(
                VCF(path),
                0,
                arrays,
                fields,
                contig_index,
                filter_index,
                n_alleles,
                ploidy,
                string_cap,
            )
        else:
            offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
            widths = [f.width for f in fields]
            jobs = [
                (part, int(lo), int(count), shared.meta, widths, n_alleles, ploidy)
                for part, lo, count in zip(partitions, offsets, counts, strict=True)
                if count
            ]
            spill = {}
            for partial in pool.map(_fill_partition, jobs, chunksize=2):
                spill.update(partial)
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
        # contig_length is optional. cyvcf2's seqlens raises when the header
        # declares contigs without lengths, and the array is omitted then;
        # when it answers, the array is written as is, including the -1 it
        # reports for a contig the index named but the header did not --
        # the reference implementation's rule exactly
        if len(contig_lengths) == len(contigs):
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
        fixed = tuple(
            spec for name, spec in core.FIXED_ARRAYS.items() if name in arrays
        )
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
    finally:
        if pool is not None:
            pool.shutdown()
        if shared is not None:
            shared.close()
