"""What an ACGT dataset is — the contract every store must meet.

This module describes the data model and nothing else: the dimension names,
the arrays a store must carry and how they are laid out, and the sentinel
values that mark missing and fill entries. No I/O and no computation live
here; opening belongs to dataset.py, queries to query.py.

Stores are written by the reference converter, so this is a reader's view of
the spec: only what ACGT itself depends on when it opens and queries a store.
The layout follows VCF Zarr 0.5 (github.com/sgkit-dev/vcf-zarr-spec); the
sentinel values match vcztools and bio2zarr bit for bit, and a test holds
that equality in place.
"""

import numpy as np

# Version of the VCF Zarr spec the store layout follows, written to the
# store's `vcf_zarr_version` group attribute.
VCF_ZARR_VERSION = "0.5"


# --------------------------------------------------------------------------
# Dimensions
# --------------------------------------------------------------------------
DIM_VARIANTS = "variants"
DIM_SAMPLES = "samples"
DIM_PLOIDY = "ploidy"
DIM_ALLELES = "alleles"
DIM_ALT_ALLELES = "alt_alleles"
DIM_GENOTYPES = "genotypes"
DIM_CONTIGS = "contigs"
DIM_FILTERS = "filters"
DIM_REGION_INDEX_VALUES = "region_index_values"
DIM_REGION_INDEX_FIELDS = "region_index_fields"

# Dimension names the spec reserves; a store must not size the same reserved
# name inconsistently across arrays. "parents" is reserved for pedigree data
# the spec defines but ACGT does not write.
RESERVED_DIMS = frozenset(
    {
        DIM_VARIANTS,
        DIM_SAMPLES,
        DIM_PLOIDY,
        DIM_ALLELES,
        DIM_ALT_ALLELES,
        DIM_GENOTYPES,
        DIM_CONTIGS,
        DIM_FILTERS,
        DIM_REGION_INDEX_VALUES,
        DIM_REGION_INDEX_FIELDS,
        "parents",
    }
)


# --------------------------------------------------------------------------
# Sentinels
# --------------------------------------------------------------------------
# Integer and string sentinels are the spec's literal values. The float
# sentinels are specific quiet-NaN bit patterns, not values, so they are
# defined by their bits and must be compared by their bits: any NaN compares
# unequal to itself, and an ordinary NaN is not a sentinel. Each float width
# has its own pattern — exponent all ones, low mantissa bits 1 (missing) and
# 2 (fill) — and they do not survive casting between widths, so the value for
# the width actually stored is the only correct one.
INT_MISSING = -1
INT_FILL = -2
STR_MISSING = "."
STR_FILL = ""
FLOAT16_MISSING, FLOAT16_FILL = np.array([0x7C01, 0x7C02], dtype=np.int16).view(
    np.float16
)
FLOAT32_MISSING, FLOAT32_FILL = np.array([0x7F800001, 0x7F800002], dtype=np.int32).view(
    np.float32
)
FLOAT64_MISSING, FLOAT64_FILL = np.array(
    [0x7FF0000000000001, 0x7FF0000000000002], dtype=np.int64
).view(np.float64)


# --------------------------------------------------------------------------
# Arrays
# --------------------------------------------------------------------------
# The arrays the spec defines, keyed by name: the dimension names each
# carries and the numpy dtype kinds a query can read it as. Every array ACGT
# reads by name is here; INFO and FORMAT arrays are named for their field and
# described only by the store's own metadata. Kinds, not widths: an int8 or
# int32 position is the writer's choice. Integers are signed because the
# missing and fill sentinels are negative; text is variable-length (the
# spec's |O, which zarr-python 3 presents as StringDType, kind T).
FIXED_ARRAYS = {
    "variant_contig": ((DIM_VARIANTS,), "i"),
    "variant_position": ((DIM_VARIANTS,), "i"),
    "variant_id": ((DIM_VARIANTS,), "OT"),
    "variant_id_mask": ((DIM_VARIANTS,), "b"),
    "variant_allele": ((DIM_VARIANTS, DIM_ALLELES), "OT"),
    "variant_quality": ((DIM_VARIANTS,), "f"),
    "variant_filter": ((DIM_VARIANTS, DIM_FILTERS), "b"),
    "variant_length": ((DIM_VARIANTS,), "i"),
    "call_genotype": ((DIM_VARIANTS, DIM_SAMPLES, DIM_PLOIDY), "i"),
    "call_genotype_phased": ((DIM_VARIANTS, DIM_SAMPLES), "b"),
    "call_genotype_mask": ((DIM_VARIANTS, DIM_SAMPLES, DIM_PLOIDY), "b"),
    "sample_id": ((DIM_SAMPLES,), "OT"),
    "contig_id": ((DIM_CONTIGS,), "OT"),
    "contig_length": ((DIM_CONTIGS,), "i"),
    "filter_id": ((DIM_FILTERS,), "OT"),
    "filter_description": ((DIM_FILTERS,), "OT"),
    "region_index": ((DIM_REGION_INDEX_VALUES, DIM_REGION_INDEX_FIELDS), "i"),
}

# Arrays the spec describes unconditionally; preflight fails a store missing
# any. Not here: call_genotype (only when the VCF has GT), variant_length
# (only with region_index), and the arrays the spec calls optional
# (contig_length, call_genotype_phased, region_index, the *_mask arrays).
REQUIRED_ARRAYS = frozenset(
    {
        "variant_contig",
        "variant_position",
        "variant_id",
        "variant_allele",
        "variant_quality",
        "variant_filter",
        "contig_id",
        "filter_id",
        "filter_description",
        "sample_id",
    }
)
