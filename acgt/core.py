"""What an ACGT dataset is — schema, provenance, build.

This module describes the data model and nothing else: the variables and
dimensions a VCZ store carries, where the data came from, and which reference
build it uses. No I/O and no computation live here; opening and saving belong
to dataset.py, queries to query.py.

Two layers of schema live here. The spec layer is what every conforming store
shares: dimension names, sentinel values, the fixed arrays and their shapes,
and the rule that turns a VCF INFO/FORMAT declaration into an array spec. The
instance layer is Schema, which records what one particular store carries:
its dimension sizes, its fields, and its provenance. convert.py builds a
Schema when it writes a store; dataset.py reads it back and preflights the
store against it before anything queries.

The layout follows VCF Zarr 0.5 (github.com/sgkit-dev/vcf-zarr-spec). The
sentinel values match vcztools and bio2zarr bit for bit, so stores written
here are readable by the reference implementations and comparable against
their output; a test holds that equality in place.
"""

import dataclasses

import numpy as np

# Version of the VCF Zarr spec the store layout follows, written to the
# store's `vcf_zarr_version` group attribute.
VCF_ZARR_VERSION = "0.5"

# Version of the Schema serialization below, independent of the spec version
# so provenance fields can evolve without pretending the spec changed.
ACGT_SCHEMA_VERSION = "0.1"

# Group attribute under which Schema.asdict() is stored.
SCHEMA_ATTR = "acgt_schema"


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
# Array specs
# --------------------------------------------------------------------------
# The schema constrains dtype kind, not width: variant_position is "int"
# whether a given store holds it as int8 or int32. Exact widths are a storage
# decision made at conversion time and readable from the store itself.
KIND_INT = "int"
KIND_FLOAT = "float"
KIND_BOOL = "bool"
KIND_STR = "str"
# VCF Character is its own kind: exactly one character, which the spec
# stores as a fixed-width U1 rather than a variable-length string. Folding
# it into str would be lossless on the way in and wrong on the way out.
KIND_CHAR = "char"
KINDS = frozenset({KIND_INT, KIND_FLOAT, KIND_BOOL, KIND_STR, KIND_CHAR})

_VCF_TYPE_TO_KIND = {
    "Integer": KIND_INT,
    "Float": KIND_FLOAT,
    "Flag": KIND_BOOL,
    "String": KIND_STR,
    "Character": KIND_CHAR,
}


@dataclasses.dataclass(frozen=True)
class ArraySpec:
    """One array in a VCZ store: its name, dtype kind, and dimension names."""

    name: str
    kind: str
    dims: tuple[str, ...]
    description: str = ""

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(
                f"array {self.name!r} has unknown kind {self.kind!r}; "
                f"expected one of {sorted(KINDS)}"
            )


# The arrays defined by the spec itself, as opposed to those derived from a
# VCF header. Keyed by array name; values follow VCF Zarr 0.5.
FIXED_ARRAYS = {
    s.name: s
    for s in (
        ArraySpec(
            "variant_contig",
            KIND_INT,
            (DIM_VARIANTS,),
            "Index into contig_id for each variant",
        ),
        ArraySpec(
            "variant_position",
            KIND_INT,
            (DIM_VARIANTS,),
            "1-based position on the contig",
        ),
        ArraySpec("variant_id", KIND_STR, (DIM_VARIANTS,), "VCF ID field"),
        ArraySpec(
            "variant_id_mask",
            KIND_BOOL,
            (DIM_VARIANTS,),
            "True where the ID is missing",
        ),
        ArraySpec(
            "variant_allele",
            KIND_STR,
            (DIM_VARIANTS, DIM_ALLELES),
            "REF then ALT alleles",
        ),
        ArraySpec("variant_quality", KIND_FLOAT, (DIM_VARIANTS,), "VCF QUAL field"),
        ArraySpec(
            "variant_filter",
            KIND_BOOL,
            (DIM_VARIANTS, DIM_FILTERS),
            "One flag per filter in filter_id",
        ),
        ArraySpec(
            "variant_length",
            KIND_INT,
            (DIM_VARIANTS,),
            "Span of the variant on the contig",
        ),
        ArraySpec(
            "call_genotype",
            KIND_INT,
            (DIM_VARIANTS, DIM_SAMPLES, DIM_PLOIDY),
            "Allele indices; 0 is REF",
        ),
        ArraySpec(
            "call_genotype_phased",
            KIND_BOOL,
            (DIM_VARIANTS, DIM_SAMPLES),
            "True where the call is phased",
        ),
        ArraySpec(
            "call_genotype_mask",
            KIND_BOOL,
            (DIM_VARIANTS, DIM_SAMPLES, DIM_PLOIDY),
            "True where the allele call is missing",
        ),
        ArraySpec("sample_id", KIND_STR, (DIM_SAMPLES,), "Sample names"),
        ArraySpec("contig_id", KIND_STR, (DIM_CONTIGS,), "Contig names"),
        ArraySpec(
            "contig_length", KIND_INT, (DIM_CONTIGS,), "Contig lengths from the header"
        ),
        ArraySpec(
            "filter_id", KIND_STR, (DIM_FILTERS,), "Filter names; PASS comes first"
        ),
        ArraySpec(
            "filter_description",
            KIND_STR,
            (DIM_FILTERS,),
            "Filter descriptions from the header",
        ),
        ArraySpec(
            "region_index",
            KIND_INT,
            (DIM_REGION_INDEX_VALUES, DIM_REGION_INDEX_FIELDS),
            "Per chunk-contig row: chunk, contig, start, end, max_end, n_records",
        ),
    )
}

# Columns of one region_index row, in order.
REGION_INDEX_COLUMNS = ("chunk", "contig", "start", "end", "max_end", "n_records")

# Arrays the spec makes mandatory; preflight fails a store missing any.
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
        "sample_id",
    }
)


def spec_for_field(category, name, number, vcf_type, description="") -> ArraySpec:
    """Map one VCF INFO or FORMAT header declaration to an ArraySpec.

    `number` is the raw VCF Number string: a count, "A", "R", "G", or ".".
    A/R/G map to the reserved dimensions the spec assigns them; a fixed count
    above one and "." get a dimension named for the field, sized at
    conversion time. `description` is the header's Description text; it
    travels to the store as the array's `description` attribute, which is
    what the reference reader uses to rebuild the header line on the way
    back out. FORMAT/GT is not a field — it becomes call_genotype, already
    in FIXED_ARRAYS — so asking for it is an error, as is any field whose
    generated name would collide with a fixed array.
    """
    if category not in ("INFO", "FORMAT"):
        raise ValueError(f"category must be INFO or FORMAT, got {category!r}")
    if category == "FORMAT" and name == "GT":
        raise ValueError("FORMAT/GT maps to call_genotype, not a field array")
    kind = _VCF_TYPE_TO_KIND.get(vcf_type)
    if kind is None:
        raise ValueError(f"unknown VCF Type {vcf_type!r} for {category}/{name}")

    if category == "INFO":
        zarr_name = "variant_" + name
        dims = [DIM_VARIANTS]
    else:
        zarr_name = "call_" + name
        dims = [DIM_VARIANTS, DIM_SAMPLES]
    if zarr_name in FIXED_ARRAYS:
        raise ValueError(
            f"{category}/{name} would collide with the fixed array {zarr_name}"
        )

    number = str(number).strip()
    if vcf_type == "Flag" or number in ("0", "1"):
        pass
    elif number == "A":
        dims.append(DIM_ALT_ALLELES)
    elif number == "R":
        dims.append(DIM_ALLELES)
    elif number == "G":
        dims.append(DIM_GENOTYPES)
    else:
        # A fixed count above one, or "." meaning the header does not say.
        dims.append(f"{category}_{name}_dim")

    return ArraySpec(zarr_name, kind, tuple(dims), description)


# --------------------------------------------------------------------------
# Instance schema
# --------------------------------------------------------------------------
@dataclasses.dataclass
class Schema:
    """What one particular store carries.

    `dims` maps every dimension name the store uses to its size, `fields`
    lists every array including the fixed ones, and the provenance fields say
    where the data came from: `source` describes the input in words, `build`
    names the reference build. Serializes to a plain dict for the store's
    group attributes.
    """

    dims: dict[str, int]
    fields: tuple[ArraySpec, ...]
    source: str = ""
    build: str = ""
    schema_version: str = ACGT_SCHEMA_VERSION

    def __post_init__(self):
        for field in self.fields:
            for dim in field.dims:
                if dim not in self.dims:
                    raise ValueError(
                        f"field {field.name!r} uses dimension {dim!r}, "
                        "which dims does not declare"
                    )
            # the spec fixes the kind and dimensions of these arrays; a
            # schema may describe them but not redefine them
            fixed = FIXED_ARRAYS.get(field.name)
            if fixed is not None and (field.kind, field.dims) != (
                fixed.kind,
                fixed.dims,
            ):
                raise ValueError(
                    f"field {field.name!r} is fixed by the spec as "
                    f"{fixed.kind} over {list(fixed.dims)}, but the schema "
                    f"says {field.kind} over {list(field.dims)}"
                )

    def asdict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def fromdict(cls, d) -> "Schema":
        version = d.get("schema_version")
        if version != ACGT_SCHEMA_VERSION:
            raise ValueError(
                f"schema version mismatch: store has {version!r}, "
                f"this code reads {ACGT_SCHEMA_VERSION!r}"
            )
        fields = tuple(
            ArraySpec(
                name=f["name"],
                kind=f["kind"],
                dims=tuple(f["dims"]),
                description=f.get("description", ""),
            )
            for f in d["fields"]
        )
        return cls(
            dims=dict(d["dims"]),
            fields=fields,
            source=d.get("source", ""),
            build=d.get("build", ""),
        )

    def field_map(self) -> dict[str, ArraySpec]:
        return {f.name: f for f in self.fields}
