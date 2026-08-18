"""Convert foreign formats to VCZ.

This is the only module where a foreign format exists. Every conversion path
lives here and the library that parses foreign formats is imported here and
nowhere else. New source formats get another function in this file, not
another module. Personal genome data, whatever format it arrives in, comes
out as a VCZ store; nothing downstream of import ever sees the source.

VCF and BCF go through bio2zarr, the reference converter for the VCF Zarr
spec. ACGT decides whether the input is in scope; bio2zarr does the rest.
Consumer array exports (23andMe, AncestryDNA) reach VCZ the same way:
bcftools turns them into VCF, and that VCF comes through here.
"""

import contextlib
from pathlib import Path

import bio2zarr.vcf as b2z
from cyvcf2 import VCF

# Variants per chunk in every array. The region index is defined per chunk,
# so this is also the granularity of region queries.
DEFAULT_CHUNK_SIZE = 10_000


def from_vcf(path, out, *, chunk_size=DEFAULT_CHUNK_SIZE) -> Path:
    """Convert a single-sample VCF or BCF at `path` into a VCZ store at `out`.

    Multi-sample files are refused: cohorts are out of scope, and a personal
    genome has one sample. Everything else is bio2zarr's to decide and report.
    """
    with contextlib.closing(VCF(str(path))) as vcf:
        n = len(vcf.samples)
    if n != 1:
        raise ValueError(f"from_vcf converts single-sample files, got {n} samples")
    b2z.convert([str(path)], str(out), variants_chunk_size=chunk_size)
    return Path(out)
