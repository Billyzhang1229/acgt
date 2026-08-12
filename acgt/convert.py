"""Convert foreign formats to VCZ.

This is the only module where a foreign format exists. Every conversion path
lives here — VCF/BCF from WGS or WES, and consumer array exports (23andMe,
AncestryDNA) — and the library that parses them is imported here and nowhere
else. New source formats get another function in this file, not another
module. Personal genome data, whatever format it arrives in, comes out as a VCZ store
carrying the schema defined in core.py.
"""
