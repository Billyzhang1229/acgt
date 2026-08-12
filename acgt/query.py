"""Gene and region queries.

Queries run against the VCZ store only, lazily: slice by region, never
materialize a whole array. bcftools is the reference for query semantics.
"""
