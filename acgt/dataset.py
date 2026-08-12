"""Open and save a VCZ store; preflight.

Everything downstream of import goes through here: an xarray Dataset backed by
Zarr, following sgkit's variant/sample dimension conventions. Preflight checks
that a store carries what the schema in core.py says it must, before any query
touches it.
"""
