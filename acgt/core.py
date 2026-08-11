"""What an ACGT dataset is — schema, provenance, build.

This module describes the data model and nothing else: the variables and
dimensions a VCZ store carries, where the data came from, and which reference
build it uses. No I/O and no computation live here; opening and saving belong
to dataset.py, queries to query.py.
"""
