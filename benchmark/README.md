# Benchmarks

Timings for the paths where speed is the point. Nothing here runs in CI and
nothing here is a test; the tests in `tests/` check that the output is right,
and this directory checks how long getting there takes.

Benchmarks run against synthetic data produced by a generator script — never
against a real genome, and never against a file committed to the repository.
The generator is checked in; what it produces goes to `benchmark/data/`, which
git ignores. See the one rule in AGENTS.md.

## Conversion

`make_genome.py` writes a single-sample VCF shaped like a germline WGS call
set (GRCh38 contig names and lengths, mostly biallelic SNPs, some indels and
multiallelics, `GT:AD:DP:GQ:PL` calls) from a seeded RNG. Run on its own it
writes one million records to `benchmark/data/`; `-n 4_000_000` is closer to
a real whole genome.

`bench_convert.py` generates a genome at each of several record counts,
converts it with `acgt.convert.from_vcf` and with bio2zarr — each in its own
process — and prints one table row per size: wall time, records per second,
peak memory, store size, and the speedup. Every store is preflighted and its
variant count checked before the numbers are printed. Genomes and stores go
to a temporary directory and are removed when the run ends.

```bash
uv run python benchmark/bench_convert.py
uv run python benchmark/bench_convert.py --sizes 100000,1000000,4000000 --repeat 3
uv run python benchmark/bench_convert.py --markdown
uv run python benchmark/bench_convert.py --compare
uv run python benchmark/bench_convert.py --vcf benchmark/data/synthetic.vcf.gz
```

The default sizes are 10k, 100k, and 1M records. `--vcf` times an existing
file instead of generating one. `--compare` keeps the last store from each
converter and checks them array for array — the same equality rule the
oracle tests use — and adds a "same data" column; when the stores differ,
each differing array is listed below the table. Both converters use the same variants chunk
size (10,000 by default, `--chunk-size` to change it), so the stores are
directly comparable. bio2zarr is a dev dependency and is available under
`uv run`; pass `--no-bio2zarr` to time acgt alone.

Timings depend on the machine, the file, and the chunking. When quoting a
number, say which of the three produced it.

## Queries

Not yet. `bcftools` is the reference for query semantics and the baseline for
timings, and `hyperfine` compares the two at the command line; both are
external binaries, so the query benchmarks will skip when they are absent
rather than fail. They land with `query.py`.
