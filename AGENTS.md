# ACGT

ACGT is a tool for personal genomics. You point it at your own
genome and explore it on your own machine.

## The one rule

**Nothing derived from the user's genome leaves this machine unless the user
sent it themselves.** "Sent it themselves" means all three of these held:

- the user saw the actual content that was going out,
- the user chose to send it, and
- there is a record of what was sent afterwards.

ACGT works offline and is fully usable
without ever sending anything, so there's no telemetry, no crash reporting, and
no background sync. Downloading reference data is fine, since asking a public
database for a fixed URL says nothing about the user. A URL with a coordinate,
a gene, or an rsid in it is a different thing — that's user data going out,
whatever the code calls it, and it needs clear consent from the user before it
happens.

The rule covers agents working on ACGT, not only ACGT itself. An agent runs on
a hosted model, so reading a real genome file into its context sends
genome-derived data off the machine as surely as an HTTP request does. Work
against the synthetic fixtures in `tests/data/`. If you need to see something
from the user's own data, ask first and say what you want to look at and why.

## Data core

VCZ (VCF Zarr) is the only format ACGT queries. Whatever the user supplies is
converted to VCZ on import and is never queried directly.

`convert.py` is the only place a foreign format exists. Every conversion path
lives there. VCF and BCF go through `bio2zarr`, the reference converter for
the VCF Zarr spec; consumer array exports (23andMe, AncestryDNA) are turned
into VCF by `bcftools` first and then take the same path. New source formats
get another function in `convert.py`, not another module.

Downstream of import there is one data model: an xarray Dataset backed by Zarr,
following sgkit's variant/sample dimension conventions.

- Single sample is the working assumption. Trios and cohorts are not in scope
  yet; `convert.py` refuses them and `dataset.preflight` refuses a store that
  carries more than one sample, so a cohort store cannot slip in by another
  route.
- Lazy by default. Slice by region; never materialize a whole array.
- Import is done by `bio2zarr`. `from_vcf` checks that the input is in
  scope (one sample) and hands the file over; intermediate files, an
  existing target, and what a failed run leaves behind are bio2zarr's
  behaviour and are not wrapped. Anything it leaves behind stays on this
  machine, so the one rule holds.

## Core packages

| Purpose | Package |
| --- | --- |
| Data core | `zarr`, `numcodecs`, `numpy`, `xarray` |
| TUI | `textual` |
| VCF/BCF → VCZ, import only | `bio2zarr` (the reference converter; `cyvcf2` underneath) |
| Dev | `pytest`, `ruff`, `ty`, `import-linter` |
| Cross-testing | `vcztools` |

That table is the dependency list. Add to it only when the work genuinely
cannot be done with what is already there, and say in the commit why the
existing packages were not enough.

The library that parses foreign formats is imported only in `convert.py`,
whatever that library is. The restriction exists because confinement is the
module's whole job: foreign formats and their parser stay in one place, so
replacing the parser touches one file and nothing downstream of import ever
sees a foreign format. `bio2zarr` (and the `cyvcf2` it parses with) is the
current converter; if it changes, this rule doesn't.

`bio2zarr` is the converter, not a second implementation to compare against:
it is the reference for the spec, and writing our own was tried and dropped
because keeping it equal to the reference cost more than the speed it bought.
What the tests hold is that `convert.py` hands the file to it faithfully and
that the result passes `dataset.preflight`; `bio2zarr` ships a store-versus-VCF
verifier, run on a fixture as the content check. `vcztools` is the test-only
reader: it renders a store back out to VCF, so query results can be checked
against the reference implementation. It does not ship in the application.

Consumer array exports reach VCZ through `bcftools convert --tsv2vcf` and then
the same path, so the same checks apply; the tsv-to-VCF step is `bcftools`'s
to get right, and a round trip against the source export is what the tests
can add.

`bcftools` is the reference for query semantics and the baseline for timings,
and `hyperfine` compares the two at the command line. Both are external
binaries rather than uv dependencies — tests and benchmarks that need them
skip when they are absent instead of failing.

Benchmarks run against synthetic data produced by a generator script, never
against a real genome and never against a file committed to the repository.
Useful timings need realistic volume, and that is exactly the point where
someone reaches for their own data. The generator is checked in; what it
produces is not.

**Network code can't see the data model.** If a module ever talks to the
network, it doesn't import `core.py`, `dataset.py`, or `query.py`, so a
`Variant` or a whole store can't be handed to it even by accident — only
primitives it declares itself. Anything in the data path may call it; the
constraint is on what it can be given, not on who calls. `import-linter` checks
the one-way contract in CI.

## Layout

The package is flat: one `.py` per module, no subpackages inside it.

```
acgt/                repository root
  acgt/
    __init__.py
    core.py          what an ACGT dataset is — the contract a store must meet
    dataset.py       open a VCZ store; preflight
    convert.py       VCF/BCF and array exports → VCZ
    query.py         gene and region queries
    annotate.py      local annotation
    tui.py           Textual app
    acgt.tcss        Textual stylesheet
  tests/
    data/            fixtures and synthetic genomes — never real data
  docs/
```

If a module grows to the point where you want to turn it into a directory,
split the work it is doing instead of growing the tree.

Screens live in `tui.py`, registered through Textual's `SCREENS` mapping, with
all styling in `acgt.tcss` rather than inline. When a widget ends up used by
two screens, it moves to a flat `widgets.py` — still not a directory.

`core.py` holds only what describes the data model — no I/O, no computation.
Anything else that lands there belongs in `dataset.py` or `query.py`.

## Toolchain

```
uv sync
uv run acgt
uv run pytest
uv run ruff check && uv run ruff format
uv run ty check
```

ACGT targets Python 3.13.

## Type hints

Code should read clearly on its own. Annotations document an interface; they do
not compensate for unclear naming, and most functions here do not need them.

Annotate the package's public surface — what someone importing `acgt` calls
directly — and only where the annotation says something the code does not:

- Functions with a definite return value the caller has to know about.
- Functions that produce a new kind of data: a genotype matrix, a summary
  record, anything whose shape differs from what went in.
- dataclass, NamedTuple, and other structured data fields.

Leave the rest alone. A function that operates on a store and hands back a store
does not need annotating, and neither does one whose return is obvious from its
name — `count_variants` returns a count, and `-> int` adds nothing to that.
Simple local helpers don't need them either.

`xr.Dataset` in particular is not worth annotating. Nearly every function in
`dataset.py` and `query.py` takes or returns one, and the annotation says
nothing about which variables or dimensions the store actually carries. What a
function requires of a store is described by the contract in `core.py`, checked by
preflight in `dataset.py`, and named in the docstring.

Where you do annotate, prefer the type that describes the actual contract over
the concrete implementation type, and do not narrow what the function really
accepts: `Iterable[str]` for contig names the function only iterates over,
`Sequence[int]` for positions it indexes into, `list[int]` only when the caller
is genuinely required to hand over a list. Avoid `Any` unless the interface is
genuinely dynamic. When modifying established code, follow the conventions
already there.

For NumPy-facing APIs, avoid overly specific array annotations unless dtype and
shape are genuinely part of the contract. Runtime validation and documentation
remain the source of truth for constraints that cannot be expressed accurately
with standard Python typing.

`ty check` must pass, but it is the floor rather than the standard: it will not
ask for an annotation that is not there, so what gets annotated is decided here
and not by the tool.

## Issues

GitHub carries the work: issues, pull requests, discussion. How a human
contributor branches and when they push is their own business.

Agents do not push branches, open or update pull requests, or post issues and
comments without the user asking for that specific action. Committing locally
is fine. Anything that reaches GitHub needs a clear yes first, and consent to
one push is not consent to the next one.

An issue states the problem before it proposes anything — what you were doing,
what happened, what you expected, in that order and in plain language. Title
the issue after the symptom rather than after a guess at the cause.

The one rule binds issues harder than anything else here, because an issue is
public and permanent. Never paste a real genome into one: no sample names, no
full local paths, no genotypes, and no traceback you have not read line by line
first. Reproduce against a synthetic fixture from `tests/data/` and report that.
When a bug only shows up on real data, describe the data's shape instead —
counts, contigs, provenance, build — and say that is what you did.

A pull request references the issue it closes. A commit references it when the
connection is not obvious from the diff.

## Commits

Plain language. No `feat:` or `fix:` prefixes, no scopes.

The subject is one imperative line under about 72 characters saying what the
commit does. The body is prose: state the problem first, then why this change
answers it. If you considered another approach and dropped it, name it and say
why. The git log is where this project's reasoning is recorded, so write the
reasons down instead of leaving them to be reconstructed later.

Agents commit under the same protocol, with two additions: an `Assisted-by`
trailer, and a body that reads correctly to someone who never saw the
conversation that produced it. No "as requested", no "per the discussion" —
restate the reason.

```
Convert everything to VCZ on import instead of querying source files

Querying a tabix-indexed VCF puts htslib in the hot path, and a chip export
cannot be queried by region at all. Converting once at import leaves a single
query engine over Zarr arrays for the whole application, and confines foreign
formats to one module.

Conversion costs time and disk up front. Accepted: a personal genome is
imported once and queried for years.

Refs: #12
Assisted-by: Claude Opus 5
```

The one rule applies here too. No real sample names, no full local paths, no
genotypes — not in a subject, a body, or an example.
