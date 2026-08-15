"""Time VCF → VCZ conversion across file sizes: acgt.convert.from_vcf against bio2zarr.

For each record count the script generates a synthetic genome with
make_genome.py, converts it with both tools, and prints one table row. Each
conversion runs in a fresh child process so wall time and peak memory are
measured per converter rather than accumulated across the run. Every output
store is checked — preflight must pass and the variant count must match —
because a fast conversion that wrote the wrong store is not a result worth
reporting.

    uv run python benchmark/bench_convert.py
    uv run python benchmark/bench_convert.py --sizes 100000,1000000,4000000 --repeat 3
    uv run python benchmark/bench_convert.py --vcf some/file.vcf.gz
    uv run python benchmark/bench_convert.py --markdown
    uv run python benchmark/bench_convert.py --compare

`--compare` keeps the last store from each converter and checks them array
for array — the same equality the oracle tests use — so a timing table can
say in the same breath whether the two tools produced the same data.

Generated genomes live in a temporary directory for the duration of the run
and are removed afterwards; nothing is committed. bio2zarr is a dev
dependency, so it is available under `uv run`; if it is not importable the
comparison is skipped and only acgt is timed. Timings depend on the machine,
the file, and the chunking, so report all three alongside the numbers.
"""

import argparse
import json
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parent))
import make_genome

CHUNK = 10_000
DEFAULT_SIZES = (10_000, 100_000, 1_000_000)


# --------------------------------------------------------------------------
# child process: one conversion, result on stdout as JSON
# --------------------------------------------------------------------------
def _peak_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    return rss / 2**20 if platform.system() == "Darwin" else rss / 2**10


def _run_acgt(src, out, chunk):
    from acgt import convert

    t0 = time.perf_counter()
    convert.from_vcf(src, out, chunk_size=chunk, overwrite=True)
    return time.perf_counter() - t0


def _run_bio2zarr(src, out, chunk):
    import bio2zarr.vcf as b2z

    t0 = time.perf_counter()
    b2z.convert([str(src)], str(out), variants_chunk_size=chunk, show_progress=False)
    return time.perf_counter() - t0


RUNNERS = {"acgt": _run_acgt, "bio2zarr": _run_bio2zarr}


def _check(out):
    from acgt import dataset

    problems = dataset.preflight(out)
    ds = dataset.open_dataset(out, check=False)
    return problems, int(ds.sizes["variants"])


def _dir_size_mb(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()) / 2**20


def worker(tool, src, out, chunk):
    seconds = RUNNERS[tool](src, out, chunk)
    problems, n = _check(out)
    print(
        json.dumps(
            {
                "tool": tool,
                "seconds": seconds,
                "peak_rss_mb": _peak_rss_mb(),
                "store_mb": _dir_size_mb(out),
                "variants": n,
                "problems": problems,
            }
        )
    )


# --------------------------------------------------------------------------
# parent process: generate, fan out, collect, report
# --------------------------------------------------------------------------
def measure(tool, src, workdir, chunk, keep=False):
    out = Path(workdir) / f"{tool}.vcz"
    proc = subprocess.run(
        [sys.executable, __file__, "--_worker", tool, str(src), str(out), str(chunk)],
        capture_output=True,
        text=True,
        check=False,
    )
    if not keep:
        shutil.rmtree(out, ignore_errors=True)
    if proc.returncode != 0:
        sys.exit(f"{tool} failed on {src}:\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _have_bio2zarr():
    try:
        import bio2zarr  # noqa: F401
    except ImportError:
        return False
    return True


def _arrays_equal(x, y):
    """Same rules as the oracle tests: strings as text, floats bit for bit
    (missing and fill are distinct NaN payloads), ints by value regardless
    of the width each writer downcast to."""
    if x.shape != y.shape:
        return False
    if x.dtype.kind in "OUT" or y.dtype.kind in "OUT":
        return np.array_equal(np.asarray(x, dtype=object), np.asarray(y, dtype=object))
    if x.dtype.kind == "f":
        return x.dtype == y.dtype and np.array_equal(
            x.view(f"i{x.dtype.itemsize}"), y.view(f"i{y.dtype.itemsize}")
        )
    if x.dtype.kind in "iu":
        return np.array_equal(x.astype(np.int64), y.astype(np.int64))
    return np.array_equal(x, y)


def _values(group, key):
    node = group[key]
    assert isinstance(node, zarr.Array)
    return np.asarray(node[:])


def compare_stores(ours, theirs):
    """Array-for-array comparison. Returns a list of human-readable diffs;
    empty means the two stores carry identical data."""
    a = zarr.open_group(ours, mode="r")
    b = zarr.open_group(theirs, mode="r")
    ka, kb = set(a.array_keys()), set(b.array_keys())
    diffs = [f"only in acgt: {k}" for k in sorted(ka - kb)]
    diffs += [f"only in bio2zarr: {k}" for k in sorted(kb - ka)]
    for k in sorted(ka & kb):
        x, y = _values(a, k), _values(b, k)
        if x.shape != y.shape:
            diffs.append(f"{k}: shape {x.shape} vs {y.shape}")
        elif not _arrays_equal(x, y):
            if x.dtype.kind in "OUT" or y.dtype.kind in "OUT":
                n = int(
                    (np.asarray(x, dtype=object) != np.asarray(y, dtype=object)).sum()
                )
            elif x.dtype.kind == "f" and x.dtype != y.dtype:
                n = x.size
            else:
                n = int((x != y).sum())
            diffs.append(f"{k}: {n}/{x.size} values differ ({x.dtype} vs {y.dtype})")
    return diffs


def bench_file(src, tools, workdir, chunk, repeat, compare=False):
    """Best-of-`repeat` per tool for one input file. Returns a table row."""
    row = {"file": src.name, "input_mb": src.stat().st_size / 2**20}
    for tool in tools:
        runs = []
        for i in range(repeat):
            print(f"  {tool} run {i + 1}/{repeat} ...", file=sys.stderr)
            keep = compare and i == repeat - 1
            runs.append(measure(tool, src, workdir, chunk, keep=keep))
        best = min(runs, key=lambda r: r["seconds"])
        row[tool] = {
            "seconds": best["seconds"],
            "peak_rss_mb": max(r["peak_rss_mb"] for r in runs),
            "store_mb": best["store_mb"],
            "variants": best["variants"],
            "problems": best["problems"],
        }
    row["records"] = next(row[t]["variants"] for t in tools)
    if compare:
        stores = {t: Path(workdir) / f"{t}.vcz" for t in tools}
        if len(tools) > 1:
            print("  comparing stores ...", file=sys.stderr)
            row["diffs"] = compare_stores(stores[tools[0]], stores[tools[1]])
        for path in stores.values():
            shutil.rmtree(path, ignore_errors=True)
    return row


def _columns(tools, compare):
    cols = [("records", 11), ("input MB", 10)]
    for tool in tools:
        cols += [
            (f"{tool} s", len(tool) + 4),
            (f"{tool} rec/s", len(tool) + 10),
            (f"{tool} peak MB", len(tool) + 10),
            (f"{tool} store MB", len(tool) + 11),
        ]
    if len(tools) > 1:
        cols.append(("speedup", 9))
    if compare and len(tools) > 1:
        cols.append(("same data", 11))
    return cols


def _cells(row, tools, compare):
    cells = [f"{row['records']:,}", f"{row['input_mb']:.1f}"]
    for tool in tools:
        r = row[tool]
        cells += [
            f"{r['seconds']:.2f}",
            f"{row['records'] / r['seconds']:,.0f}",
            f"{r['peak_rss_mb']:.0f}",
            f"{r['store_mb']:.1f}",
        ]
    if len(tools) > 1:
        first, *rest = tools
        cells.append(
            "/".join(f"{row[t]['seconds'] / row[first]['seconds']:.1f}x" for t in rest)
        )
    if compare and len(tools) > 1:
        diffs = row["diffs"]
        cells.append("yes" if not diffs else f"no ({len(diffs)})")
    return cells


def report(rows, tools, chunk, repeat, markdown, compare):
    print(
        f"\n{platform.platform()}  Python {platform.python_version()}  "
        f"chunk={chunk}  best-of-{repeat}"
    )
    if len(tools) > 1:
        print(f"speedup = {tools[0]} time relative to {', '.join(tools[1:])}")
    print()
    cols = _columns(tools, compare)
    if markdown:
        print("| " + " | ".join(name for name, _ in cols) + " |")
        print("|" + "|".join("---:" for _ in cols) + "|")
        for row in rows:
            print("| " + " | ".join(_cells(row, tools, compare)) + " |")
    else:
        print("".join(f"{name:>{width}}" for name, width in cols))
        for row in rows:
            print(
                "".join(
                    f"{cell:>{width}}"
                    for cell, (_, width) in zip(
                        _cells(row, tools, compare), cols, strict=True
                    )
                )
            )
    for row in rows:
        for tool in tools:
            if row[tool]["problems"]:
                print(
                    f"\nWARNING: {tool} store for {row['file']} failed preflight: "
                    f"{row[tool]['problems']}"
                )
        counts = {row[t]["variants"] for t in tools}
        if len(counts) > 1:
            print(
                f"\nWARNING: variant counts disagree for {row['file']}: {sorted(counts)}"
            )
        if row.get("diffs"):
            print(f"\n{row['file']}: acgt and bio2zarr stores differ")
            for d in row["diffs"]:
                print(f"  {d}")


def _sizes(text):
    return [int(s.replace("_", "")) for s in text.split(",") if s.strip()]


def main(argv=None):
    if argv is None and len(sys.argv) > 1 and sys.argv[1] == "--_worker":
        tool, src, out, chunk = sys.argv[2:6]
        worker(tool, src, out, int(chunk))
        return
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--sizes",
        type=_sizes,
        default=list(DEFAULT_SIZES),
        help="comma-separated record counts to generate and time "
        f"(default: {','.join(str(s) for s in DEFAULT_SIZES)})",
    )
    parser.add_argument(
        "--vcf",
        type=Path,
        help="time this single-sample VCF/BCF instead of generating genomes",
    )
    parser.add_argument("--repeat", type=int, default=1, help="runs per converter")
    parser.add_argument("--chunk-size", type=int, default=CHUNK)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--no-bio2zarr", action="store_true", help="time acgt only")
    parser.add_argument(
        "--markdown", action="store_true", help="print a Markdown table"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="compare the acgt and bio2zarr stores array for array and report "
        "whether they carry the same data",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        help="where genomes and stores are written during the run (default: a temp dir)",
    )
    args = parser.parse_args(argv)
    if args.vcf is not None and not args.vcf.exists():
        sys.exit(f"{args.vcf} not found")

    tools = ["acgt"]
    if not args.no_bio2zarr:
        if _have_bio2zarr():
            tools.append("bio2zarr")
        else:
            print("bio2zarr not importable; timing acgt only", file=sys.stderr)

    rows = []
    with tempfile.TemporaryDirectory(dir=args.workdir) as tmp:
        if args.vcf is not None:
            print(f"{args.vcf.name}", file=sys.stderr)
            rows.append(
                bench_file(
                    args.vcf, tools, tmp, args.chunk_size, args.repeat, args.compare
                )
            )
        else:
            for n in args.sizes:
                print(f"generating {n:,} records ...", file=sys.stderr)
                vcf = make_genome.write_genome(
                    Path(tmp) / f"synthetic_{n}.vcf", n, args.seed
                )
                vcf = make_genome.compress(vcf)
                rows.append(
                    bench_file(
                        vcf, tools, tmp, args.chunk_size, args.repeat, args.compare
                    )
                )
                for leftover in Path(tmp).glob(f"synthetic_{n}.vcf*"):
                    leftover.unlink()
    report(rows, tools, args.chunk_size, args.repeat, args.markdown, args.compare)


if __name__ == "__main__":
    main()
