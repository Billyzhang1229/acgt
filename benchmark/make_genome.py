"""Generate a synthetic single-sample VCF large enough to time against.

The fixtures under tests/data are tiny on purpose; a benchmark needs volume,
and volume is exactly the point where someone reaches for a real genome. This
script exists so nobody has to. Everything it writes is invented from a seeded
RNG — the contigs borrow GRCh38's names and lengths, and the rest of the file
is shaped like a germline WGS call set (mostly biallelic SNPs, some indels and
multiallelics, GT:AD:DP:GQ:PL calls) without being derived from any person.

Output goes to benchmark/data/ by default, which is ignored by git. Nothing
this script produces is meant to be committed.

    uv run python benchmark/make_genome.py                # 1M records
    uv run python benchmark/make_genome.py -n 4_000_000   # WGS-sized

bgzip and tabix (htslib) are used when present, so the result can also be
converted as .vcf.gz; without them the plain .vcf is left as is.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DEFAULT_OUT = HERE / "data" / "synthetic.vcf"

# GRCh38 primary assembly, autosomes plus X. Names and lengths only.
CONTIGS = [
    ("chr1", 248956422),
    ("chr2", 242193529),
    ("chr3", 198295559),
    ("chr4", 190214555),
    ("chr5", 181538259),
    ("chr6", 170805979),
    ("chr7", 159345973),
    ("chr8", 145138636),
    ("chr9", 138394717),
    ("chr10", 133797422),
    ("chr11", 135086622),
    ("chr12", 133275309),
    ("chr13", 114364328),
    ("chr14", 107043718),
    ("chr15", 101991189),
    ("chr16", 90338345),
    ("chr17", 83257441),
    ("chr18", 80373285),
    ("chr19", 58617616),
    ("chr20", 64444167),
    ("chr21", 46709983),
    ("chr22", 50818468),
    ("chrX", 156040895),
]

BASES = np.array(list("ACGT"))

HEADER = """\
##fileformat=VCFv4.2
##source=acgt/benchmark/make_genome.py
##reference=GRCh38
{contigs}
##FILTER=<ID=PASS,Description="All filters passed">
##FILTER=<ID=LowQual,Description="Low quality">
##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count in genotypes">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency">
##INFO=<ID=AN,Number=1,Type=Integer,Description="Total number of alleles">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Approximate read depth">
##INFO=<ID=MQ,Number=1,Type=Float,Description="RMS mapping quality">
##INFO=<ID=DB,Number=0,Type=Flag,Description="dbSNP membership">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality">
##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phred-scaled likelihoods">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE
"""


def _alleles(rng, n):
    """REF and ALT strings for n sites: ~85% SNPs, ~10% indels, ~5% multiallelic."""
    kind = rng.choice(3, size=n, p=[0.85, 0.10, 0.05])
    ref = rng.choice(BASES, size=n)
    alt = np.empty(n, dtype=object)
    for i in range(n):
        r = ref[i]
        if kind[i] == 0:
            alt[i] = rng.choice(BASES[r != BASES])
        elif kind[i] == 1:
            if rng.random() < 0.5:  # insertion
                alt[i] = r + "".join(rng.choice(BASES, size=rng.integers(1, 6)))
            else:  # deletion
                ref[i] = r + "".join(rng.choice(BASES, size=rng.integers(1, 6)))
                alt[i] = r
        else:
            others = BASES[r != BASES]
            alt[i] = ",".join(rng.choice(others, size=2, replace=False))
    return ref, alt


def _records(rng, chrom, positions):
    n = len(positions)
    ref, alt = _alleles(rng, n)
    n_alt = np.char.count(alt.astype(str), ",") + 1
    # Germline single sample: het and hom-alt dominate; a few hom-ref
    # sites stand in for what a joint-called file would carry.
    gt_kind = rng.choice(3, size=n, p=[0.55, 0.40, 0.05])
    phased = rng.random(n) < 0.2
    dp = rng.poisson(32, size=n).clip(1)
    qual = np.round(rng.gamma(2.0, 200.0, size=n), 2)
    lowqual = qual < 30
    gq = np.where(lowqual, rng.integers(3, 30, size=n), rng.integers(30, 100, size=n))
    mq = np.round(rng.normal(59.5, 1.5, size=n), 2)
    db = rng.random(n) < 0.6
    ids = np.where(db, "rs" + rng.integers(1_000, 200_000_000, size=n).astype(str), ".")

    lines = []
    for i in range(n):
        na = n_alt[i]
        if gt_kind[i] == 0:
            gt = ("0|1", "0/1")[not phased[i]] if na == 1 else "1/2"
            ac = "1" if na == 1 else "1,1"
            af = "0.5" if na == 1 else "0.5,0.5"
        elif gt_kind[i] == 1:
            gt = "1|1" if phased[i] else "1/1"
            ac = "2" if na == 1 else "2,0"
            af = "1.0" if na == 1 else "1.0,0.0"
        else:
            gt = "0/0"
            ac = "0" if na == 1 else "0,0"
            af = "0.0" if na == 1 else "0.0,0.0"
        d = int(dp[i])
        if na == 1:
            ref_reads = {0: d // 2, 1: 0, 2: d}[gt_kind[i]]
            ad = f"{ref_reads},{d - ref_reads}"
            pl = {0: f"{d * 3},0,{d * 3}", 1: f"{d * 3},{d},0", 2: f"0,{d},{d * 3}"}[
                gt_kind[i]
            ]
        else:
            third = d // 3
            ad = f"{third},{third},{d - 2 * third}"
            pl = f"{d * 3},{d},0,{d},{d * 2},{d * 3}"
        info = f"AC={ac};AF={af};AN=2;DP={d};MQ={mq[i]}"
        if db[i]:
            info += ";DB"
        lines.append(
            f"{chrom}\t{positions[i]}\t{ids[i]}\t{ref[i]}\t{alt[i]}\t{qual[i]}\t"
            f"{'LowQual' if lowqual[i] else 'PASS'}\t{info}\t"
            f"GT:AD:DP:GQ:PL\t{gt}:{ad}:{d}:{gq[i]}:{pl}\n"
        )
    return lines


def write_genome(out, n_records, seed, batch=200_000):
    rng = np.random.default_rng(seed)
    total = sum(length for _, length in CONTIGS)
    per_contig = np.round(
        np.array([length for _, length in CONTIGS]) / total * n_records
    )
    per_contig = per_contig.astype(int)
    per_contig[0] += n_records - per_contig.sum()  # keep the total exact

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write(
            HEADER.format(
                contigs="\n".join(
                    f"##contig=<ID={name},length={length}>" for name, length in CONTIGS
                )
            )
        )
        for (chrom, length), count in zip(CONTIGS, per_contig, strict=True):
            # Sorted, unique positions across the contig, written in batches
            # so a WGS-sized run does not hold every line in memory.
            positions = np.sort(rng.choice(length - 1, size=count, replace=False) + 1)
            for start in range(0, count, batch):
                fh.writelines(_records(rng, chrom, positions[start : start + batch]))
    return out


def compress(vcf):
    if shutil.which("bgzip") is None or shutil.which("tabix") is None:
        print("bgzip/tabix not found; leaving the plain .vcf", file=sys.stderr)
        return vcf
    subprocess.run(["bgzip", "-f", vcf], check=True)
    gz = Path(f"{vcf}.gz")
    subprocess.run(["tabix", "-f", "-p", "vcf", gz], check=True)
    return gz


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-n", "--records", type=int, default=1_000_000)
    parser.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--no-bgzip", action="store_true", help="leave the output as plain text"
    )
    args = parser.parse_args(argv)
    path = write_genome(args.out, args.records, args.seed)
    if not args.no_bgzip:
        path = compress(path)
    print(f"wrote {args.records:,} records to {path}")


if __name__ == "__main__":
    main()
