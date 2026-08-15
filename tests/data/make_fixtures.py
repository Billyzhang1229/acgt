"""Generate the synthetic VCF fixtures in this directory.

Every file here is invented, tiny, and committed to the repository; this
script exists so the fixtures can be audited and regenerated rather than
treated as opaque blobs. Regeneration needs bgzip and tabix (htslib) and,
for the BCF twin, bcftools — all external binaries, per AGENTS.md.

The rich fixture (sample.vcf) is deliberately awkward: multiallelic sites,
a monomorphic-reference site (ALT="."), missing QUAL/ID/FILTER, phased and
unphased calls, a haploid contig, alleles and INFO strings longer than the
converter's inline string cap, Number=A/R/G/./fixed fields, a Flag, a
Character, comma-separated FORMAT strings, and a contig with no records.
"""

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

LONG_ALT = "T" + "ACGT" * 15  # 61 characters, past the default 48 cap
LONG_STR = "ABCDEFGHIJ" * 6  # 60 characters, same idea for an INFO string

SAMPLE_HEADER = """\
##fileformat=VCFv4.2
##reference=GRCh38
##contig=<ID=chr1,length=1000000>
##contig=<ID=chr2,length=800000>
##contig=<ID=chrM,length=16569>
##contig=<ID=chrEmpty,length=5000>
##FILTER=<ID=PASS,Description="All filters passed">
##FILTER=<ID=q10,Description="Quality below 10">
##FILTER=<ID=s50,Description="Less than 50% of samples have data">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total depth">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency">
##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count">
##INFO=<ID=DB,Number=0,Type=Flag,Description="dbSNP membership">
##INFO=<ID=ANN,Number=.,Type=String,Description="Annotations">
##INFO=<ID=LONGS,Number=1,Type=String,Description="Long string field">
##INFO=<ID=BQ,Number=1,Type=Float,Description="Base quality">
##INFO=<ID=CH,Number=1,Type=Character,Description="One character">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">
##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phred likelihoods">
##FORMAT=<ID=AB,Number=1,Type=Float,Description="Allele balance">
##FORMAT=<ID=FS,Number=.,Type=String,Description="Free-form strings">
##FORMAT=<ID=PGT,Number=1,Type=String,Description="Physical phasing genotype">
##FORMAT=<ID=FC,Number=1,Type=Character,Description="One character per call">
##FORMAT=<ID=FF,Number=.,Type=Float,Description="Float vector">
##FORMAT=<ID=HQ,Number=2,Type=Integer,Description="Haplotype qualities">
"""

# One row per record, wrapped like the VCF line it becomes. Covers scalar
# and vector fields, Character values, and missing/fill padding.
# fmt: off
SAMPLE_RECORDS = [
    ("chr1", 100, "rs1", "A", "G", "50", "PASS",
     "DP=30;AF=0.5;AC=1;DB;ANN=upstream;CH=x",
     "GT:DP:GQ:AD:PL:AB:PGT:FC:FF:HQ",
     "0/1:30:99:15,15:50,0,60:0.5:0|1:a:.,0.5:20,30"),
    ("chr1", 200, ".", "C", "T,G", "12.7", "q10",
     "DP=18;AF=0.25,0.25;AC=1,1;ANN=.,exon",
     "GT:DP:AD:PL:FF", "1/2:18:.,6,6:90,45,20,80,0,70:0.2"),
    ("chr1", 300, ".", "G", ".", ".", ".", "DP=5", "GT:DP", "./.:5"),
    ("chr1", 400, "rs4", "T", LONG_ALT, "99", "PASS",
     f"DP=44;LONGS={LONG_STR}", "GT", "1|1"),
    ("chr1", 500, ".", "A", "T", "30", "q10;s50", "DP=2;BQ=13.5",
     "GT:GQ:HQ", "0/0:12:.,7"),
    ("chr2", 150, ".", "AACG", "A", "40", "PASS", "DP=22;AF=0.5;AC=1",
     "GT:AD:AB:PGT:FC", "0|1:11,11:0.47:0|1:."),
    ("chr2", 250, ".", "G", "C", "33", "PASS", "DP=9;ANN=a,b,c",
     "GT:FS", "0/1:xx,yy"),
    ("chr2", 350, ".", "C", "A", "21", "PASS", "DP=7", "GT", "./."),
    ("chr2", 450, "rs9", "T", "A", "65", "PASS", "DP=51;AF=1.0;AC=2;DB",
     "GT:DP:GQ:AD:PL", "1/1:50:99:0,50:200,60,0"),
    ("chrM", 100, ".", "A", "G", "80", "PASS", "DP=1000", "GT:DP", "1:999"),
    ("chrM", 200, ".", "C", "T", ".", "PASS", "DP=800", "GT", "0"),
]
# fmt: on

MINI = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000000>
##FILTER=<ID=PASS,Description="All filters passed">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total depth">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
chr1\t100\t.\tA\tG\t40\tPASS\tDP=10\tGT:DP\t0/1:10
chr1\t200\trs2\tC\tT\t50\tPASS\tDP=20\tGT:DP\t1|1:20
chr1\t300\t.\tG\tGA\t.\tPASS\t.\tGT\t0/0
"""

MULTISAMPLE = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000000>
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2
chr1\t100\t.\tA\tG\t40\tPASS\t.\tGT\t0/1\t1/1
"""

# One triploid call with a Number=G field: the genotype count is the
# multiset coefficient, which the diploid formula undercounts.
TRIPLOID = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000000>
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phred likelihoods">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
chr1\t100\t.\tA\tG\t30\tPASS\t.\tGT:PL\t0/1/1:50,0,10,99
"""

# A header and nothing else. bio2zarr cannot convert a zero-record file,
# so this one is preflight-checked rather than oracle-compared.
EMPTY = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000000>
##FILTER=<ID=PASS,Description="All filters passed">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total depth">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
"""

# Records that contradict their own header; conversion must refuse loudly
# (the reference implementation rejects the filter case the same way)
# rather than drop data or throw a bare KeyError.
UNDECLARED_FILTER = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000000>
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
chr1\t100\t.\tA\tG\t40\tq10\t.\tGT\t0/1
"""

UNDECLARED_CONTIG = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000000>
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
chr1\t100\t.\tA\tG\t40\tPASS\t.\tGT\t0/1
chr9\t200\t.\tC\tT\t50\tPASS\t.\tGT\t1/1
"""

# GT declared but omitted from one record's FORMAT: a legal file whose
# absent call must come out missing, not crash the parser.
PARTIAL_GT = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000000>
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
chr1\t100\t.\tA\tG\t40\tPASS\t.\tGT:DP\t0/1:7
chr1\t200\t.\tC\tT\t50\tPASS\t.\tDP\t9
chr1\t300\t.\tG\tA\t60\tPASS\t.\tGT\t./.
"""

# GT declared, but the only record carries three PL values and no GT.
PL_WITHOUT_GT = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000000>
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phred likelihoods">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
chr1\t100\t.\tA\tG\t40\tPASS\t.\tPL\t0,10,20
"""

# No FORMAT/GT at all, with a three-wide Number=G field.
NO_GT = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000000>
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">
##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phred likelihoods">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
chr1\t100\t.\tA\tG\t40\tPASS\t.\tDP:PL\t7:0,10,20
chr1\t200\t.\tC\tT\t50\tPASS\t.\tDP\t9
"""

# Ordering violations. region_index takes a chunk's first and last
# position as bounds, so these must be refused, not silently indexed.
UNSORTED = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000000>
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
chr1\t200\t.\tA\tG\t40\tPASS\t.\tGT\t0/1
chr1\t100\t.\tC\tT\t50\tPASS\t.\tGT\t1/1
"""

INTERLEAVED = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000000>
##contig=<ID=chr2,length=800000>
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
chr1\t100\t.\tA\tG\t40\tPASS\t.\tGT\t0/1
chr2\t100\t.\tC\tT\t50\tPASS\t.\tGT\t1/1
chr1\t300\t.\tG\tA\t60\tPASS\t.\tGT\t0/0
"""

# Dots in every awkward place, all legal: a whole-sample "." call, the *
# spanning-deletion allele, and "." for single values inside INFO vectors
# (MISSING inside a present value, distinct from FILL past its end).
DOT_AND_STAR = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000000>
##FILTER=<ID=PASS,Description="All filters passed">
##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
chr1\t100\t.\tA\tG\t40\tPASS\t.\tGT:DP\t.
chr1\t200\t.\tCT\tC,*\t50\tPASS\tAC=.,7;AF=.,0.5\tGT:DP\t1/2:9
"""

# Legal VCF with no ##contig lines at all: rejected up front with a clear
# message (the reference implementation cannot convert these either).
NO_CONTIG_HEADER = """\
##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
chr1\t100\t.\tA\tG\t40\tPASS\t.\tGT\t0/1
"""

# A contig declared without a length: contig_length must be omitted, not
# written with cyvcf2's -1 placeholder.
NO_CONTIG_LENGTH = """\
##fileformat=VCFv4.2
##contig=<ID=chr1>
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
chr1\t100\t.\tA\tG\t40\tPASS\t.\tGT\t0/1
"""

# Keys the header never declared: INFO/XX outright, and FORMAT/DP, which
# shares its name with a declared INFO/DP but lives in a different
# namespace. htslib assumes Type=String while parsing, the reference drops
# both fields, and so do we — with a warning naming each one.
UNDECLARED_INFO = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000000>
##FILTER=<ID=PASS,Description="All filters passed">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Depth">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
chr1\t100\t.\tA\tG\t40\tPASS\tDP=5;XX=7\tGT:DP\t0/1:9
"""


def write_sample():
    lines = [SAMPLE_HEADER]
    lines.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n")
    for chrom, pos, vid, ref, alt, qual, filt, info, fmt, call in SAMPLE_RECORDS:
        lines.append(
            f"{chrom}\t{pos}\t{vid}\t{ref}\t{alt}\t{qual}\t{filt}\t"
            f"{info}\t{fmt}\t{call}\n"
        )
    path = HERE / "sample.vcf"
    path.write_text("".join(lines))
    subprocess.run(["bgzip", "-kf", path], check=True)
    subprocess.run(["tabix", "-f", "-p", "vcf", f"{path}.gz"], check=True)
    return path


def write_bcf(vcf_gz):
    bcf = HERE / "sample.bcf"
    subprocess.run(["bcftools", "view", "-O", "b", "-o", bcf, vcf_gz], check=True)
    subprocess.run(["bcftools", "index", "-f", bcf], check=True)


def main():
    for tool in ("bgzip", "tabix"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} not found; install htslib to regenerate fixtures")
    sample = write_sample()
    (HERE / "mini.vcf").write_text(MINI)
    (HERE / "multisample.vcf").write_text(MULTISAMPLE)
    (HERE / "triploid.vcf").write_text(TRIPLOID)
    (HERE / "empty.vcf").write_text(EMPTY)
    (HERE / "undeclared_filter.vcf").write_text(UNDECLARED_FILTER)
    (HERE / "undeclared_contig.vcf").write_text(UNDECLARED_CONTIG)
    (HERE / "partial_gt.vcf").write_text(PARTIAL_GT)
    (HERE / "pl_without_gt.vcf").write_text(PL_WITHOUT_GT)
    (HERE / "no_gt.vcf").write_text(NO_GT)
    (HERE / "unsorted.vcf").write_text(UNSORTED)
    (HERE / "interleaved.vcf").write_text(INTERLEAVED)
    (HERE / "dot_and_star.vcf").write_text(DOT_AND_STAR)
    (HERE / "no_contig_header.vcf").write_text(NO_CONTIG_HEADER)
    (HERE / "no_contig_length.vcf").write_text(NO_CONTIG_LENGTH)
    (HERE / "undeclared_info.vcf").write_text(UNDECLARED_INFO)
    if shutil.which("bcftools"):
        write_bcf(f"{sample}.gz")
    else:
        print("bcftools not found; skipped the BCF twin")
    print("wrote fixtures under", HERE)


if __name__ == "__main__":
    main()
