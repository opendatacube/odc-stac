#!/usr/bin/env sh

set -e

indir="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)"
outdir="$(dirname $indir)/docs/notebooks"

mkdir -p $outdir


for infile in $(find $indir -type f -maxdepth 1 -name '*py'); do
    outfile="${outdir}/$(basename ${infile%%.py}.ipynb)"
    $indir/render-nb.sh $infile $outfile
done
