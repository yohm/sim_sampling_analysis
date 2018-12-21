#!/bin/bash
set -eux

INDIR=$(cd $1 && pwd)
OUTDIR=${2:-.}
OUTDIR=$(cd $OUTDIR && pwd)
script_dir=$(cd $(dirname $BASH_SOURCE); pwd)

mkdir -p $OUTDIR
cd $OUTDIR

plotpy="${script_dir}/plot.py"

# plot topological properties
python $plotpy "${INDIR}/cc_degree_correlation.dat" --xlog --ylog --xlabel 'X' --ylabel Y -f png
python $plotpy "${INDIR}/degree_distribution.dat" --ylog --xlabel 'degree' --ylabel 'frequency' -f png
python $plotpy "${INDIR}/neighbor_degree_correlation.dat" --xlog --xlabel 'k' --ylabel '$k_{nn}(k)$' -f png

# plot weight-related properties
if [ -e $INDIR/edge_weight_distribution.dat ]; then

ensemble_average_rb=$script_dir/../ensemble/ensemble_averaging/ensemble_averaging.rb
ruby ${ensemble_average_rb} -l -f "${INDIR}/edge_weight_distribution.dat" -o "${INDIR}/edge_weight_distribution_logbin.dat"
ruby ${ensemble_average_rb} -l "${INDIR}/overlap_weight_correlation.dat" -o "${INDIR}/overlap_weight_correlation_logbin.dat"
ruby ${ensemble_average_rb} -l -f "${INDIR}/strength_distribution.dat" -o "${INDIR}/strength_distribution_logbin.dat"

python $plotpy "${INDIR}/edge_weight_distribution_logbin.dat" --xlog --ylog --xlabel '$w_{ij}$', --ylabel 'frequency' -f png
python $plotpy "${INDIR}/overlap_weight_correlation_logbin.dat" --xlog --xlabel 'w' --ylabel 'O(w)' -f png
python $plotpy "${INDIR}/strength_degree_correlation.dat" --xlog --ylog --xlabel 'k' --ylabel 's(k)' -f png
python $plotpy "${INDIR}/strength_distribution_logbin.dat" --xlog --ylog --xlabel 's' --ylabel 'frequency' -f png

python "${script_dir}/link_removal_percolation.py" "${INDIR}/link_removal_percolation.dat" "${OUTDIR}/link_removal_percolation.png"
python "${script_dir}/link_removal_percolation_scale.py" "${INDIR}/link_removal_percolation.dat" "${OUTDIR}/link_removal_percolation_scale.png"

fi

