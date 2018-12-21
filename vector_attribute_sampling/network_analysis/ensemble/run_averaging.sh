#!/bin/bash
set -eux

# Usage:
#   ./run_averaging.sh '_input/*' .
#
# The first argument indicates the directories containing input files.
# It specifies the patten of input directories, like 'data/*'.
# Make sure that the first argument is surrounded by single quotes, otherwise the argument is expanded by the shell.
#
# The second argument specifies the output directory. It is an optional argument and its default value is '.'.
#
PATTERN=$1
OUT_DIR=${2:-.}  # default output directory is the current directory
script_dir=$(cd $(dirname $BASH_SOURCE); pwd)

mkdir -p $OUT_DIR

ensemble_averaging_rb=${script_dir}/ensemble_averaging/ensemble_averaging.rb

ls $PATTERN/cc_degree_correlation.dat       | xargs ruby ${ensemble_averaging_rb}    > $OUT_DIR/cc_degree_correlation_ave.dat
ls $PATTERN/degree_distribution.dat         | xargs ruby ${ensemble_averaging_rb} -f > $OUT_DIR/degree_distribution_ave.dat
ls $PATTERN/neighbor_degree_correlation.dat | xargs ruby ${ensemble_averaging_rb}    > $OUT_DIR/neighbor_degree_correlation_ave.dat

cd $OUT_DIR
gnuplot $script_dir/cc_degree_correlation_ave.plt
gnuplot $script_dir/degree_distribution_ave.plt
gnuplot $script_dir/neighbor_degree_correlation_ave.plt
cd -

if ls $PATTERN/edge_weight_distribution.dat 1> /dev/null 2>&1; then  # if file exists
  ls $PATTERN/edge_weight_distribution.dat    | xargs ruby ${ensemble_averaging_rb} -f -l    > $OUT_DIR/edge_weight_distribution_ave.dat
  ls $PATTERN//link_removal_percolation.dat   | xargs ruby ${ensemble_averaging_rb} -b 0.002 > $OUT_DIR/link_removal_percolation_ave.dat
  ls $PATTERN/overlap_weight_correlation.dat  | xargs ruby ${ensemble_averaging_rb} -l       > $OUT_DIR/overlap_weight_correlation_ave.dat
  ls $PATTERN/strength_degree_correlation.dat | xargs ruby ${ensemble_averaging_rb}          > $OUT_DIR/strength_degree_correlation_ave.dat
  ls $PATTERN/strength_distribution.dat       | xargs ruby ${ensemble_averaging_rb} -f -l    > $OUT_DIR/strength_distribution_ave.dat

  cd $OUT_DIR
  gnuplot $script_dir/edge_weight_distribution_ave.plt
  gnuplot $script_dir/link_removal_percolation_ave.plt
  gnuplot $script_dir/overlap_weight_correlation_ave.plt
  gnuplot $script_dir/strength_distribution_ave.plt
  gnuplot $script_dir/strength_distribution_ave.plt
fi

