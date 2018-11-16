#!/bin/bash

set -eux

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
ls _input/*/pk.dat | xargs ruby $script_dir/../ensemble_averaging/ensemble_averaging.rb -f > pk_ave.dat
ls _input/*/ck.dat | xargs ruby $script_dir/../ensemble_averaging/ensemble_averaging.rb > ck_ave.dat
ls _input/*/knn.dat | xargs ruby $script_dir/../ensemble_averaging/ensemble_averaging.rb > knn_ave.dat

PIPFILE=$script_dir/../Pipfile
export PIPENV_PIPFILE=$(cd $(dirname $PIPFILE) && pwd)/$(basename $PIPFILE)
pipenv run python $script_dir/make_plot_ave.py
