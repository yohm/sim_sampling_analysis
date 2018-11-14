#!/bin/bash

set -eux

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
PIPFILE=$script_dir/../Pipfile
export PIPENV_PIPFILE=$(cd $(dirname $PIPFILE) && pwd)/$(basename $PIPFILE)
python --version
pipenv --version
which python
which pipenv
pipenv run python $script_dir/correlated_h.py _input.json
