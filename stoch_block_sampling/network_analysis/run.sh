#!/bin/bash
set -eux

NET_FILE=$(cd $(dirname $1) && pwd)/$(basename $1)
OUT_DIR=${2:-.}  # default output dir is the current directory
script_dir=$(cd $(dirname $BASH_SOURCE); pwd)

mkdir -p $OUT_DIR
cd $OUT_DIR
$script_dir/analyzer.out $NET_FILE

