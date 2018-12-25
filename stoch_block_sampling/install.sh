#!/bin/bash

set -ex

script_dir=$(cd $(dirname $0); pwd)
cd "$script_dir"
git submodule update --init --recursive
cd "$script_dir/simulator" && make
cd "$script_dir/network_analysis" && make

if [ -n "$OACIS_ROOT" ];
then
  "$OACIS_ROOT/bin/oacis_ruby" "$script_dir/register_oacis.rb"
fi

