#!/usr/bin/env bash
set -euo pipefail

mamba env create -f environment.yml -v
mamba run -n warp pip install "nnunetv2>=2.3.1"
mamba run -n warp pip install "TotalSegmentator>=2.5" --no-deps
mamba run -n warp pip install "git+ssh://git@github.com/uncbiag/uniGradICONLung.git"
mamba run -n warp pip check

echo "Done"
