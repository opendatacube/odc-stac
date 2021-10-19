#!/bin/bash

src=$1
dst=${2:-${src%%.py}.ipynb}

echo "$src -> $dst"

jupytext $src --to ipynb -o - \
  | jupyter nbconvert \
      --stdin \
      --to notebook \
      --stdout \
      --ExecutePreprocessor.store_widget_state=True \
      --execute > "${dst}"
