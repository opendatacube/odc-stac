#!/bin/bash

src=$1
dst=${2:-${src%%.py}.html}

echo "$src -> $dst"

jupytext $src --set-kernel "python3" --to ipynb -o - \
  | jupyter nbconvert \
      --stdin \
      --to html \
      --stdout \
      --ExecutePreprocessor.store_widget_state=True \
      --execute > "${dst}"
