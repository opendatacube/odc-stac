#!/bin/bash

# Expects percent script notebook on stdin
# produces rendered ipynb notebook on stdout

jupytext --from 'py:percent' --to ipynb -o - \
  | jupyter nbconvert \
      --stdin \
      --to notebook \
      --stdout \
      --ExecutePreprocessor.store_widget_state=True \
      --execute
