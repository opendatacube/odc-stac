# Sample Notebooks


## Developer Notes

Do not commit `*.ipynb` files here! We use `jupytext` for keeping notebooks in
version control, specifically "py:percent" format. Install `jupytext` into your
jupyterlab environment, then you should be able to "Open With->Notebook" on
these `.py` files.

To create a new one, start with a notebook file (`.ipynb`) then use "Pair
Notebook with percent Script" command (type `Ctr-Shift-C` when editing notebook,
then start typing "percent" to fuzzy find the command)


## Rendered Notebooks

Notebooks are executed by github action and results are uploaded to:

```
s3://datacube-core-deployment/odc-stac/nb/odc-stac-notebooks-{nb_hash}.tar.gz
https://packages.dea.ga.gov.au/odc-stac/nb/odc-stac-notebooks-{nb_hash}.tar.gz
```

Where `{nb_hash}` is a 16 character hash computed from the content of `notebooks/*.py` (see `scripts/notebook_hash.py`).

By the time changes are merged into `develop` branch there should be
pre-rendered notebook archive accessible without authentication via https.
Building documentation on read the docs site will use that archive rather than
attempting to run notebooks directly.
