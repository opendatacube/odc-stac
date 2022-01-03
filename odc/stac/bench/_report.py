"""Helper methods for benchmark reporting."""
import glob
import pickle
from typing import Any, Dict, Iterable, Iterator, Union

import pandas as pd


def load_results(
    sources: Union[str, Iterable[str]],
) -> pd.DataFrame:
    """
    Load benchmark run results.

    :param sources: A glob pattern or a stream of pickle file paths
    :return: Pandas dataframe
    """

    def _stream(paths: Iterable[str]) -> Iterator[Dict[str, Any]]:
        for idx, fname in enumerate(paths):
            with open(fname, "rb") as src:
                dd = pickle.load(src)
            ctx = dd["context"]
            samples = dd["samples"]
            rr = ctx.to_pandas_dict()

            for sample in samples:
                t0, t1, t2 = sample
                yield {"experiment": idx, **rr, "t0": t0, "t1": t1, "t2": t2}

    if isinstance(sources, str):
        # glob
        pkl_paths: Iterable[str] = sorted(glob.glob(sources))
    else:
        pkl_paths = sources

    xx = pd.DataFrame(list(_stream(pkl_paths)))
    xx = xx.set_index("experiment")
    xx["submit"] = xx.t1 - xx.t0
    xx["elapsed"] = xx.t2 - xx.t0
    return xx
