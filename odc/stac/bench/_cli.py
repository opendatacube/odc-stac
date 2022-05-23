"""CLI app for benchmarking."""
import json
from datetime import datetime
from time import sleep
from typing import Any, Dict, Optional

import click
import distributed
import rasterio.enums

from odc.stac.bench import (
    SAMPLE_SITES,
    BenchLoadParams,
    dump_site,
    load_from_json,
    load_results,
    run_bench,
)

# pylint: disable=too-many-arguments,too-many-locals

RIO_RESAMPLING_NAMES = [it.name for it in rasterio.enums.Resampling]


@click.group("odc-stac-bench")
def main():
    """Benchmarking tool for odc.stac."""


@main.command("prepare")
@click.option("--sample-site", type=str, help="Use one of sample sites")
@click.option(
    "--list-sample-sites",
    is_flag=True,
    default=False,
    help="Print available sample sites",
)
@click.option(
    "--from-file",
    help="From json config file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option("--overwrite", is_flag=True, help="Overwite output file")
def prepare(sample_site, list_sample_sites, from_file, overwrite):
    """Prepare benchmarking dataset."""
    if list_sample_sites:
        click.echo("Sample sites:")
        for site_name in SAMPLE_SITES:
            click.echo(f"   {site_name}")
        return

    site: Optional[Dict[str, Any]] = None
    if sample_site is not None:
        site = SAMPLE_SITES.get(sample_site, None)
        if site is None:
            raise click.ClickException(f"No such site: {sample_site}")
        print("Site config:")
        print("------------------------------------------")
        print(json.dumps(site, indent=2))
        print("------------------------------------------")
    elif from_file is not None:
        with open(from_file, "rt", encoding="utf8") as src:
            site = json.load(src)

    if site is None:
        raise click.ClickException("Have to supply one of --sample-site or --from-file")
    dump_site(site, overwrite=overwrite)


@main.command("dask")
@click.option(
    "--n-workers", type=int, default=1, help="Number of workers to launch (1)"
)
@click.option(
    "--threads-per-worker", type=int, help="Number of threads per worker (all cpus)"
)
@click.option("--memory-limit", type=str, help="Configure worker memory limit")
def _dask(n_workers, threads_per_worker, memory_limit):
    """Launch local Dask Cluster."""
    client = distributed.Client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
    )
    info = client.scheduler_info()
    print(f"Launched Dask Cluster: {info['address']}")
    print(f"   --scheduler='{info['address']}'")
    while True:
        try:
            sleep(1)
        except KeyboardInterrupt:
            print("Terminating")
            client.shutdown()
            return


@main.command("run")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=False,
    help="Experiment configuration in json format",
)
@click.option(
    "--ntimes", "-n", type=int, default=1, help="Configure number of times to run"
)
@click.option(
    "--method",
    help="Data loading method",
    type=click.Choice(["odc-stac", "stackstac"]),
)
@click.option("--bands", type=str, help="Comma separated list of bands")
@click.option("--chunks", type=int, help="Chunk size Y,X order", nargs=2)
@click.option("--resolution", type=float, help="Set output resolution")
@click.option("--crs", type=str, help="Set CRS")
@click.option(
    "--resampling",
    help="Resampling method when changing resolution/projection",
    type=click.Choice(RIO_RESAMPLING_NAMES),
)
@click.option("--show-config", is_flag=True, help="Show configuration only, don't run")
@click.option(
    "--scheduler", default="tcp://localhost:8786", help="Dask server to connect to"
)
@click.argument("site", type=click.Path(exists=True, dir_okay=False, readable=True))
def run(
    site,
    config,
    method,
    ntimes,
    bands,
    chunks,
    resolution,
    crs,
    resampling,
    show_config,
    scheduler,
):
    """
    Run data load benchmark using Dask.

    SITE is a GeoJSON file produced by `prepare` step.
    """
    cfg: Optional[BenchLoadParams] = None
    if config is not None:
        with open(config, "rt", encoding="utf8") as src:
            cfg = BenchLoadParams.from_json(src.read())
    else:
        cfg = BenchLoadParams(
            method="odc-stac",
            chunks=(2048, 2048),
            extra={
                "stackstac": {"dtype": "uint16", "fill_value": 0},
                "odc-stac": {
                    "groupby": "solar_day",
                    "stac_cfg": {"*": {"warnings": "ignore"}},
                },
            },
        )

    if chunks:
        cfg.chunks = chunks
    if method is not None:
        cfg.method = method
    if bands is not None:
        cfg.bands = tuple(bands.split(","))
    if resolution is not None:
        cfg.resolution = resolution
    if crs is not None:
        cfg.crs = crs
    if resampling is not None:
        cfg.resampling = resampling
    if not cfg.scenario:
        cfg.scenario = site.rsplit(".", 1)[0]

    with open(site, "rt", encoding="utf8") as src:
        site_geojson = json.load(src)

    print(f"Loaded: {len(site_geojson['features'])} STAC items from '{site}'")

    print("Will use following load configuration")
    print("-" * 60)
    print(cfg.to_json(indent=2))
    print("-" * 60)

    if show_config:
        return

    print(f"Connecting to Dask Scheduler: {scheduler}")
    client = distributed.Client(scheduler)

    print("Constructing Dask graph")
    xx = load_from_json(site_geojson, cfg)
    print(f"Starting benchmark run ({ntimes} runs)")
    print("=" * 60)

    ts = datetime.now().strftime("%Y%m%dT%H%M%S.%f")
    results_file = f"{cfg.scenario}_{ts}.pkl"
    print(f"Will write results to: {results_file}")
    _ = run_bench(xx, client, ntimes=ntimes, results_file=results_file)
    print("=" * 60)
    print("Finished")


@main.command("report")
@click.option(
    "--matching", type=str, help="Supply glob pattern instead of individual .pkl files"
)
@click.option(
    "--output",
    type=str,
    help="File to write CSV data, if not supplied will write to stdout",
)
@click.argument(
    "pkls", type=click.Path(exists=True, dir_okay=False, readable=True), nargs=-1
)
def report(matching, output, pkls):
    """
    Collate results of multiple benchmark experiments.

    Read pickle files produced by the `run` command and assemble
    them into one CSV file.
    """
    if matching is not None:
        data_raw = load_results(matching)
    else:
        data_raw = load_results(pkls)

    if output is None:
        print(data_raw.to_csv())
    else:
        data_raw.to_csv(output)
