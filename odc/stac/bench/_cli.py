"""CLI app for benchmarking."""
import json
from typing import Any, Dict, Optional

import click

from odc.stac.bench import SAMPLE_SITES, dump_site


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


@main.command("run")
def run():
    """Run data load benchmark using Dask."""
    print("TODO")


@main.command("report")
def report():
    """Assemble report."""
    print("TODO")
