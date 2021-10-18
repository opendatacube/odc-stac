"""Tools for dealing with UTM grids."""
from types import SimpleNamespace
from typing import Optional, Tuple, Union

from datacube.model import GridSpec
from datacube.utils.geometry import CRS


def mk_utm_gs(
    epsg: int,
    resolution: Union[Tuple[float, float], float] = 10,
    pixels_per_cell: int = 10_000,
    origin: Tuple[float, float] = (0, 0),
) -> GridSpec:
    """Construct GridSpec."""
    if not isinstance(resolution, tuple):
        resolution = (-resolution, resolution)

    ty, tx = (abs(r) * pixels_per_cell for r in resolution)

    return GridSpec(
        crs=CRS(f"epsg:{epsg}"),
        resolution=resolution,
        tile_size=(ty, tx),
        origin=origin,
    )


def utm_region_code(
    epsg: Union[int, Tuple[int, int, int]], tidx: Optional[Tuple[int, int]] = None
) -> str:
    """
    Construct UTM gridspec identifier.

    Examples:
    - 32751          -> "51S"
    - 32633, 10, 2   -> "33N_10_02"
    - 32603, (1, 2)  -> "03N_01_02"
    """
    if isinstance(epsg, tuple):
        tidx = epsg[1:]
        epsg = epsg[0]

    if 32601 <= epsg <= 32660:
        zone, code = epsg - 32600, "N"
    elif 32701 <= epsg <= 32760:
        zone, code = epsg - 32700, "S"
    else:
        raise ValueError(
            f"Not a utm epsg: {epsg}, valid ranges [32601, 32660] and [32701, 32760]"
        )

    if tidx is None:
        return f"{zone:02d}{code}"

    return f"{zone:02d}{code}_{tidx[0]:02d}_{tidx[1]:02d}"


def utm_zone_to_epsg(zone):
    """
    Convert UTM zone string to EPSG code.

    56S -> 32756
    55N -> 32655
    """
    if len(zone) < 2:
        raise ValueError(f'Not a valid zone: "{zone}", expect <int: 1-60><str:S|N>')

    offset = dict(S=32700, N=32600).get(zone[-1].upper())

    if offset is None:
        raise ValueError(f'Not a valid zone: "{zone}", expect <int: 1-60><str:S|N>')

    try:
        i = int(zone[:-1])
    except ValueError:
        i = None

    if i is not None and (i < 0 or i > 60):
        i = None

    if i is None:
        raise ValueError(f'Not a valid zone: "{zone}", expect <int: 1-60><str:S|N>')

    return offset + i


def utm_tile_dss(dss, **gridspec_options):
    """
    Given a sequence of Dataset objects each using UTM projection bin them into tiles.

    Equivalent to:

    - Group Datasets by UTM zone
    - Create GridSpec for every observed UTM zone
    - Bin datasets within a UTM zone group into tiles as specified by GridSpec

    For **gridspec_options see mk_utm_gs
      - resolution
      - align
      - pixels_per_cell

    Returns
    =======

    List of Tile objects, each is:
      .region    : (epsg,  tile_idx_x, tile_idx_y)
      .grid_spec : GridSpec
      .geobox    : GeoBox
      .dss       : [Dataset]

    Tiles are sorted by `.region`
    """
    grid_specs = {}
    tiles = {}

    for ds in dss:
        epsg = ds.crs.epsg
        _cached = grid_specs.get(epsg)
        if _cached is None:
            gs, g_cache = (mk_utm_gs(epsg, **gridspec_options), {})
            grid_specs[epsg] = (gs, g_cache)
        else:
            gs, g_cache = _cached

        for tidx, geobox in gs.tiles_from_geopolygon(ds.extent, geobox_cache=g_cache):
            region = (epsg, *tidx)
            tile = tiles.get(region, None)

            if tile is None:
                tile = SimpleNamespace(
                    region=region, grid_spec=gs, geobox=geobox, dss=[]
                )
                tiles[region] = tile

            tile.dss.append(ds)

    return sorted(tiles.values(), key=lambda t: t.region)
