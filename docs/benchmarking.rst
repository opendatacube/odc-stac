Benchmarking Utilities
######################

Module :py:mod:`odc.stac.bench` provides utilities for benchmarking data loading. It is both a
library that can be used directly from a notebook and a command line application.

.. code-block:: none 

   Usage: python -m odc.stac.bench [OPTIONS] COMMAND [ARGS]...
   
     Benchmarking tool for odc.stac.
   
   Options:
     --help  Show this message and exit.
   
   Commands:
     dask     Launch local Dask Cluster.
     prepare  Prepare benchmarking dataset.
     report   Collate results of multiple benchmark experiments.
     run      Run data load benchmark using Dask.


Define Test Site
================

To start you need to define a test site, or use one of the pre-configured examples. Site
configuration is a json file that describes STAC API query and some other metadata. Below is a
definition of the ``s2-ms-mosaic`` sample site.

.. code-block:: json

   {
     "file_id": "s2-ms-mosaic_2020-06-06--P1D",
     "api": "https://planetarycomputer.microsoft.com/api/stac/v1",
     "search": {
       "collections": ["sentinel-2-l2a"],
       "datetime": "2020-06-06",
       "bbox": [ 27.345815, -14.98724, 27.565542, -7.710992]
     }
   }

This would query Planetary Computer STAC API endpoint for Sentinel 2 collection and store results to
a geojson file ``{file_id}.geojson``. Try it now:

.. code-block:: bash

    python -m odc.stac.bench prepare --sample-site s2-ms-mosaic

Command above will write a GeoJSON file to your current directory. We will use this file to run
benchmarks later on.


Prepare Load Configuration
==========================

Let's create base data loading configuration file suitable for running benchmarks with the site
configuration produced previously. Save example below as ``cfg.json``.

.. code-block:: json

   {
     "method": "odc-stac",
     "bands": ["B02", "B03", "B04"],
     "patch_url": "planetary_computer.sas.sign",
     "extra": {
       "stackstac": {
         "dtype": "uint16",
         "fill_value": 0
       },
       "odc-stac": {
         "groupby": "solar_day",
         "stac_cfg": {"*": {"warnings": "ignore"}}
       }
     }
   }

Making your own is simple:

1. Create :py:class:`~odc.stac.bench.BenchLoadParams` object
2. Modify configuration options to match your needs
3. Dump it to JSON

.. code-block:: python3

   from odc.stac.bench import BenchLoadParams
   
   params = BenchLoadParams()
   params.scenario = "web-zoom-8"
   params.bands = ["red", "green", "blue"]
   params.crs = "EPSG:3857"
   params.resolution = 610
   params.chunks = (512, 512)
   params.resampling = "bilinear"
   
   print(params.to_json())


Start Dask Cluster
==================

Before we can run the benchmark we need to have an active Dask cluster. You can connect to a remote
cluster or run a local one. A convenience local Dask cluster launcher is provided. In a separate
shell run this command:

.. code-block:: none

    > python -m odc.stac.bench dask --memory-limit=8GiB

    GDAL_DISABLE_READDIR_ON_OPEN = EMPTY_DIR
    GDAL_HTTP_MAX_RETRY          = 10
    GDAL_HTTP_RETRY_DELAY        = 0.5
    GDAL_DATA                    = /srv/conda/envs/notebook/share/gdal
    Launched Dask Cluster: tcp://127.0.0.1:43677
       --scheduler='tcp://127.0.0.1:43677'

This will start a local Dask cluster, configure GDAL on Dask workers and print out the address of
the Dask scheduler. Leave this running and take a note of the ``--scheduler=...`` option that was
printed out, we will use it the next step.


Run Benchmark
=============

We are now ready to run some benchmarking with the ``run`` command documented below:

.. code-block:: none

   Usage: python -m odc.stac.bench run [OPTIONS] SITE

     Run data load benchmark using Dask.

     SITE is a GeoJSON file produced by `prepare` step.

   Options:
     -c, --config FILE               Experiment configuration in json format
     -n, --ntimes INTEGER            Configure number of times to run
     --method [odc-stac|stackstac]   Data loading method
     --bands TEXT                    Comma separated list of bands
     --chunks INTEGER...             Chunk size Y,X order
     --resolution FLOAT              Set output resolution
     --crs TEXT                      Set CRS
     --resampling [nearest|bilinear|cubic|cubic_spline|lanczos|average|mode|gauss|max|min|med|q1|q3|sum|rms]
                                     Resampling method when changing
                                     resolution/projection
     --show-config                   Show configuration only, don't run
     --scheduler TEXT                Dask server to connect to
     --help                          Show this message and exit.


First let's check configuration, note we will run with the reduced resolution for quicker turn
around (``--resolution=80`` option). Command line arguments take precedence over configuration
parameters supplied in the json file.

.. code-block:: bash

    python -m odc.stac.bench run \
      s2-ms-mosaic_2020-06-06--P1D.geojson \
      --config cfg.json \
      --resolution=80 \
      --show-config

If the above went well we can start the benchmark, remove ``--show-config`` option, and add
``--scheduler=`` option that was printed when we started Dask cluster. Let's also configure number
of benchmarking passes to run with ``-n 10`` option.

.. code-block:: bash

    python -m odc.stac.bench run \
      s2-ms-mosaic_2020-06-06--P1D.geojson \
      --config cfg.json \
      --resolution=80 \
      -n 10 \
      --scheduler='tcp://127.0.0.1:43677' 


.. note::
    
    Don't forget to edit ``--scheduler=``, part of the above command.

This will first print out configuration that will be used,

.. code-block:: none

   Loaded: 9 STAC items from 's2-ms-mosaic_2020-06-06--P1D.geojson'
   Will use following load configuration
   ------------------------------------------------------------
   { /** NOTE: this section was edited for brevity **/
     "scenario": "s2-ms-mosaic_2020-06-06--P1D",
     "method": "odc-stac",
     "chunks": [ 2048, 2048 ],
     "bands": [ "B02", "B03", "B04" ],
     "resolution": 80.0,
     "crs": null,
     "resampling": null,
     "patch_url": "planetary_computer.sas.sign",
     "extra": {
       "stackstac": { "dtype": "uint16", "fill_value": 0 },
       "odc-stac": { "groupby": "solar_day", "stac_cfg": {"*": {"warnings": "ignore" }}}
     }
   }
   ------------------------------------------------------------


followed by information about data being loaded and some stats about the Dask cluster on which the
benchmark will run:

.. code-block:: none

   Connecting to Dask Scheduler: tcp://127.0.0.1:43677
   Constructing Dask graph
   Starting benchmark run (10 runs)
   ============================================================
   Will write results to: s2-ms-mosaic_2020-06-06--P1D_20220104T080235.133458.pkl
   method      : odc-stac
   Scenario    : s2-ms-mosaic_2020-06-06--P1D
   T.slice     : 2020-06-06
   Data        : 1.3.11373.1374.uint16,  89.42 MiB
   Chunks      : 1.1.2048.1374 (T.B.Y.X)
   GEO         : epsg:32735
               | 80, 0, 499920|
               | 0,-80, 9200080|
   Cluster     : 1 workers, 4 threads, 8.00 GiB 
   ------------------------------------------------------------

As benchmark runs are completed brief summaries are printed:

.. code-block:: none

   T.Elapsed   :    2.845 seconds
   T.Submit    :    0.228 seconds
   Throughput  :   16.480 Mpx/second (overall)
               |    4.120 Mpx/second (per thread)
   ------------------------------------------------------------
   T.Elapsed   :    2.448 seconds
   T.Submit    :    0.015 seconds
   Throughput  :   19.152 Mpx/second (overall)
               |    4.788 Mpx/second (per thread)
   ... continues

You can terminate early without losing data with ``Ctrl-C``. Benchmark results are saved after each
benchmark pass (overwriting previous save-point) in case there is a crash or some other fatal
error.


Review Results
==============

To convert benchmark results stored in ``.pkl`` file(s) to CSV use the following:

.. code-block:: bash

   python -m odc.stac.bench report *.pkl --output results.csv

The idea is to run benchmarks with different load configurations, different chunk sizes for example,
or comparing relative costs of resampling modes, then combine those into one data table.
