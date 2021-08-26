import datetime
import jinja2


def render_eo3_yaml(task, processing_datetime=None, transform_precision=3):
    """

    task:
       .uuid          UUID
       .product       str
       .period        (datetime, datetime)
       .geobox        GeoBox
       .dss           [Dataset]
       .bands         [str]
       .region_code   str
       .file_prefix   str
    """
    if processing_datetime is None:
        processing_datetime = datetime.datetime.utcnow()

    return _YAML.render(
        task=task,
        transform_precision=transform_precision,
        processing_datetime=processing_datetime,
    )


_YAML = jinja2.Template(
    """---
# Dataset
$schema: https://schemas.opendatacube.org/dataset
id: {{ task.uuid}}

product:
  name: {{ task.product }}
  href: https://collections.dea.ga.gov.au/product/{{ task.product }}

crs: {{ task.geobox.crs }}
grids:
  default:
    shape: [{{ task.geobox.shape[0] }}, {{ task.geobox.shape[1] }}]
    transform: [{% for t in task.geobox.transform[:6] %}{{'%.*f' % (transform_precision, t)}},{% endfor %} 0, 0, 1]

properties:
  odc:region_code: {{ task.region_code }}
  datetime: {{ task.period[0].strftime("%Y-%m-%dT%H:%M:%S.%f") }}
  dtr:start_datetime: {{ task.period[0].strftime("%Y-%m-%dT%H:%M:%S.%f") }}
  dtr:end_datetime: {{ task.period[1].strftime("%Y-%m-%dT%H:%M:%S.%f") }}
  odc:file_format: GeoTIFF
  odc:processing_datetime: {{ processing_datetime.strftime("%Y-%m-%dT%H:%M:%S") }}

measurements:
{% for band in task.bands %}
  {{ band }}:
    path: {{ task.file_prefix }}_{{ band }}.tif
{% endfor %}

lineage:
  inputs:
{% for ds in task.dss %}
  - {{ ds.id }}
{% endfor %}
...
""",
    trim_blocks=True,
)
