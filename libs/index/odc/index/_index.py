from datacube.index.hl import Doc2Dataset
from odc.io.text import parse_yaml


def from_metadata_stream(metadata_stream, index, **kwargs):
    """
    Given a stream of (uri, metadata) tuples convert them into Datasets, using
    supplied index and options for Doc2Dataset.


    **kwargs**:
    - skip_lineage
    - verify_lineage
    - fail_on_missing_lineage
    - products
    - exclude_products

    returns a sequence of tuples where each tuple is either

        (Dataset, None) or (None, error_message)
    """
    doc2ds = Doc2Dataset(index, **kwargs)

    for uri, metadata in metadata_stream:
        if metadata is None:
            yield (None, 'Error: empty doc %s' % (uri))
        else:
            ds, err = doc2ds(metadata, uri)
            if ds is not None:
                yield (ds, None)
            else:
                yield (None, 'Error: %s, %s' % (uri, err))


def parse_doc_stream(doc_stream, on_error=None):
    """Replace doc bytes/strings with parsed dicts.

    doc_stream -- sequence of (uri, doc: byges|string)
    on_error -- uri, doc, exception -> None

    On output doc is replaced with python dict parsed from yaml, or with None
    if parsing error occured.
    """
    for uri, doc in doc_stream:
        try:
            metadata = parse_yaml(doc)
        except Exception as e:
            if on_error is not None:
                on_error(uri, doc, e)
            metadata = None

        yield uri, metadata


def from_yaml_doc_stream(doc_stream, index, logger=None, **kwargs):
    """ returns a sequence of tuples where each tuple is either

        (Dataset, None) or (None, error_message)
    """
    def on_parse_error(uri, doc, err):
        if logger is not None:
            logger.error("Failed to parse: %s", uri)

    metadata_stream = parse_doc_stream(doc_stream, on_error=on_parse_error)

    return from_metadata_stream(metadata_stream, index, **kwargs)
