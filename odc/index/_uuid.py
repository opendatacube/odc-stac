import uuid
from uuid import UUID
from typing import Sequence

# Some random UUID to be ODC namespace
ODC_NS = UUID("6f34c6f4-13d6-43c0-8e4e-42b6c13203af")


def odc_uuid(
    algorithm: str,
    algorithm_version: str,
    sources: Sequence[UUID],
    deployment_id: str = "",
    **other_tags
) -> UUID:
    """Generate deterministic UUID for a derived Dataset

    :param algorithm: Name of the algorithm
    :param algorithm_version: Version string of the algorithm
    :param sources: Sequence of input Dataset UUIDs
    :param deployment_id: Some sort of identifier for installation that performs
                          the run, for example Docker image hash, or dea module version on NCI.
    :param **other_tags: Any other identifiers necessary to uniquely identify dataset
    """
    tags = ["{key}={value}".format(key=k, value=str(v)) for k, v in other_tags.items()]

    ss = (
        [str(algorithm), str(algorithm_version), str(deployment_id)]
        + sorted(tags)
        + [str(u) for u in sorted(sources)]
    )

    srcs_hashes = "\n".join(s.lower() for s in ss)
    return uuid.uuid5(ODC_NS, srcs_hashes)
