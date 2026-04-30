import pytest

from micro_agent.backends import (
    Backend,
    BioImageBackend,
    EMPIARBackend,
    IDRBackend,
    MICrONSBackend,
    OpenOrganelleBackend,
)
from micro_agent.loader import _get_backend


@pytest.mark.parametrize(
    "repository,backend_cls",
    [
        ("OpenOrganelle", OpenOrganelleBackend),
        ("MICrONS", MICrONSBackend),
        ("FlyEM", MICrONSBackend),
        ("Google", MICrONSBackend),
        ("OpenNeuroData", MICrONSBackend),
        ("BossDB", MICrONSBackend),
        ("EMPIAR", EMPIARBackend),
        ("IDR", IDRBackend),
        ("BioImage Archive", IDRBackend),
        ("Allen", BioImageBackend),
        ("HPA", BioImageBackend),
        ("CellImageLibrary", BioImageBackend),
        ("Zenodo", BioImageBackend),
    ],
)
def test_backend_dispatch(repository: str, backend_cls: type[Backend]) -> None:
    assert isinstance(_get_backend(repository), backend_cls)


def test_backend_dispatch_unknown_raises() -> None:
    with pytest.raises(NotImplementedError, match="Available:"):
        _get_backend("NotAReal_Repository_X7")


@pytest.mark.parametrize(
    "backend_cls",
    [
        OpenOrganelleBackend,
        MICrONSBackend,
        EMPIARBackend,
        IDRBackend,
        BioImageBackend,
    ],
)
def test_backend_is_Backend_subclass(backend_cls: type[Backend]) -> None:
    assert issubclass(backend_cls, Backend)
