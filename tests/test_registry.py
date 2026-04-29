from micro_agent.registry import DatasetEntry, Registry


def test_registry_loads(registry: Registry) -> None:
    assert len(registry) > 0
    assert all(isinstance(e, DatasetEntry) for e in registry.entries)


def test_registry_search_unfiltered_returns_everything(registry: Registry) -> None:
    assert registry.search() == registry.entries


def test_registry_search_by_organelle_is_a_subset(registry: Registry) -> None:
    mito = registry.search(organelle="mito")
    assert isinstance(mito, list)
    assert 0 < len(mito) <= len(registry)
    for e in mito:
        assert any("mito" in o.lower() for o in e.organelles)


def test_registry_search_repository_matches(registry: Registry) -> None:
    hits = registry.search(repository="OpenOrganelle")
    assert hits, "expected at least one OpenOrganelle entry in the curated catalog"
    for e in hits:
        assert "openorganelle" in e.repository.lower()


def test_registry_search_has_segmentation_true(registry: Registry) -> None:
    with_seg = registry.search(has_segmentation=True)
    for e in with_seg:
        assert e.has_segmentation is True


def test_registry_list_methods_return_sorted_unique(registry: Registry) -> None:
    for items in (
        registry.list_organelles(),
        registry.list_organisms(),
        registry.list_repositories(),
    ):
        assert items == sorted(items)
        assert len(items) == len(set(items))
