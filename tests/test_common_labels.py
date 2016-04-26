import pytest


def test_load_classmap(classmap):
    assert classmap is not None


def test_classmap_allnames(classmap):
    assert isinstance(classmap.allnames, list)
    assert len(classmap.allnames) > 1
    assert map(lambda x: isinstance(x, str), classmap.allnames)


def test_classmap_classnames(classmap):
    assert isinstance(classmap.classnames, list)
    assert len(classmap.classnames) == 12
    assert map(lambda x: isinstance(x, str), classmap.allnames)


def test_classmap_getattr(classmap):
    assert classmap["bassoon"] == "bassoon"
    assert classmap["acoustic-guitar"] == "guitar"
    assert classmap["Trumpet"] == "trumpet"


def test_classmap_index(classmap):
    assert classmap.get_index("bassoon") == 0
    assert classmap.from_index(0) == "bassoon"
    assert classmap.get_index("violin") == 11
    assert classmap.from_index(11) == "violin"
    assert classmap.size == 12


def test_classmap_indeces_match(classmap):
    for i in range(classmap.size):
        classname = classmap.from_index(i)
        assert classmap.get_index(classname) == i
