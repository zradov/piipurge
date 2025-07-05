import re
import types
import pytest
from importlib import import_module
from unittest.mock import patch, MagicMock
from src.piipurge.documents_processor import (
    get_closest_ent_name, _update_subs, _should_process_entity, 
)


@pytest.fixture(autouse=True)
def patch_consts(monkeypatch):
    dummy_consts = types.SimpleNamespace()
    dummy_consts.PATTERNS = {
        "url": re.compile(r"https?://[^\s]+"),
        "email": re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"),
    }
    monkeypatch.setattr("src.piipurge.documents_processor.consts", dummy_consts)
    yield


def test_get_closest_ent_name_relation_none(monkeypatch):
    monkeypatch.setattr(
        "src.piipurge.documents_processor.get_most_similar_text",
        lambda texts, candidates, encoder: None
    )
    mock_encoder = object()
    result = get_closest_ent_name(["foo"], {"bar"}, [("bar", "baz")], mock_encoder)

    assert result is None


def test_get_closest_ent_name_when_no_relations_matched(monkeypatch):
    monkeypatch.setattr(
        "src.piipurge.documents_processor.get_most_similar_text",
        lambda texts, candidates, encoder: ("National Aeronautics and Space Administration", "NASA")
    )
    mock_encoder = object()
    result = get_closest_ent_name(["foo"], {"bar"}, [], mock_encoder)

    assert result is None


@pytest.fixture
def dummy_entity():
    return type("Span", (), {"label_": "ORG", "text": "NASA"})()


@pytest.fixture
def dummy_paragraph():
    return {
        "index": 0,
        "text": "NASA is an organization.",
        "lines": [],
        "page_number": 0,
    }


def test_update_subs_skips_entity_when_should_not_process(monkeypatch, dummy_entity, dummy_paragraph):
    monkeypatch.setattr("src.piipurge.documents_processor._should_process_entity", lambda ent, acronyms, encoder: False)
    subs = {}
    ents = [dummy_entity]
    relations = []
    common_acronyms = []
    text_encoder = object()
    _update_subs(subs, ents, relations, common_acronyms, text_encoder, dummy_paragraph)

    assert subs == {}


def test_update_subs_calls_process_org_entity(monkeypatch, dummy_entity, dummy_paragraph):
    monkeypatch.setattr("src.piipurge.documents_processor._should_process_entity", lambda ent, acronyms, encoder: True)
    monkeypatch.setattr(
        "src.piipurge.documents_processor.consts",
        type("Dummy", (), {"ENTITY_DESC": {"ORG": ("Organization", str)}})
    )
    called = {}
    def fake_process_org_entity(ent, relations, ent_subs, text_encoder, suffix_type, paragraph):
        called["called"] = True
        ent_subs["NASA"] = ["dummy"]
    monkeypatch.setattr("src.piipurge.documents_processor._process_org_entity", fake_process_org_entity)
    subs = {}
    ents = [dummy_entity]
    relations = [("NASA", "National Aeronautics and Space Administration")]
    common_acronyms = []
    text_encoder = object()
    _update_subs(subs, ents, relations, common_acronyms, text_encoder, dummy_paragraph)
    
    assert called.get("called") is True
    assert "ORG" in subs
    assert subs["ORG"]["NASA"] == ["dummy"]


def test_update_subs_calls_process_standard_entity(monkeypatch, dummy_paragraph):
    entity = type("Span", (), {"label_": "PERSON", "text": "Alice"})()
    monkeypatch.setattr("src.piipurge.documents_processor._should_process_entity", lambda ent, acronyms, encoder: True)
    monkeypatch.setattr(
        "src.piipurge.documents_processor.consts",
        type("Dummy", (), {"ENTITY_DESC": {"PERSON": ("Person", int)}})
    )
    called = {}
    def fake_process_standard_entity(ent, ent_subs, suffix_type, paragraph):
        called["called"] = True
        ent_subs["Alice"] = ["dummy"]
    monkeypatch.setattr("src.piipurge.documents_processor._process_standard_entity", fake_process_standard_entity)
    subs = {}
    ents = [entity]
    relations = []
    common_acronyms = []
    text_encoder = object()
    _update_subs(subs, ents, relations, common_acronyms, text_encoder, dummy_paragraph)
    
    assert called.get("called") is True
    assert "PERSON" in subs
    assert subs["PERSON"]["Alice"] == ["dummy"]


def test_update_subs_multiple_entities(monkeypatch, dummy_paragraph):
    org_entity = type("Span", (), {"label_": "ORG", "text": "NASA"})()
    person_entity = type("Span", (), {"label_": "PERSON", "text": "Alice"})()
    monkeypatch.setattr("src.piipurge.documents_processor._should_process_entity", lambda ent, acronyms, encoder: True)
    monkeypatch.setattr(
        "src.piipurge.documents_processor.consts",
        type("Dummy", (), {"ENTITY_DESC": {"ORG": ("Organization", str), "PERSON": ("Person", int)}})
    )
    def fake_process_org_entity(ent, relations, ent_subs, text_encoder, suffix_type, paragraph):
        ent_subs["NASA"] = ["dummy_org"]
    def fake_process_standard_entity(ent, ent_subs, suffix_type, paragraph):
        ent_subs["Alice"] = ["dummy_person"]
    monkeypatch.setattr("src.piipurge.documents_processor._process_org_entity", fake_process_org_entity)
    monkeypatch.setattr("src.piipurge.documents_processor._process_standard_entity", fake_process_standard_entity)
    subs = {}
    ents = [org_entity, person_entity]
    relations = []
    common_acronyms = []
    text_encoder = object()
    _update_subs(subs, ents, relations, common_acronyms, text_encoder, dummy_paragraph)
    
    assert "ORG" in subs and "PERSON" in subs
    assert subs["ORG"]["NASA"] == ["dummy_org"]
    assert subs["PERSON"]["Alice"] == ["dummy_person"]
    
    
@pytest.mark.parametrize(
    "label,entity_desc,acronym_return,expected",
    [
        # Not in ENTITY_DESC
        ("UNKNOWN", {}, None, False),
        # ORG, acronym found
        ("ORG", {"ORG": ("Organization", str)}, "NASA", False),
        # ORG, acronym not found
        ("ORG", {"ORG": ("Organization", str)}, None, True),
        # PERSON, not ORG, in ENTITY_DESC
        ("PERSON", {"PERSON": ("Person", int)}, None, True),
    ]
)
def test_should_process_entity(monkeypatch, label, entity_desc, acronym_return, expected):
    monkeypatch.setattr(
        "src.piipurge.documents_processor.consts",
        type("Dummy", (), {"ENTITY_DESC": entity_desc})
    )
    if label == "ORG":
        monkeypatch.setattr(
            "src.piipurge.documents_processor.check_common_acronyms",
            lambda text, acronyms, encoder: acronym_return
        )
    ent = type("Span", (), {"label_": label, "text": "NASA"})()
    result = _should_process_entity(ent, ["NASA"], object())

    assert result == expected


def test_should_process_entity_calls_check_common_acronyms(monkeypatch):
    called = {}
    def fake_check_common_acronyms(text, acronyms, encoder):
        called["called"] = (text, tuple(acronyms))
        return None
    monkeypatch.setattr(
        "src.piipurge.documents_processor.consts",
        type("Dummy", (), {"ENTITY_DESC": {"ORG": ("Organization", str)}})
    )
    monkeypatch.setattr(
        "src.piipurge.documents_processor.check_common_acronyms",
        fake_check_common_acronyms
    )
    ent = type("Span", (), {"label_": "ORG", "text": "NASA"})()
    _should_process_entity(ent, ["NASA"], object())

    assert called["called"][0] == "NASA"
    assert called["called"][1] == ("NASA",)


