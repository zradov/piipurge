import re
import pytest
from spacy.tokens import Doc
from spacy.vocab import Vocab
from src.piipurge import consts
from unittest.mock import MagicMock
from src.piipurge.utils import nlp
from src.piipurge.utils.nlp import (
    is_link, get_spans_similarity, _find_matching_entities, check_common_acronyms,
    _normalize_utf8_text, is_single_entity_annotation, _group_entity_examples, 
    _oversample, _undersample, balance_examples, get_most_similar_text,
    add_utf8_normalization, load_nlp_model, load_nlp_acronyms_model,
    load_text_encoder_model, get_ent_replacement, _find_best_matching_string,
    find_entity_boundaries
)


@pytest.fixture
def sample_doc():
    vocab = Vocab()
    text = "Apple is looking at buying U.K. startup for $1 billion on 2021-09-01."

    return Doc(vocab, words=text.split(" "))


def test_find_matching_entities_with_matches(sample_doc):
    pattern = re.compile(r"Apple|U\.K\.")
    label = "ORG"
    
    result = _find_matching_entities(sample_doc, pattern, label)
    
    assert len(result.ents) > 0
    entities = [ent.text for ent in result.ents]
    assert "Apple" in entities
    assert "U.K." in entities
    for ent in result.ents:
        if ent.text in ["Apple", "U.K."]:
            assert ent.label_ == label


def test_find_matching_entities_with_no_matches(sample_doc):
    pattern = re.compile(r"XYZ")
    label = "ORG"
    
    original_ents = list(sample_doc.ents)
    result = _find_matching_entities(sample_doc, pattern, label)
    
    assert len(result.ents) == len(original_ents)


def test_find_matching_entities_preserves_existing_ents(sample_doc):
    pattern = re.compile(r"\$\d+ billion")
    label = "MONEY"
    sample_doc = _find_matching_entities(sample_doc, pattern, label) 
    new_pattern = re.compile(r"Apple|U\.K\.")
    new_label = "ORG"

    result = _find_matching_entities(sample_doc, new_pattern, new_label)
    
    money_ents = [ent for ent in result.ents if ent.label_ == "MONEY"]
    org_ents = [ent for ent in result.ents if ent.label_ == "ORG"]
    assert len(money_ents) > 0
    assert len(org_ents) > 0


@pytest.mark.parametrize(
    "text,expected",
    [
        ("https://example.com", True),
        ("http://test.org", True),
        ("Visit https://site.com", False),
        ("user@example.com", True),
        ("test@com", False),
        ("not an email", False),
        ("", False),
        ("ftp://example.com", True),
        ("user@domain.co.uk", True),
    ],
)
def test_is_link(text, expected):
    assert is_link(text) == expected


@pytest.fixture
def mock_text_encoder():
    encoder = MagicMock()
    encoder.encode = MagicMock()
    encoder.encode.side_effect = [[0.1, 0.2], [0.3, 0.4]]
    encoder.similarity = MagicMock(return_value=0.9)

    return encoder


def test_get_spans_similarity(mock_text_encoder):
    span1 = "span1"
    span2 = "span2"
    
    similarity_score = get_spans_similarity(span1, span2, mock_text_encoder)

    assert similarity_score == 0.9
    assert mock_text_encoder.encode.call_count == 2
    assert mock_text_encoder.encode.call_args_list[0][0][0] == span1
    assert mock_text_encoder.encode.call_args_list[1][0][0] == span2
    assert mock_text_encoder.similarity.call_count == 1
    mock_text_encoder.similarity.assert_called_once_with([0.1, 0.2], [0.3, 0.4])


@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("A\u00a0B", "A B"),  # no-break space replaced
        ("“Hello”—‘World’", '"Hello"-\'World\''),  # smart quotes and dash
        ("foo\u200Bbar", "foobar"),  # zero-width space removed
        ("foo   bar", "foo bar"),  # whitespace normalized
        ("foo\u2026", "foo..."),  # ellipsis
        ("foo\u00b7bar", "foo-bar"),  # middle dot
        ("\u2013\u2014", "--"),  # en dash and em dash
        ("  foo\tbar\nbaz  ", "foo bar baz"),  # leading/trailing and mixed whitespace
        ("", ""),  # empty string
        ("normal text", "normal text"),  # no changes
    ],
)
def test_normalize_utf8_text(input_text, expected):
    assert _normalize_utf8_text(input_text) == expected


def test_is_single_entity_annotation_is_true():
    annots = ("'Company A' has large number of employees", [(1, 12, "ORG")])

    res = is_single_entity_annotation(annots)

    assert res == True


def test_is_single_entity_annotation_is_false():
    annots = ("The CEO of the 'Company A' is PERSON B.", [(15, 26, "ORG"), (30, 38, "PERSON")])

    res = is_single_entity_annotation(annots)

    assert res == False
    
    
def test_group_entity_examples_basic():
    docs_annots = [
        ("John works at Acme.", [(0, 4, "PERSON"), (15, 19, "ORG")]),
        ("Jane is from XYZ.", [(0, 4, "PERSON"), (14, 17, "ORG")]),
        ("The event is tomorrow.", []),
        ("Contact: john@example.com", [(9, 25, "EMAIL")]),
    ]

    result = _group_entity_examples(docs_annots)

    assert set(result.keys()) == {"PERSON", "ORG", "EMAIL"}
    assert ("John works at Acme.", [(0, 4, "PERSON"), (15, 19, "ORG")]) in result["PERSON"]
    assert ("Jane is from XYZ.", [(0, 4, "PERSON"), (14, 17, "ORG")]) in result["PERSON"]
    assert ("John works at Acme.", [(0, 4, "PERSON"), (15, 19, "ORG")]) in result["ORG"]
    assert ("Jane is from XYZ.", [(0, 4, "PERSON"), (14, 17, "ORG")]) in result["ORG"]
    assert ("Contact: john@example.com", [(9, 25, "EMAIL")]) in result["EMAIL"]


def test_group_entity_examples_empty():
    docs_annots = []

    result = _group_entity_examples(docs_annots)

    assert result == {}


def test_group_entity_examples_multiple_entities_in_sentence():
    docs_annots = [
        ("Alice met Bob at Acme.", [(0, 5, "PERSON"), (10, 13, "PERSON"), (17, 21, "ORG")])
    ]

    result = _group_entity_examples(docs_annots)

    assert set(result.keys()) == {"PERSON", "ORG"}
    assert ("Alice met Bob at Acme.", [(0, 5, "PERSON"), (10, 13, "PERSON"), (17, 21, "ORG")]) in result["PERSON"]
    assert ("Alice met Bob at Acme.", [(0, 5, "PERSON"), (10, 13, "PERSON"), (17, 21, "ORG")]) in result["ORG"]


def test_group_entity_examples_output_structure():
    docs_annots = [
        ("Foo bar.", [(0, 3, "test1")]),
        ("Bar baz.", [(0, 3, "test2")]),
    ]

    result = _group_entity_examples(docs_annots)

    for _, items in result.items():
        for item in items:
            assert isinstance(item, tuple)
            assert isinstance(item[0], str)
            assert isinstance(item[1], list)


def test_oversample_no_oversampling(monkeypatch):
    annots = {
        "PERSON": [("A", [(0, 1, "PERSON")]), ("B", [(0, 1, "PERSON")])],
        "ORG": [("C", [(0, 1, "ORG")]), ("D", [(0, 1, "ORG")])]
    }
    monkeypatch.setattr(nlp, "generate_text", lambda x, y: pytest.fail("Should not be called"))
    
    result = _oversample(annots, max_majority_minority_ratio=2.0)
    
    assert result == annots


def test_oversample_with_oversampling(monkeypatch):
    annots = {
        "PERSON": [("A", [(0, 1, "PERSON")]), ("B", [(0, 1, "PERSON")]), ("C", [(0, 1, "PERSON")]), ("D", [(0, 1, "PERSON")])],
        "ORG": [("E", [(0, 1, "ORG")])]
    }
    called = {}
    def fake_generate_text(examples, num_aug):
        called["args"] = (examples, num_aug)
        return [("E_aug", [(0, 1, "ORG")])] * (num_aug * len(examples))
    monkeypatch.setattr(nlp, "generate_text", fake_generate_text)

    result = _oversample(annots, max_majority_minority_ratio=2.0)

    print(f"called: {called}")

    assert result["PERSON"] == annots["PERSON"]
    assert result["ORG"] == [("E_aug", [(0, 1, "ORG")])] * (called["args"][1] * len(called["args"][0]))
    assert called["args"][0] == [("E", [(0, 1, "ORG")])]
    assert called["args"][1] >= 5 


@pytest.mark.parametrize("annots", [{}, None,])
def test_oversample_empty_input(annots):
    result = _oversample(annots, max_majority_minority_ratio=2.0)

    assert result == {}


def test_oversample_single_entity_type(monkeypatch):
    annots = {
        "PERSON": [("A", [(0, 1, "PERSON")]), ("B", [(0, 1, "PERSON")])]
    }
    monkeypatch.setattr(nlp, "generate_text", lambda x, y: pytest.fail("Should not be called"))
    
    result = _oversample(annots, max_majority_minority_ratio=1.5)
    
    assert result == annots


def test_oversample_multiple_minority_types(monkeypatch):
    annots = {
        "PERSON": [("A", [(0, 1, "PERSON")]), ("B", [(0, 1, "PERSON")]), ("C", [(0, 1, "PERSON")]), ("D", [(0, 1, "PERSON")])],
        "ORG": [("E", [(0, 1, "ORG")])],
        "EMAIL": [("F", [(0, 1, "EMAIL")])]
    }
    calls = []
    def fake_generate_text(examples, num_aug):
        calls.append((examples, num_aug))
        return [("AUG", [(0, 1, examples[0][1][0][2])])] * (num_aug * len(examples))
    monkeypatch.setattr(nlp, "generate_text", fake_generate_text)
    
    result = _oversample(annots, max_majority_minority_ratio=2.0)
    
    assert result["PERSON"] == annots["PERSON"]
    assert result["ORG"] == [("AUG", [(0, 1, "ORG")])] * (calls[0][1] * len(calls[0][0]))
    assert result["EMAIL"] == [("AUG", [(0, 1, "EMAIL")])] * (calls[1][1] * len(calls[1][0]))


def test_undersample_no_undersampling():
    annots = {
        "PERSON": [("A", [(0, 1, "PERSON")]), ("B", [(0, 1, "PERSON")])],
        "ORG": [("C", [(0, 1, "ORG")]), ("D", [(0, 1, "ORG")])]
    }
    
    result = _undersample(annots, max_majority_minority_ratio=2.0)

    assert result == annots


def test_undersample_with_undersampling(monkeypatch):
    annots = {
        "PERSON": [("A", [(0, 1, "PERSON")]), 
                   ("B", [(0, 1, "PERSON")]), 
                   ("C", [(0, 1, "PERSON")]), 
                   ("D", [(0, 1, "PERSON")])],
        "ORG": [("E", [(0, 1, "ORG")])]
    }
    monkeypatch.setattr("random.sample", lambda population, k: population[:k])
    result = _undersample(annots, max_majority_minority_ratio=2.0)

    assert result["ORG"] == annots["ORG"]
    assert result["PERSON"] == annots["PERSON"][:2]


@pytest.mark.parametrize("annots", [{}, None])
def test_undersample_empty_input(annots):
    result = _undersample(annots, max_majority_minority_ratio=2.0)

    assert result == {}


def test_undersample_single_entity_type():
    annots = {
        "PERSON": [("A", [(0, 1, "PERSON")]), ("B", [(0, 1, "PERSON")])]
    }
    
    result = _undersample(annots, max_majority_minority_ratio=1.5)
    
    assert result == annots


def test_undersample_multiple_majority_types(monkeypatch):
    annots = {
        "PERSON": [("A", [(0, 1, "PERSON")]), ("B", [(0, 1, "PERSON")]), ("C", [(0, 1, "PERSON")]), ("D", [(0, 1, "PERSON")])],
        "ORG": [("E", [(0, 1, "ORG")]), ("F", [(0, 1, "ORG")]), ("G", [(0, 1, "ORG")])],
        "EMAIL": [("H", [(0, 1, "EMAIL")])]
    }
    monkeypatch.setattr("random.sample", lambda population, k: population[:k])
    
    result = _undersample(annots, max_majority_minority_ratio=2.0)
    
    assert result["EMAIL"] == annots["EMAIL"]
    assert result["PERSON"] == annots["PERSON"][:2]
    assert result["ORG"] == annots["ORG"][:2]


def test_undersample_output_structure():
    annots = {
        "test1": [("Foo bar.", [(0, 3, "test1")]), ("Baz foo.", [(0, 3, "test1")])],
        "test2": [("Bar baz.", [(0, 3, "test2")])]
    }
    
    result = _undersample(annots, max_majority_minority_ratio=2.0)
    
    for _, items in result.items():
        for item in items:
            assert isinstance(item, tuple)
            assert isinstance(item[0], str)
            assert isinstance(item[1], list)


@pytest.mark.parametrize("docs_annots,groupped_annots,max_ratio", [
    (
        [
            ("John works at Acme.", [(0, 4, "PERSON")]),
            ("Jane is from XYZ.", [(0, 4, "PERSON")]),
            ("Contact: john@example.com", [(9, 25, "EMAIL")]),
            ("The CEO of Acme is John.", [(11, 15, "ORG"), (22, 26, "PERSON")])
        ],
        {
            "PERSON": [("John works at Acme.", [(0, 4, "PERSON")]), ("Jane is from XYZ.", [(0, 4, "PERSON")])],
            "ORG": [("The CEO of Acme is John.", [(11, 15, "ORG"), (22, 26, "PERSON")])],
            "EMAIL": [("Contact: john@example.com", [(9, 25, "EMAIL")])]
        },
        2.0
    ),
    (
        [
            ("A", [(0, 1, "PERSON")]),
            ("B", [(0, 1, "PERSON")]),
            ("C", [(0, 1, "ORG")]),
            ("D", [(0, 1, "ORG")]),
            ("E", [(0, 1, "ORG")]),
            ("F", [(0, 1, "EMAIL")])
        ],
        {
            "PERSON": [("A", [(0, 1, "PERSON")]), ("B", [(0, 1, "PERSON")])],
            "ORG": [("C", [(0, 1, "ORG")]), ("D", [(0, 1, "ORG")]), ("E", [(0, 1, "ORG")])],
            "EMAIL": [("F", [(0, 1, "EMAIL")])]
        },
        1.5
    ),
])
def test_balance_examples_basic(monkeypatch, docs_annots, groupped_annots, max_ratio):
    monkeypatch.setattr(nlp, "_oversample", lambda x, y: x)
    monkeypatch.setattr(nlp, "_undersample", lambda x, y: x)
    monkeypatch.setattr(nlp, "_group_entity_examples", lambda x: groupped_annots)
    monkeypatch.setattr(nlp, "is_single_entity_annotation", lambda ann: len(set(a[2] for a in ann[1])) == 1)

    result = balance_examples(docs_annots, max_majority_minority_ratio=max_ratio)
    
    assert isinstance(result, list) or hasattr(result, "__iter__")
    flat_result = list(result)
    for ann in docs_annots:
        assert ann in flat_result


def test_balance_examples_calls(monkeypatch):
    called = {}
    def fake_group_entity_examples(x):
        called["group"] = True
        return {"PERSON": [("A", [(0, 1, "PERSON")])]}
    def fake_oversample(x, y):
        called["over"] = True
        return x
    def fake_undersample(x, y):
        called["under"] = True
        return x
    monkeypatch.setattr(nlp, "_group_entity_examples", fake_group_entity_examples)
    monkeypatch.setattr(nlp, "_oversample", fake_oversample)
    monkeypatch.setattr(nlp, "_undersample", fake_undersample)
    monkeypatch.setattr(nlp, "is_single_entity_annotation", lambda ann: True)
    docs_annots = [("A", [(0, 1, "PERSON")])]
    
    balance_examples(docs_annots)
    
    assert "group" in called
    assert "over" in called
    assert "under" in called


def test_balance_examples_empty(monkeypatch):
    monkeypatch.setattr(nlp, "_group_entity_examples", lambda x: {})
    monkeypatch.setattr(nlp, "_oversample", lambda x, y: {})
    monkeypatch.setattr(nlp, "_undersample", lambda x, y: {})
    monkeypatch.setattr(nlp, "is_single_entity_annotation", lambda ann: True)
    docs_annots = []

    result = balance_examples(docs_annots)

    assert list(result) == []


def test_balance_examples_mixed_entities(monkeypatch):
    monkeypatch.setattr(nlp, "_group_entity_examples", lambda x: {"PERSON": [("A", [(0, 1, "PERSON")])], "ORG": [("B", [(0, 1, "ORG")])]})
    monkeypatch.setattr(nlp, "_oversample", lambda x, y: x)
    monkeypatch.setattr(nlp, "_undersample", lambda x, y: x)
    monkeypatch.setattr(nlp, "is_single_entity_annotation", lambda ann: len(set(a[2] for a in ann[1])) == 1)
    docs_annots = [
        ("A", [(0, 1, "PERSON")]),
        ("B", [(0, 1, "ORG")]),
        ("C", [(0, 1, "ORG"), (2, 3, "PERSON")])
    ]
    result = balance_examples(docs_annots)
    
    flat_result = list(result)
    
    assert ("A", [(0, 1, "PERSON")]) in flat_result
    assert ("B", [(0, 1, "ORG")]) in flat_result
    assert ("C", [(0, 1, "ORG"), (2, 3, "PERSON")]) in flat_result


def test_balance_examples_output_type(monkeypatch):
    monkeypatch.setattr(nlp, "_group_entity_examples", lambda x: {"PERSON": [("A", [(0, 1, "PERSON")])]})
    monkeypatch.setattr(nlp, "_oversample", lambda x, y: x)
    monkeypatch.setattr(nlp, "_undersample", lambda x, y: x)
    monkeypatch.setattr(nlp, "is_single_entity_annotation", lambda ann: True)
    docs_annots = [("A", [(0, 1, "PERSON")])]
    
    result = balance_examples(docs_annots)
    
    for item in result:
        assert isinstance(item, tuple)
        assert isinstance(item[0], str)
        assert isinstance(item[1], list)


@pytest.mark.parametrize(
    "span_text,common_acronyms,similarities,threshold,expected",
    [
        ("test1", [], [], 0.95, None),
        ("test1", ["test2", "test3"], [0.5, 0.6], 0.95, None),
        ("test1", ["test2", "test1", "test3"], [0.5, 0.99, 0.6], 0.95, "test1"),
        ("test1", ["test2", "1test", "test1"], [0.5, 0.96, 0.98], 0.95, "test1"),
        ("test1", ["test2", "test1"], [0.5, 0.95], 0.95, None),
        ("test1", ["test2", "test1"], [0.5, 0.951], 0.95, "test1"),
        ("test1", ["test2", "te1st", "tst"], [0.7, 0.8, 0.9], 0.95, None),
    ]
)
def test_check_common_acronyms(monkeypatch, span_text, common_acronyms, similarities, threshold, expected):
    sim_iter = iter(similarities)
    monkeypatch.setattr(
        nlp, "get_spans_similarity",
        lambda acronym, span, encoder: next(sim_iter)
    )
    mock_encoder = object() 
    result = check_common_acronyms(
        span_text, common_acronyms, mock_encoder, similarity_score_threshold=threshold
    )

    assert result == expected


def test_check_common_acronyms_returns_none_when_no_acronyms(monkeypatch):
    mock_encoder = object()
    result = check_common_acronyms("test1", [], mock_encoder)
    assert result is None


def test_check_common_acronyms_returns_none_when_all_below_threshold(monkeypatch):
    monkeypatch.setattr(
        nlp, "get_spans_similarity",
        lambda acronym, span, encoder: 0.5
    )
    mock_encoder = object()
    result = check_common_acronyms("test1", ["test2", "test3"], mock_encoder, similarity_score_threshold=0.95)
    assert result is None


def test_check_common_acronyms_returns_highest_similarity(monkeypatch):
    similarities = iter([0.8, 0.96, 0.99])
    monkeypatch.setattr(
        nlp, "get_spans_similarity",
        lambda acronym, span, encoder: next(similarities)
    )
    mock_encoder = object()
    result = check_common_acronyms("test1", ["test2", "1test", "test1"], mock_encoder, similarity_score_threshold=0.95)

    assert result == "test1"


@pytest.mark.parametrize(
    "texts_to_compare,texts,similarities,threshold,expected",
    [
        # No texts to compare or texts is None
        (None, ["a"], [], 0.95, None),
        (["a"], None, [], 0.95, None),
        # No candidates above threshold
        (["test1"], ["test2", "test3"], [0.5, 0.6], 0.95, None),
        # One string candidate above threshold
        (["test1"], ["test2", "test1", "test3"], [0.5, 0.99, 0.6], 0.95, "test1"),
        # Multiple string candidates above threshold, pick highest
        (["test1"], ["test2", "test3", "test1"], [0.5, 0.96, 0.98], 0.95, "test1"),
        # Similarity exactly at threshold (should not match)
        (["test1"], ["test2", "test1"], [0.5, 0.95], 0.95, None),
        # Similarity just above threshold
        (["test1"], ["test2", "test1"], [0.5, 0.951], 0.95, "test1"),
        # All below threshold
        (["test1"], ["test2", "test3", "test4"], [0.7, 0.8, 0.9], 0.95, None),
        # Tuple candidate, first element higher
        (["test1"], [("test1", "test2"), "test3"], [0.98, 0.5, 0.6], 0.95, ("test1", "test2")),
        # Tuple candidate, second element higher
        (["test1"], [("test2", "test1"), "test3"], [0.5, 0.98, 0.6], 0.95, ("test2", "test1")),
        # Tuple candidate, both below threshold
        (["test1"], [("test2", "test3")], [0.7, 0.8], 0.95, None),
        # Multiple texts_to_compare, pick best
        (["test1", "test3"], ["test2", "test3"], [0.5, 0.6, 0.5, 0.97], 0.95, "test3"),
    ]
)
def test_get_most_similar_text(monkeypatch, texts_to_compare, texts, similarities, threshold, expected):
    sim_iter = iter(similarities)
    def fake_get_spans_similarity(a, b, encoder):
        return next(sim_iter)
    monkeypatch.setattr(
        nlp, "get_spans_similarity",
        fake_get_spans_similarity
    )
    mock_encoder = object()
    result = get_most_similar_text(
        texts_to_compare, texts, mock_encoder, similarity_score_threshold=threshold
    )

    assert result == expected


def test_get_most_similar_text_returns_none_when_no_candidates(monkeypatch):
    mock_encoder = object()
    result = get_most_similar_text(["test1"], [], mock_encoder)

    assert result is None


def test_get_most_similar_text_returns_none_when_no_texts_to_compare(monkeypatch):
    mock_encoder = object()
    result = get_most_similar_text([], ["test1"], mock_encoder)

    assert result is None


def test_get_most_similar_text_tuple_and_string(monkeypatch):
    similarities = iter([0.7, 0.99, 0.8])
    monkeypatch.setattr(
        nlp, "get_spans_similarity",
        lambda a, b, encoder: next(similarities)
    )
    mock_encoder = object()
    result = get_most_similar_text(["test1"], [("test2", "test1"), "test3"], mock_encoder, similarity_score_threshold=0.95)

    assert result == ("test2", "test1")


def test_add_utf8_normalization_changes_make_doc(monkeypatch):
    class DummyNLP:
        def __init__(self):
            self.make_doc_called = False
            self.last_text = None
            self.make_doc = self._make_doc

        def _make_doc(self, text):
            self.make_doc_called = True
            self.last_text = text
            return text

    called = {}
    def fake_normalize_utf8_text(text):
        called["text"] = text
        return "normalized:" + text

    monkeypatch.setattr(nlp, "_normalize_utf8_text", fake_normalize_utf8_text)

    nlp_obj = DummyNLP()
    add_utf8_normalization(nlp_obj)

    result = nlp_obj.make_doc("  Simple\u00a0text  ")
    assert nlp_obj.make_doc_called
    assert nlp_obj.last_text == "normalized:  Simple\u00a0text  "
    assert called["text"] == "  Simple\u00a0text  "
    assert result == "normalized:  Simple\u00a0text  "


def test_add_utf8_normalization_preserves_original_behavior(monkeypatch):
    class DummyNLP:
        def __init__(self):
            self.make_doc = lambda text: f"original:{text}"

    monkeypatch.setattr(nlp, "_normalize_utf8_text", lambda text: text.upper())

    nlp_obj = DummyNLP()
    nlp.add_utf8_normalization(nlp_obj)

    result = nlp_obj.make_doc("abc")
    assert result == "original:ABC"


def test_add_utf8_normalization_multiple_calls(monkeypatch):
    class DummyNLP:
        def __init__(self):
            self.calls = []
            self.make_doc = lambda text: self.calls.append(text) or text

    monkeypatch.setattr(nlp, "_normalize_utf8_text", lambda text: text[::-1])

    nlp_obj = DummyNLP()
    nlp.add_utf8_normalization(nlp_obj)

    texts = ["abc", "def", "ghi"]
    for t in texts:
        out = nlp_obj.make_doc(t)
        assert out == t[::-1]
    assert nlp_obj.calls == [t[::-1] for t in texts]


@pytest.mark.skipif("spacy" not in globals(), reason="spacy not available")
def test_load_nlp_model_calls_spacy_load(monkeypatch):
    called = {}
    class DummyNLP:
        def __init__(self):
            self.pipes = []
            self.added_pipes = []
            self.make_doc = lambda text: text
        def add_pipe(self, name, **kwargs):
            self.added_pipes.append((name, kwargs))
            return self
        def add_patterns(self, patterns):
            self.patterns = patterns

    def fake_spacy_load(model_name):
        called["model_name"] = model_name
        return DummyNLP()

    monkeypatch.setattr("spacy.load", fake_spacy_load)
    monkeypatch.setattr(nlp, "add_utf8_normalization", lambda nlp: called.setdefault("utf8", True))

    nlp_obj = load_nlp_model("dummy_model")

    assert called["model_name"] == "dummy_model"
    assert called["utf8"] is True
    assert isinstance(nlp_obj, DummyNLP)


def test_load_nlp_model_adds_pipes_and_patterns(monkeypatch):
    added = {"pipes": [], "patterns": []}
    class DummyRuler:
        def add_patterns(self, patterns):
            added["patterns"].extend(patterns)
    class DummyNLP:
        def __init__(self):
            self.pipes = []
            self.make_doc = lambda text: text
        def add_pipe(self, name, **kwargs):
            added["pipes"].append((name, kwargs))
            if name == "entity_ruler":
                return DummyRuler()
            return self

    monkeypatch.setattr("spacy.load", lambda model_name: DummyNLP())
    monkeypatch.setattr(nlp, "add_utf8_normalization", lambda nlp: None)

    nlp.load_nlp_model("dummy_model")
    pipe_names = [p[0] for p in added["pipes"]]

    assert "entity_ruler" in pipe_names
    assert "sentencizer" in pipe_names
    for c in ["ipv4", "ipv6", "phone", "email", "ssn", "medicare", "vin", "url"]:
        assert c in pipe_names
    assert any(p[1].get("before") == "ner" for p in added["pipes"] if p[0] == "entity_ruler")
    assert {"label": "DATE", "pattern": [{"TEXT": {"REGEX": r"\d{1,2}/\d{1,2}/\d{4}"}}]} in added["patterns"]


def test_load_nlp_model_returns_nlp_instance(monkeypatch):
    class DummyNLP:
        def __init__(self):
            self.make_doc = lambda text: text
        def add_pipe(self, name, **kwargs):
            return self
        def add_patterns(self, patterns):
            pass

    monkeypatch.setattr("spacy.load", lambda model_name: DummyNLP())
    monkeypatch.setattr(nlp, "add_utf8_normalization", lambda nlp: None)

    nlp_obj = nlp.load_nlp_model()

    assert hasattr(nlp_obj, "make_doc")
    assert callable(nlp_obj.make_doc)


def test_load_nlp_acronyms_model_loads_model_and_adds_sentencizer(monkeypatch):
    called = {}

    class DummyNLP:
        def __init__(self):
            self.pipes = []
            self.sentencizer_added = False
        def add_pipe(self, name, **kwargs):
            self.pipes.append((name, kwargs))
            if name == "sentencizer":
                self.sentencizer_added = True
            return self

    def fake_spacy_load(model_dir):
        called["model_dir"] = model_dir
        return DummyNLP()

    monkeypatch.setattr("spacy.load", fake_spacy_load)

    nlp_obj = load_nlp_acronyms_model()

    assert called["model_dir"] == consts.ACRONYMS_MODEL_DIR
    assert isinstance(nlp_obj, DummyNLP)
    assert nlp_obj.sentencizer_added


def test_load_nlp_acronyms_model_returns_nlp_instance(monkeypatch):
    nlp_mock = MagicMock()
    nlp_mock.add_pipe = MagicMock()

    monkeypatch.setattr("spacy.load", lambda model_dir: nlp_mock)

    nlp_obj = nlp.load_nlp_acronyms_model()

    nlp_mock.add_pipe.assert_called_once_with("sentencizer")
    assert nlp_mock == nlp_obj


def test_load_text_encoder_model(monkeypatch):
    sentence_transformer_mock = MagicMock()
    monkeypatch.setattr(nlp, "SentenceTransformer", sentence_transformer_mock)

    model_name = "dummy_model"
    _ = load_text_encoder_model(model_name)

    sentence_transformer_mock.assert_called_once_with(model_name)



@pytest.mark.parametrize(
    "label,desc,suffix_type,entities_count,expected",
    [
        # Suffix type is str, entities_count = 0
        ("ORG", ("Organization", str), str, 0, '"Organization A"'),
        # Suffix type is str, entities_count = 25 (last letter)
        ("ORG", ("Organization", str), str, 25, '"Organization AZ"'),
        # Suffix type is str, entities_count = 26 (wraps to AA)
        ("ORG", ("Organization", str), str, 26, '"Organization AA"'),
        # Suffix type is int, entities_count = 0
        ("PERSON", ("Person", int), int, 0, '"Person 1"'),
        # Suffix type is int, entities_count = 5
        ("PERSON", ("Person", int), int, 5, '"Person 6"'),
    ]
)
def test_get_ent_replacement_valid(monkeypatch, label, desc, suffix_type, entities_count, expected):
    monkeypatch.setattr(nlp, "consts", type("Dummy", (), {"ENTITY_DESC": {label: desc}}))
    ent = type("Span", (), {"label_": label})()

    result = get_ent_replacement(ent, suffix_type, entities_count)

    assert result == expected


def test_get_ent_replacement_raises_on_unsupported_type(monkeypatch):
    label = "ORG"
    desc = ("Organization", str)
    monkeypatch.setattr(nlp, "consts", type("Dummy", (), {"ENTITY_DESC": {label: desc}}))
    ent = type("Span", (), {"label_": label})()
    
    with pytest.raises(Exception) as excinfo:
        get_ent_replacement(ent, float, 0)

    assert "Unsupported suffix type" in str(excinfo.value)


@pytest.mark.parametrize(
    "source_text,target_text,start_char,end_char,expected_ratio,expected_start,expected_end",
    [
        ("hello", "hello world", 0, 5, 1.0, 0, 5),
        ("world", "hello world", 6, 11, 1.0, 6, 11),
        ("abc", "xyz", 0, 3, 0.0, 0, 3),
        ("abc", "eabcf", 1, 4, 1.0, 1, 4),
        ("abc", "xabcy", 0, 3, 6/7, 0, 4),
        ("abc", "zabc", 1, 4, 1.0, 1, 4),
        ("abc", "zabcz", 1, 4, 1.0, 1, 4),
    ]
)
def test_find_best_matching_string_basic(source_text, target_text, start_char, end_char, expected_ratio, expected_start, expected_end):
    def iterate_fn(s, e):
        return s, e + 1
    
    ratio, s, e = _find_best_matching_string(
        source_text, target_text, start_char, end_char, 0.0, iterate_fn
    )

    print(f"Ratio: {ratio}, Start: {s}, End: {e}")
    print(f"Expected Ratio: {expected_ratio}, Start: {expected_start}, End: {expected_end}")

    assert pytest.approx(ratio, abs=1e-6) == expected_ratio
    assert s == expected_start
    assert e == expected_end


def test_find_best_matching_string_stops_when_ratio_decreases():
    source_text = "abc"
    target_text = "abcxyz"
    def iterate_fn(s, e):
        return s, e + 1
    
    ratio, s, e = _find_best_matching_string(
        source_text, target_text, 0, 3, 0.0, iterate_fn
    )

    assert pytest.approx(ratio, abs=1e-6) == 1.0
    assert s == 0
    assert e == 3


def test_find_best_matching_string_iterate_fn_contracts_window():
    source_text = "abc"
    target_text = "zabc"
    def iterate_fn(s, e):
        return s + 1, e
    
    ratio, s, e = _find_best_matching_string(
        source_text, target_text, 0, 4, 0.0, iterate_fn
    )

    assert pytest.approx(ratio, abs=1e-6) == 1.0
    assert s == 1
    assert e == 4


def test_find_best_matching_string_handles_empty_source():
    source_text = ""
    target_text = "abc"
    def iterate_fn(s, e):
        return s, e + 1
    
    ratio, s, e = _find_best_matching_string(
        source_text, target_text, 0, 0, 0.0, iterate_fn
    )

    assert ratio == 0.0
    assert s == 0
    assert e == 0


def test_find_best_matching_string_handles_empty_target():
    source_text = "abc"
    target_text = ""
    def iterate_fn(s, e):
        return s, e + 1
    
    ratio, s, e = _find_best_matching_string(
        source_text, target_text, 0, 0, 0.0, iterate_fn
    )

    assert ratio == 0.0
    assert s == 0
    assert e == 0


@pytest.mark.parametrize("source_text,target_text,start_char,end_char,expected_exception_message", [
    ("abc", "abc", -1, 3, r"start_char [-]?\d+ and end_char \d+ must be within the bounds of target_text length [-]?\d+.*"),
    ("abc", "abc", 0, 4, r"start_char [-]?\d+ and end_char [-]?\d+ must be within the bounds of target_text length \d+.*"),
    ("abc", "abc", 0, -1, r"start_char [-]?\d+ must not be greater than end_char [-]?\d+.*")
])
def test_find_best_matching_string_when_start_or_end_char_are_out_of_bounds(
    source_text, target_text, start_char, end_char, expected_exception_message):
    with pytest.raises(ValueError, match=expected_exception_message):
        _find_best_matching_string(source_text, target_text, start_char, end_char, 0.0, None)


def test_find_best_matching_string_iterate_fn_breaks_loop():
    source_text = "abc"
    target_text = "abc"
    def iterate_fn(s, e):
        return e, e
    
    ratio, s, e = _find_best_matching_string(
        source_text, target_text, 0, 3, 0.0, iterate_fn
    )

    assert pytest.approx(ratio, abs=1e-6) == 1.0
    assert s == 0
    assert e == 3


def test_find_best_matching_string_best_match_not_first_window():
    source_text = "abc"
    target_text = "xxabc"
    def iterate_fn(s, e):
        return s + 1, e + 1
    
    ratio, s, e = _find_best_matching_string(
        source_text, target_text, 0, 3, 0.0, iterate_fn
    )

    assert pytest.approx(ratio, abs=1e-6) == 1.0
    assert target_text[s:e] == "abc"


@pytest.mark.skip("Not implemented yet")
def test_find_entity_boundaries():
    ...
