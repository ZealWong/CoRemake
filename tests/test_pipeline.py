"""Test data pipeline components."""
from coremake.data.pair_mining import build_positive_pairs, build_negative_pairs
from coremake.data.relation_labeler import _match_relation
from coremake.data.anchor_builder import build_anchor_dataset


def test_positive_pairs():
    papers = [{"paper_id": "a"}, {"paper_id": "b"}]
    citations = [{"src_paper_id": "a", "dst_paper_id": "b"}]
    pairs = build_positive_pairs(papers, citations)
    assert len(pairs) == 1
    assert pairs[0]["label"] == 1


def test_negative_pairs():
    papers = [{"paper_id": str(i)} for i in range(10)]
    pos_set = {("0", "1")}
    pairs = build_negative_pairs(papers, pos_set, num_negatives_per_paper=2)
    assert len(pairs) > 0
    assert all(p["label"] == 0 for p in pairs)


def test_relation_pattern():
    assert _match_relation("This work extends the previous approach") == "extends"
    assert _match_relation("We confirm the findings of earlier work") == "strengthens"
    assert _match_relation("Nothing special here") is None


def test_anchor_builder():
    papers = [
        {"paper_id": "old", "year": 2005},
        {"paper_id": "new1", "year": 2020},
        {"paper_id": "new2", "year": 2021},
    ]
    citations = [
        {"src_paper_id": "new1", "dst_paper_id": "old"},
        {"src_paper_id": "new2", "dst_paper_id": "old"},
    ]
    records = build_anchor_dataset(papers, citations, legacy_year_threshold=2010)
    assert len(records) == 1
    assert records[0]["legacy_paper_id"] == "old"
    assert "new1" in records[0]["positive_ids"]
