from app import sparse


def test_tokenize_has_unigrams_and_bigrams():
    toks = sparse.tokenize("Mức lương tối thiểu")
    assert "mức" in toks and "mức lương" in toks and "lương tối" in toks


def test_nfc_normalisation_makes_composed_and_decomposed_equal():
    assert sparse.tokenize("Điều") == sparse.tokenize("Điều")


def test_document_vector_is_sorted_and_positive():
    idx, val = sparse.doc_vector("Mức lương tối thiểu là mức lương thấp nhất")
    assert idx == sorted(idx) and len(idx) == len(val) and all(v > 0 for v in val)


def test_query_terms_overlap_document_terms():
    d_idx, _ = sparse.doc_vector("Mức lương tối thiểu vùng I")
    q_idx, q_val = sparse.query_vector("mức lương")
    assert set(q_idx) <= set(d_idx) and set(q_val) == {1.0}


def test_hash_ids_are_stable_across_runs():
    # crc32, not Python's randomised hash(): query and document vectors must agree between processes
    assert sparse._index("lương") == 538637
