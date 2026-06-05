from openevallab.metrics import contains_answer, exact_match, heuristic_score, normalized_exact_match


def test_exact_match_is_strict_after_strip():
    assert exact_match("Paris", "Paris").passed
    assert not exact_match("paris", "Paris").passed


def test_normalized_exact_match_normalizes_case_punctuation_and_articles():
    assert normalized_exact_match(" The Paris! ", "paris").passed


def test_contains_answer_accepts_longer_prediction():
    assert contains_answer("The answer is red blood cells.", "red blood cells").score == 1.0


def test_heuristic_score_uses_token_overlap():
    result = heuristic_score("red cells", "red blood cells")
    assert 0.0 < result.score < 1.0
