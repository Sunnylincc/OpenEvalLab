from openevallab.metrics import contains_answer, exact_match, llm_judge_placeholder


def test_exact_match_normalizes_whitespace_and_case():
    assert exact_match("  Paris\n", "paris").passed


def test_contains_answer_accepts_longer_prediction():
    assert contains_answer("The answer is red blood cells.", "red blood cells").score == 1.0


def test_llm_judge_placeholder_is_explicitly_disabled():
    result = llm_judge_placeholder("a", "b")
    assert not result.passed
    assert "not executed" in result.explanation
