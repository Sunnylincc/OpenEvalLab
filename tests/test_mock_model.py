from openevallab.models import MockModelClient


def test_mock_model_uses_prompt_mapping():
    model = MockModelClient({"hello": "world"})
    assert model.generate("hello") == "world"


def test_mock_model_heuristic_answers_sample_reasoning():
    model = MockModelClient()
    assert model.generate("Mira has 12 apples, gives away 5, then buys 4 more. How many apples does she have?") == "11"
