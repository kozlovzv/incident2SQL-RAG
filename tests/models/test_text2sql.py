import pytest
from src.models.text2sql import Text2SQLGenerator


@pytest.fixture
def text2sql():
    return Text2SQLGenerator()


def test_sql_generation():
    generator = Text2SQLGenerator()
    query = "Show all high severity incidents from last month"
    result = generator.generate_sql(query)
    assert isinstance(result, str)
    assert "SELECT" in result.upper()


def test_sql_injection_prevention():
    generator = Text2SQLGenerator()
    dangerous_query = "Show incidents; DROP TABLE incidents;"
    with pytest.raises(ValueError):
        generator.generate_sql(dangerous_query)


def test_context_enhanced_generation():
    generator = Text2SQLGenerator()
    query = "Show similar incidents"
    context = [{"content": "Network outage incident from last week"}]
    result = generator.generate_sql(query, context)
    assert isinstance(result, str)
    assert "SELECT" in result.upper()
