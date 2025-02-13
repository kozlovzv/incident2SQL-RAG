import pytest
from src.models.text2sql import Text2SQLGenerator


@pytest.fixture
def text2sql():
    return Text2SQLGenerator()


def test_limit_queries(text2sql):
    # Test default limit
    sql = text2sql.generate_sql("Покажи последние инциденты")
    assert "LIMIT 10" in sql

    # Test custom limit
    sql = text2sql.generate_sql("Покажи 3 инцидента с высоким риском")
    assert "LIMIT 3" in sql


def test_complex_conditions(text2sql):
    # Test risk + category + time period
    sql = text2sql.generate_sql(
        "Покажи инциденты категории server crash за последний месяц с риском выше 0.8"
    )
    assert "c.name = 'Server Crash'" in sql
    assert "i.risk_level >= 0.8" in sql
    assert "date('now', '-1 month')" in sql

    # Test quarter + status
    sql = text2sql.generate_sql(
        "Отобрази все инциденты за Q2 2024 со статусом resolved"
    )
    assert "date_occurred >= '2024-04-01'" in sql
    assert "date_occurred < '2024-07-01'" in sql
    assert "i.status = 'resolved'" in sql


def test_sorting_queries(text2sql):
    # Test risk sorting
    sql = text2sql.generate_sql("Покажи инциденты, сортируй по риску по убыванию")
    assert "ORDER BY i.risk_level DESC" in sql

    # Test date sorting
    sql = text2sql.generate_sql("Покажи старейшие инциденты")
    assert "ORDER BY" in sql
    assert "date_occurred ASC" in sql

    # Test combined sorting
    sql = text2sql.generate_sql(
        "Покажи инциденты с риском 0.9, сортированные по дате, newest first"
    )
    assert "i.risk_level = 0.9" in sql
    assert "ORDER BY" in sql
    assert "date_occurred DESC" in sql


def test_russian_language_support(text2sql):
    # Test Russian commands
    sql = text2sql.generate_sql("Отобрази 5 последних инцидентов")
    assert "LIMIT 5" in sql

    sql = text2sql.generate_sql("Сортируй по риску по убыванию")
    assert "ORDER BY i.risk_level DESC" in sql

    sql = text2sql.generate_sql("Покажи инциденты с риском выше 0.7")
    assert "i.risk_level >= 0.7" in sql
