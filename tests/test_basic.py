import os

import pytest

from chemrag.data import normalize_cas
from chemrag.orchestrator import Orchestrator


@pytest.fixture(scope="module")
def bot():
    os.environ.pop("LLM_API_KEY", None)  # offline, no LLM calls
    return Orchestrator()


def test_normalize_cas():
    assert normalize_cas("CAS #79-81-2") == "79-81-2"
    assert normalize_cas("0") is None


def test_cas_question(bot):
    r = bot.answer("Which products contain CAS 75-07-0?")
    assert r["details"]["totals"]["products"] == 30 and r["evidence"]
    assert "Planner: route = sql" in r["query_plan"]


def test_typo_fixed(bot):
    r = bot.answer("Which products contain titanium dioxde?")
    assert any("Titanium dioxide" in w for w in r["warnings"])


def test_unknown_name_runs_no_query(bot):
    assert bot.answer("Which products contain acetone?")["sql"] == []


def test_year_outside_data(bot):
    r = bot.answer("Show products discontinued in 2024 that had ChemicalName = Titanium dioxide")
    assert any("June 2020" in w for w in r["warnings"])


def test_compare(bot):
    assert len(bot.answer("Compare titanium dioxide and talc")["details"]["sides"]) == 2
