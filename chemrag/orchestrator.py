"""Orchestrator: planner -> extraction -> semantic -> SQL -> synthesizer."""

from .agents import SemanticAgent, SQLAgent, choose_route, classify_intent, decompose, extract, synthesize
from .data import get_connection, normalize_cas

DATA_END_YEAR = 2020  # dataset ends June 2020


class Orchestrator:
    def __init__(self, con=None):
        self.con = con or get_connection()
        self.semantic = SemanticAgent(self.con)

    def answer(self, question):
        intent = classify_intent(question)
        parts = decompose(question, intent)
        if intent != "compare" and len(parts) > 1:  # multi-part: answer each part, then merge
            subs = [self.answer(p) for p in parts]
            merged = {k: [x for s in subs for x in s[k]] for k in ["evidence", "query_plan", "sql", "warnings"]}
            merged["query_plan"].insert(0, f"Planner: split into {len(parts)} sub-questions")
            return {"question": question, "answer": "\n".join(s["answer"] for s in subs), **merged,
                    "confidence": min((s["confidence"] for s in subs), key=["low", "medium", "high"].index),
                    "details": {}}

        warnings, plan, sql = [], [f"Planner: intent = {intent}"], SQLAgent(self.con)
        ex = extract(question)
        c = ex["constraints"]
        plan.append(f"Extraction: entities={ex['entities']} cas={ex['cas']} constraints={c}")
        if c.get("year_from", 0) > DATA_END_YEAR:
            warnings.append("Dataset only goes up to June 2020 - no data for that year.")
        if intent == "compare":
            return self._compare(question, parts, c, plan, sql, warnings)

        route = choose_route(ex)
        plan.append(f"Planner: route = {route}")
        filters, not_found = self._resolve(ex, plan, warnings) if route != "none" else ({}, [])
        # safety: never run an unfiltered query when the user's entity wasn't found
        if not filters and (not_found or intent not in ("data_quality", "trend", "summarize")):
            msg = (f"I couldn't find {', '.join(not_found)} in the data." if not_found
                   else "Please mention a chemical, CAS, product, brand, company or category.")
            return self._result(question, msg, [], plan, sql, warnings, "low")

        w, p = sql.where(filters, c)
        plan.append(f"SQL: WHERE {w}  params={p}")
        facts = {"totals": sql.totals(w, p)}
        if intent == "trend":
            facts["trend"] = sql.trend(w, p)
        elif intent in ("list_chemicals", "lookup"):
            facts["chemicals"] = sql.chemicals(w, p)
        elif intent == "summarize":
            facts["top_chemicals"] = sql.top("ChemicalName", w, p, 5)
        elif intent == "data_quality":
            facts["quality"] = sql.data_quality(w, p)
            facts["duplicates_removed"] = self.con.execute(
                "SELECT (SELECT COUNT(*) FROM raw) - (SELECT COUNT(*) FROM clean)").fetchone()[0]
        else:
            facts["companies"] = sql.top("CompanyName", w, p)

        if n := facts["totals"]["date_conflicts"]:
            warnings.append(f"{n} records have DiscontinuedDate before InitialDateReported (conflicting data).")
        if filters.get("brand"):
            for b in sql.brand_companies(w, p):
                warnings.append(f"Brand {b['brand']} is reported by several companies: {b['companies']}")

        evidence = sql.evidence(w, p)
        scope = ", ".join(f"{k}={v}" for k, v in filters.items()) or "whole dataset"
        text, mode = synthesize(question, intent, scope, facts, evidence, warnings)
        plan.append(f"Synthesizer: {mode}")
        return self._result(question, text, evidence, plan, sql, warnings,
                            "medium" if warnings else "high", facts)

    def _resolve(self, ex, plan, warnings):
        """Semantic agent: map phrases to real values; CAS and IDs pass through."""
        filters, not_found = {}, []
        for etype, phrases in ex["entities"].items():
            for phrase in phrases:
                r = self.semantic.resolve(etype, phrase)
                plan.append(f"Semantic: {etype} '{phrase}' -> {r['status']} {r['values'][:3]}")
                if r["status"] == "not_found":
                    not_found.append(phrase)
                    warnings.append(f"'{phrase}' not found. Closest: {r['suggestions']}")
                    continue
                if r["status"] == "fuzzy":
                    warnings.append(f"Assumed '{phrase}' means '{r['values'][0]}' (score {r['score']})")
                if r["status"] == "partial" and len(r["values"]) > 1:
                    warnings.append(f"'{phrase}' matched several {etype} names: {r['values'][:5]}")
                filters.setdefault(etype, []).extend(r["values"])
        if filters.get("chemical"):  # synonyms via CAS
            filters["cas"] = self.semantic.cas_for(filters["chemical"])
        for raw in ex["cas"]:
            cas = normalize_cas(raw)
            names = self.semantic.names_for_cas(cas) if cas else []
            if not names:
                not_found.append(raw)
                warnings.append(f"CAS {raw} is invalid or not in the dataset")
                continue
            filters.setdefault("cas", []).append(cas)
            plan.append(f"Semantic: CAS {cas} = {names}")
        filters.update(ex["ids"])
        return filters, not_found

    def _compare(self, question, parts, c, plan, sql, warnings):
        plan.append(f"Planner: compare -> sub-tasks {parts}")
        sides, evidence = [], []
        for part in parts:
            ex = extract(part)
            if not (ex["entities"] or ex["cas"] or ex["ids"]) and (match := self.semantic.resolve_any(part)):
                ex["entities"] = {match[0]: [part]}
            filters, _ = self._resolve(ex, plan, warnings)
            if not filters:
                warnings.append(f"Could not resolve '{part}' for comparison")
                continue
            w, p = sql.where(filters, c)
            sides.append({"label": ", ".join(str(v[0]) for v in filters.values()), "totals": sql.totals(w, p)})
            evidence += sql.evidence(w, p, 5)
        if len(sides) < 2:
            return self._result(question, "Need two items I can find in the data to compare.",
                                evidence, plan, sql, warnings, "low")
        facts = {"sides": sides}
        text, mode = synthesize(question, "compare", "comparison", facts, evidence, warnings)
        plan.append(f"Synthesizer: {mode}")
        return self._result(question, text, evidence, plan, sql, warnings, "medium" if warnings else "high", facts)

    @staticmethod
    def _result(question, answer, evidence, plan, sql, warnings, confidence, facts=None):
        return {"question": question, "answer": answer, "evidence": evidence, "query_plan": plan,
                "sql": sql.log, "warnings": warnings, "confidence": confidence, "details": facts or {}}
