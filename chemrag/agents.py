"""The 5 agents: planner, extraction, semantic retrieval, SQL, synthesizer."""

import json
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- planner ----------
INTENTS = [("data_quality", r"data quality|missing|invalid|duplicate|conflict"),
           ("compare", r"\bcompare\b|\bvs\b|versus|difference between"),
           ("trend", r"trend|over time|by year|per year"),
           ("summarize", r"summari[sz]e|summary|overview"),
           ("lookup", r"what is the cas|cas (number|no) (of|for)"),
           ("list_chemicals", r"\bchemicals?\b.*\b(reported|for|in)\b")]


def classify_intent(q):
    return next((name for name, pat in INTENTS if re.search(pat, q.lower())), "list")


def decompose(q, intent):
    if intent == "compare":  # "compare A and B" -> [A, B]
        body = re.sub(r"(?i)^.*?\b(compare|difference between)\b\s*", "", q).rstrip("?. ")
        return [p for p in re.split(r"(?i)\s+(?:and|vs\.?|versus|with)\s+", body) if p]
    parts = [p.strip() for p in re.split(r"\?\s+|;\s*|\.\s+(?=[A-Z])", q) if p.strip()]
    return parts if len(parts) > 1 else [q]


def choose_route(ex):
    """names -> semantic+sql, only CAS / IDs -> sql, nothing -> none"""
    return "semantic+sql" if ex["entities"] else "sql" if ex["cas"] or ex["ids"] else "none"


# ---------- extraction ----------
CAS_RE = r"\b\d{2,7}-\d{2}-\d\b"
STOP = (r"(?=\s+(?:in|for|by|from|or|that|which|with|between|after|before|since|made|"
        r"discontinued|reported|removed|still|had|contain|containing)\b|\s+and\s+(?-i:[a-z])|[,?]|$)")
CUES = {"chemical": r"\b(?:contain(?:s|ing)?|chemical(?:name)?\s*=?|had(?:\s+chemicalname\s*=)?|"
                    r"cas (?:number|no) (?:of|for))\s+",
        "brand": r"\bbrand\s+", "company": r"\b(?:company|made by)\s+",
        "subcategory": r"\bsub-?category\s+", "category": r"\b(?<!sub-)category\s+",
        "product": r"\bproduct\s+(?:named|called)\s+"}


def extract(q):
    entities = {}
    for etype, cue in CUES.items():
        for m in re.finditer(cue + r"(.+?)" + STOP, q, re.IGNORECASE):
            phrase = m.group(1).strip(" '\"")
            if phrase and not re.search(CAS_RE, phrase) and not phrase.lower().startswith("cas"):
                entities.setdefault(etype, []).append(phrase)
    c, ql = {}, q.lower()
    if m := re.search(r"between\s+(\d{4})\s+and\s+(\d{4})", ql):
        c["year_from"], c["year_to"] = int(m[1]), int(m[2])
    elif m := re.search(r"\b(?:in|during)\s+(\d{4})\b", ql):
        c["year_from"] = c["year_to"] = int(m[1])
    else:
        if m := re.search(r"\b(?:after|since)\s+(\d{4})\b", ql):
            c["year_from"] = int(m[1])
        if m := re.search(r"\bbefore\s+(\d{4})\b", ql):
            c["year_to"] = int(m[1]) - 1
    if "discontinu" in ql:
        c["status"] = "discontinued"
    elif re.search(r"\bstill\b|\bactive\b", ql):
        c["status"] = "active"
    if re.search(r"\bremoved\b|reformulat", ql):
        c["removed"] = True
    ids = {col: [int(x) for x in re.findall(rf"\b{col}\s*[:=#]?\s*(\d+)", q, re.I)]
           for col in ["CDPHId", "ChemicalId"]}
    return {"entities": entities, "cas": re.findall(CAS_RE, q),
            "ids": {k: v for k, v in ids.items() if v}, "constraints": c}


# ---------- semantic retrieval ----------
COLUMNS = {"chemical": "ChemicalName", "company": "CompanyName", "brand": "BrandName",
           "category": "PrimaryCategory", "subcategory": "SubCategory", "product": "ProductName"}


class SemanticAgent:
    """tf-idf char n-gram vectors per entity type: exact -> partial -> fuzzy."""

    def __init__(self, con):
        self.con, self.index = con, {}
        for etype, col in COLUMNS.items():
            names = [r[0] for r in con.execute(
                f"SELECT DISTINCT TRIM({col}) FROM clean WHERE {col} IS NOT NULL ORDER BY 1").fetchall()]
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
            self.index[etype] = (names, vec, vec.fit_transform(names))

    def resolve(self, etype, phrase):
        names, vec, matrix = self.index[etype]
        if exact := [n for n in names if n.lower() == phrase.lower()]:
            return {"status": "exact", "values": exact, "score": 1.0}
        p = re.escape(phrase.lower().replace("'", ""))
        if partial := [n for n in names if re.search(rf"\b{p}s?\b", n.lower().replace("'", ""))]:
            return {"status": "partial", "values": partial[:10], "score": 0.9}
        scores = cosine_similarity(vec.transform([phrase]), matrix)[0]
        top = [(names[i], round(float(scores[i]), 2)) for i in scores.argsort()[::-1][:3]]
        if top[0][1] >= 0.75:
            return {"status": "fuzzy", "values": [top[0][0]], "score": top[0][1]}
        return {"status": "not_found", "values": [], "score": top[0][1], "suggestions": top}

    def resolve_any(self, phrase):
        """Bare name (compare sides): best match across entity types."""
        hits = [(t, self.resolve(t, phrase)) for t in ["chemical", "company", "brand", "subcategory", "category"]]
        hits = [h for h in hits if h[1]["status"] != "not_found"]
        return max(hits, key=lambda h: h[1]["score"]) if hits else None

    def cas_for(self, names):
        sql = f"SELECT DISTINCT cas FROM name_cas WHERE ChemicalName IN ({','.join('?' * len(names))})"
        return [r[0] for r in self.con.execute(sql, names).fetchall() if r[0]]

    def names_for_cas(self, cas):
        sql = "SELECT DISTINCT TRIM(ChemicalName) FROM clean WHERE cas_clean = ?"
        return [r[0] for r in self.con.execute(sql, [cas]).fetchall()]


# ---------- structured query ----------
def _in(expr, vals):
    return f"{expr} IN ({','.join('?' * len(vals))})"


class SQLAgent:
    def __init__(self, con):
        self.con, self.log = con, []

    def run(self, sql, params):
        self.log.append({"sql": " ".join(sql.split()), "params": params})
        cur = self.con.execute(sql, params)
        return [dict(zip([d[0] for d in cur.description], r)) for r in cur.fetchall()]

    @staticmethod
    def where(f, c):
        clauses, params = [], []
        names, cas = f.get("chemical", []), f.get("cas", [])
        if names or cas:
            parts = ([_in("TRIM(ChemicalName)", names)] if names else []) + ([_in("cas_clean", cas)] if cas else [])
            clauses.append("(" + " OR ".join(parts) + ")")
            params += names + cas
        for key in ["CDPHId", "ChemicalId", "company", "brand", "category", "subcategory", "product"]:
            if f.get(key):
                clauses.append(_in(key if key in ("CDPHId", "ChemicalId") else f"TRIM({COLUMNS[key]})", f[key]))
                params += f[key]
        date_col = "InitialDateReported"
        if c.get("status") == "discontinued":
            clauses.append("DiscontinuedDate IS NOT NULL")
            date_col = "DiscontinuedDate"
        if c.get("status") == "active":
            clauses.append("DiscontinuedDate IS NULL AND ChemicalDateRemoved IS NULL")
        if c.get("removed"):
            clauses.append("ChemicalDateRemoved IS NOT NULL")
            date_col = "ChemicalDateRemoved"
        for key, op in [("year_from", ">="), ("year_to", "<=")]:
            if key in c:
                clauses.append(f"YEAR({date_col}) {op} ?")
                params.append(c[key])
        return " AND ".join(clauses) or "TRUE", params

    def totals(self, w, p):
        return self.run(f"""SELECT COUNT(DISTINCT CDPHId) AS products, COUNT(*) AS records,
            COUNT(DISTINCT CompanyName) AS companies, SUM(CASE WHEN date_conflict THEN 1 ELSE 0 END)
            AS date_conflicts FROM clean WHERE {w}""", p)[0]

    def evidence(self, w, p, limit=10):
        return self.run(f"""SELECT CDPHId, MIN(CSFId) AS CSFId, ChemicalId,
            MAX(TRIM(ProductName)) AS ProductName, MAX(TRIM(CompanyName)) AS CompanyName,
            MAX(TRIM(ChemicalName)) AS ChemicalName, MAX(cas_clean) AS CasNumber
            FROM clean WHERE {w} GROUP BY CDPHId, ChemicalId ORDER BY CDPHId DESC, ChemicalId LIMIT {limit}""", p)

    def top(self, col, w, p, n=3):
        return self.run(f"""SELECT TRIM({col}) AS name, COUNT(DISTINCT CDPHId) AS products
            FROM clean WHERE {w} GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT {n}""", p)

    def chemicals(self, w, p):
        return self.run(f"""SELECT TRIM(ChemicalName) AS name, MAX(cas_clean) AS cas,
            COUNT(DISTINCT CDPHId) AS products FROM clean WHERE {w} GROUP BY 1 ORDER BY 3 DESC, 1 LIMIT 10""", p)

    def trend(self, w, p):
        return self.run(f"""SELECT YEAR(InitialDateReported) AS year, COUNT(DISTINCT CDPHId) AS products_reported,
            COUNT(DISTINCT CASE WHEN DiscontinuedDate IS NOT NULL THEN CDPHId END) AS later_discontinued
            FROM clean WHERE {w} GROUP BY 1 ORDER BY 1""", p)

    def data_quality(self, w, p):
        return self.run(f"""SELECT COUNT(*) AS records,
            SUM(CASE WHEN CasNumber IS NULL THEN 1 ELSE 0 END) AS cas_missing_raw,
            SUM(CASE WHEN cas_clean IS NULL THEN 1 ELSE 0 END) AS cas_missing_after_cleaning,
            SUM(CASE WHEN BrandName IS NULL THEN 1 ELSE 0 END) AS brand_missing,
            SUM(CASE WHEN date_conflict THEN 1 ELSE 0 END) AS date_conflicts FROM clean WHERE {w}""", p)[0]

    def brand_companies(self, w, p):
        return self.run(f"""SELECT TRIM(BrandName) AS brand, LIST(DISTINCT TRIM(CompanyName) ORDER BY TRIM(CompanyName)) AS companies
            FROM clean WHERE {w} GROUP BY 1 HAVING COUNT(DISTINCT TRIM(CompanyName)) > 1""", p)


# ---------- synthesizer ----------
def _fmt(rows):
    return ", ".join(f"{r['name']} ({r['products']})" for r in rows)


def template_answer(intent, scope, f):
    """Answer built only from SQL results (no LLM)."""
    if intent == "compare":
        return "Comparison: " + " | ".join(f"{s['label']}: {s['totals']['products']} products, "
                                           f"{s['totals']['companies']} companies" for s in f["sides"])
    t = f["totals"]
    if not t["products"]:
        return f"No products found for {scope}."
    if intent == "data_quality":
        q = f["quality"]
        return (f"{scope}: {q['records']} records. CAS missing in {q['cas_missing_raw']} "
                f"({q['cas_missing_after_cleaning']} after cleaning), brand missing in {q['brand_missing']}, "
                f"{q['date_conflicts']} with discontinued date before report date. "
                f"{f['duplicates_removed']} duplicate rows removed at load.")
    if intent in ("list_chemicals", "lookup"):
        return f"Chemicals for {scope}: " + ", ".join(
            f"{r['name']} (CAS {r['cas']}, {r['products']} products)" for r in f["chemicals"]) + "."
    if intent == "trend":
        rows = f["trend"]
        peak = max(rows, key=lambda r: r["products_reported"])
        return (f"{scope}: products reported from {rows[0]['year']} to {rows[-1]['year']}, "
                f"peak in {peak['year']} ({peak['products_reported']} products).")
    if intent == "summarize":
        return f"{scope}: {t['products']} products from {t['companies']} companies. Top chemicals: {_fmt(f['top_chemicals'])}."
    return (f"{t['products']} products ({t['records']} records) match {scope}, from "
            f"{t['companies']} companies. Top companies: {_fmt(f['companies'])}.")


PROMPT = """You answer questions about a cosmetics chemical dataset.
Use ONLY the facts and evidence rows given. Never add numbers, names or claims that are not there.
Format: one short answer sentence, then 2-4 bullet points of details.
When you mention a specific product, cite it as [CDPHId 12345] using IDs from the evidence."""


def synthesize(question, intent, scope, facts, evidence, warnings):
    """LLM answer grounded in SQL rows if LLM_API_KEY is set, else template. Returns (text, mode)."""
    draft = template_answer(intent, scope, facts)
    if not os.getenv("LLM_API_KEY"):
        return draft, "template"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL") or None)
        ctx = json.dumps({"question": question, "draft_answer": draft, "facts": facts,
                          "evidence": evidence, "warnings": warnings}, default=str)
        text = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0,
            messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": ctx}],
        ).choices[0].message.content
        # guard: every cited ID must be in the evidence
        if not set(re.findall(r"CDPHId\s*(\d+)", text)) <= {str(e["CDPHId"]) for e in evidence}:
            return draft, "template (llm cited unknown ids)"
        return text, "llm"
    except Exception as exc:
        return draft, f"template (llm failed: {type(exc).__name__})"
