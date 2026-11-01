"""CLI:  python main.py "Which products contain CAS 75-07-0?"   (add --json for raw output)"""

import json
import sys

from chemrag.orchestrator import Orchestrator


def show(r):
    print("\nQ:", r["question"])
    print("\nANSWER:", r["answer"])
    print("CONFIDENCE:", r["confidence"])
    for w in r["warnings"]:
        print("WARNING:", w)
    for t in r["details"].get("trend", []):
        print(f"  {t['year']}: {t['products_reported']} reported, {t['later_discontinued']} later discontinued")
    if r["evidence"]:
        print("\nEVIDENCE (CDPHId | CSFId | ChemicalId | product | company | chemical | CAS)")
        for e in r["evidence"]:
            print(f"  {e['CDPHId']} | {e['CSFId']} | {e['ChemicalId']} | {e['ProductName'][:35]} | "
                  f"{e['CompanyName'][:25]} | {e['ChemicalName'][:30]} | {e['CasNumber']}")
    print("\nQUERY PLAN")
    for step in r["query_plan"]:
        print("  -", step)


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    args = [a for a in sys.argv[1:] if a != "--json"]
    bot = Orchestrator()
    questions = args or iter(lambda: input("\n> "), "")
    for q in questions:
        r = bot.answer(q)
        print(json.dumps(r, indent=2, default=str)) if "--json" in sys.argv else show(r)
