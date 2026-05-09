"""
Grants vs. master's-thesis supervision in Finnish Computer Science.

Pipeline:
  1. Pull every research.fi funding decision tagged
     fieldsOfScience.nameEnScience = "Computer and information sciences".
     Source: undocumented but public Elasticsearch behind research.fi at
     https://researchfi-api-production.2.rahtiapp.fi/portalapi/funding/_search
  2. Aggregate per researcher: sum `shareOfFundingInEur`, deduplicating
     consortium-replicated rows by (consortiumProject, lastName, share).
     Research Council of Finland consortia are stored as one document per
     partner organization with the full participant list repeated; naive
     sums overcount by the number of partners.
  3. Take the top-N by total euros.
  4. For each, query the Finna API
     (https://api.finna.fi/api/v1/search) for master's theses
     (format:1/Thesis/Masters/) where the researcher appears in
     nonPresenterAuthors with a supervision role. Roles accepted: MARC
     codes `dgs` (degree-granting supervisor) and `ths` (thesis advisor),
     plus textual `ohjaaja`, `valvoja`, `supervisor`, `advisor`.
  5. Render an anonymized scatter (M€ vs. master's count).

Run:
    python grants_vs_theses.py                # full pipeline
    python grants_vs_theses.py --top 60       # bigger ranking
    python grants_vs_theses.py --plot         # re-render plot from cache

CSV outputs land next to the script. Delete to force a refetch.

Caveats:
  - research.fi `fieldsOfScience` tagging is sparse. Many CS PIs are
    missing because none of their grants got the CS tag.
  - The CS-FOS tag is also leaky — a few digital-humanities grants land in
    the list. Inspect the CSV before quoting names.
  - Funding coverage on research.fi is solid from ~2014 only. Earlier
    grants are essentially absent.
  - Finna name matching depends on the same first-name spelling appearing
    in both research.fi and the library catalog. The script feeds every
    observed first-name variant from research.fi into Finna and dedupes
    Finna records by id.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from urllib import parse, request

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
RFI_URL = ("https://researchfi-api-production.2.rahtiapp.fi/portalapi/"
           "funding/_search?request_cache=true")
FINNA_URL = "https://api.finna.fi/api/v1/search"
HEADERS = {"User-Agent": "grants-thesis-study/1.0", "Accept": "application/json"}
PAGE_RFI = 200
PAGE_FINNA = 100
CS_FOS = "Computer and information sciences"
SUPERVISOR_ROLES = {
    "dgs", "ths", "thr",
    "ohjaaja", "valvoja", "supervisor", "advisor", "thesis advisor",
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _get(url: str) -> dict:
    req = request.Request(url, headers=HEADERS)
    with request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _post(body: dict) -> dict:
    req = request.Request(
        RFI_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={**HEADERS, "Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


# ---------------------------------------------------------------------------
# Step 1+2: aggregate CS funding per researcher
# ---------------------------------------------------------------------------
def aggregate_cs_funding() -> list[dict]:
    query = {"nested": {
        "path": "fieldsOfScience",
        "query": {"term": {"fieldsOfScience.nameEnScience.keyword": CS_FOS}},
    }}
    by_key: dict[tuple[str, str], dict] = {}
    seen_consortium: set[tuple] = set()
    after = None
    fetched = 0
    while True:
        body = {
            "size": PAGE_RFI,
            "query": query,
            "sort": [{"projectId": "asc"}],
            "_source": ["projectId", "fundingStartYear", "fundingGroupPerson"],
        }
        if after is not None:
            body["search_after"] = [after]
        data = _post(body)
        hits = data["hits"]["hits"]
        if not hits:
            break
        for h in hits:
            src = h["_source"]
            for p in src.get("fundingGroupPerson") or []:
                share = float(p.get("shareOfFundingInEur") or 0)
                if share <= 0:
                    continue
                first = (p.get("fundingGroupPersonFirstNames") or "").strip()
                last = (p.get("fundingGroupPersonLastName") or "").strip()
                if not last:
                    continue
                cp = (p.get("consortiumProject") or "").strip()
                key_dedup = ((cp, last.lower(), share) if cp
                             else ("pid:" + str(src["projectId"]), last.lower(), share))
                if key_dedup in seen_consortium:
                    continue
                seen_consortium.add(key_dedup)
                first_tok = first.split()[0] if first else ""
                bk = (first_tok.lower(), last.lower())
                rec = by_key.setdefault(bk, {
                    "first_name": first_tok, "last_name": last,
                    "total_eur": 0.0, "n_decisions": 0, "n_as_leader": 0,
                    "years": set(), "orcids": set(), "first_variants": set(),
                })
                rec["total_eur"] += share
                rec["n_decisions"] += 1
                if (p.get("roleInFundingGroup") or "").lower() == "leader":
                    rec["n_as_leader"] += 1
                if first:
                    rec["first_variants"].add(first)
                orcid = (p.get("fundingGroupPersonOrcid") or "").strip()
                if orcid:
                    rec["orcids"].add(orcid)
                y = src.get("fundingStartYear")
                if y and y > 1900:
                    rec["years"].add(y)
        fetched += len(hits)
        print(f"[funding] {fetched} docs scanned", file=sys.stderr)
        after = hits[-1]["sort"][0]
        if len(hits) < PAGE_RFI:
            break

    rows = []
    for r in by_key.values():
        rows.append({
            "first_name": r["first_name"],
            "last_name": r["last_name"],
            "total_eur": round(r["total_eur"], 2),
            "n_decisions": r["n_decisions"],
            "n_as_leader": r["n_as_leader"],
            "first_year": min(r["years"]) if r["years"] else "",
            "last_year": max(r["years"]) if r["years"] else "",
            "orcids": ";".join(sorted(r["orcids"])),
            "first_variants": ";".join(sorted(r["first_variants"])),
        })
    rows.sort(key=lambda x: -x["total_eur"])
    return rows


# ---------------------------------------------------------------------------
# Step 4: Finna master's-thesis supervision lookup
# ---------------------------------------------------------------------------
def _is_supervisor(role: str) -> bool:
    if not role:
        return False
    role_lc = role.lower()
    return role_lc in SUPERVISOR_ROLES or any(t in role_lc for t in SUPERVISOR_ROLES)


def _name_matches(name: str, first: str, last: str) -> bool:
    n = (name or "").lower()
    if last.lower() not in n:
        return False
    f = first.lower().strip()
    if not f:
        return True
    return f in n or (f + ".") in n or (f[:1] + ".") in n


def finna_masters(first_variants: list[str], last: str) -> dict:
    fvs = [f for f in dict.fromkeys(first_variants) if f]
    seen: set[str] = set()
    years: list[int] = []
    for first in fvs:
        page = 1
        while True:
            params = [
                ("lookfor", f"{last}, {first}"),
                ("type", "Author"),
                ("filter[]", "format:1/Thesis/Masters/"),
                ("limit", PAGE_FINNA),
                ("page", page),
                ("field[]", "id"),
                ("field[]", "year"),
                ("field[]", "nonPresenterAuthors"),
            ]
            url = FINNA_URL + "?" + parse.urlencode(params, doseq=True)
            data = _get(url)
            recs = data.get("records") or []
            if not recs:
                break
            for r in recs:
                rid = r.get("id")
                if not rid or rid in seen:
                    continue
                if any(_name_matches(c.get("name", ""), first, last)
                       and _is_supervisor(c.get("role", ""))
                       for c in r.get("nonPresenterAuthors") or []):
                    seen.add(rid)
                    y = r.get("year")
                    if y and re.fullmatch(r"\d{4}", str(y)):
                        years.append(int(y))
            if len(recs) < PAGE_FINNA:
                break
            page += 1
            time.sleep(0.15)
    return {
        "masters": len(seen),
        "thesis_first_year": min(years) if years else "",
        "thesis_last_year": max(years) if years else "",
    }


# ---------------------------------------------------------------------------
# Step 5: anonymized scatter
# ---------------------------------------------------------------------------
def scatter(rows: list[dict], out_path: str) -> None:
    eurs = [float(r["total_eur"] or 0) / 1e6 for r in rows]
    ths = [int(r["masters"] or 0) for r in rows]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(eurs, ths, s=80, alpha=0.7,
               edgecolor="black", linewidth=0.7, color="#1f6feb")
    ax.set_xlabel("Grant funding (M€) per research.fi, CS field of science, 2014–",
                  fontsize=15)
    ax.set_ylabel("Supervised master's theses (Finna, all Finnish universities)",
                  fontsize=15)
    ax.set_title(f"Top-{len(rows)} funded researchers in Finnish CS:\n"
                 "grant euros vs. supervised master's theses",
                 fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[plot] wrote {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40,
                    help="how many top-funded researchers to enrich")
    ap.add_argument("--plot", action="store_true",
                    help="render only; reuse cached enriched.csv")
    args = ap.parse_args()

    funding_path = os.path.join(HERE, "funding_cs_ranked.csv")
    out_path = os.path.join(HERE, "enriched.csv")
    plot_path = os.path.join(HERE, "scatter.png")

    if args.plot and os.path.exists(out_path):
        rows = list(csv.DictReader(open(out_path)))
        scatter(rows[: args.top], plot_path)
        return

    if os.path.exists(funding_path):
        ranked = list(csv.DictReader(open(funding_path)))
        ranked.sort(key=lambda r: -float(r["total_eur"] or 0))
        print(f"[step1] cached: {len(ranked)} CS-tagged researchers", file=sys.stderr)
    else:
        ranked = aggregate_cs_funding()
        with open(funding_path, "w") as f:
            w = csv.DictWriter(f, fieldnames=list(ranked[0].keys()))
            w.writeheader()
            w.writerows(ranked)
        print(f"[step1] wrote {funding_path} ({len(ranked)} researchers)", file=sys.stderr)

    top = ranked[: args.top]
    enriched: list[dict] = []
    for i, r in enumerate(top, 1):
        first, last = r["first_name"], r["last_name"]
        variants = (r.get("first_variants") or first).split(";")
        variants = [v for v in variants if v]
        if first not in variants:
            variants.insert(0, first)
        print(f"[step4] {i:>3}/{len(top)}  {first} {last}  "
              f"€{float(r['total_eur'])/1e6:.2f}M", file=sys.stderr)
        try:
            ths = finna_masters(variants, last)
        except Exception as e:
            print(f"  finna lookup failed: {e}", file=sys.stderr)
            ths = {"masters": 0}
        enriched.append({**r, **ths})
        time.sleep(0.2)

    with open(out_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(enriched[0].keys()))
        w.writeheader()
        w.writerows(enriched)
    print(f"[step4] wrote {out_path}", file=sys.stderr)

    scatter(enriched, plot_path)


if __name__ == "__main__":
    main()
