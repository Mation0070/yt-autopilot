"""Extra research beyond Wikipedia (all free, no API keys):

- News articles: recent coverage found via Bing News RSS (GDELT as backup), full text extracted with trafilatura
- Key facts: Wikidata (founded, founders, HQ, CEO, employees, revenue, net income, market cap...)
- Country data: World Bank (GDP, growth, inflation, population) for the countries the story is about

Every function fails quietly and returns "" / [] so a research hiccup never stops a video.
"""
import re
import time
import urllib.parse as up
import xml.etree.ElementTree as ET

import requests

UA = {"User-Agent": "yt-autopilot/1.0 (https://github.com/Mation0070/yt-autopilot; educational channel bot)"}
BROWSER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/126.0 Safari/537.36"}
SKIP_DOMAINS = ("msn.com", "youtube.com", "facebook.com", "x.com", "twitter.com", "instagram.com", "tiktok.com",
                "linkedin.com", "reddit.com", "wsj.com", "ft.com", "bloomberg.com")  # unreadable or paywalled


def log(*a):
    print("[yt]", *a, flush=True)


# ---------------------------------------------------------------- news articles
def _bing_news(q, n=12):
    try:
        r = requests.get("https://www.bing.com/news/search", params={"q": q, "format": "rss"}, headers=BROWSER, timeout=25)
        out = []
        for it in ET.fromstring(r.content).findall(".//item")[:n]:
            link = it.findtext("link") or ""
            url = up.parse_qs(up.urlparse(link).query).get("url", [link])[0]
            out.append({"url": url, "title": it.findtext("title") or "", "date": (it.findtext("pubDate") or "")[:16]})
        return out
    except Exception as e:
        log("bing news failed:", str(e)[:120])
        return []


def _gdelt(q, n=12):
    for attempt in range(2):
        try:
            r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", headers=UA, timeout=30, params={
                "query": f'"{q}" sourcelang:english', "mode": "artlist", "format": "json", "maxrecords": n,
                "timespan": "12months", "sort": "hybridrel"})
            if r.text.startswith("Please limit"):
                time.sleep(6); continue
            return [{"url": a["url"], "title": a.get("title", ""), "date": a.get("seendate", "")[:8]}
                    for a in r.json().get("articles", [])]
        except Exception:
            time.sleep(3)
    return []


def news_articles(query, max_articles=4, chars_each=2500):
    """Full text of a few recent, readable news articles about the topic."""
    try:
        import trafilatura
    except Exception:
        log("trafilatura missing - skipping article research")
        return []
    cands = _bing_news(query)
    if len(cands) < 4:
        cands += _gdelt(query)
    out, seen_domains = [], set()
    for c in cands:
        if len(out) >= max_articles:
            break
        dom = up.urlparse(c["url"]).netloc.lower().removeprefix("www.")
        if not dom or dom in seen_domains or any(dom.endswith(s) for s in SKIP_DOMAINS):
            continue
        try:
            html = requests.get(c["url"], headers=BROWSER, timeout=20).text
            text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
        except Exception:
            continue
        if len(text) < 900:
            continue
        seen_domains.add(dom)
        out.append({"source": dom, "title": c["title"], "date": c["date"], "text": text[:chars_each]})
    if out:
        log("news research:", ", ".join(a["source"] for a in out))
    return out


# ---------------------------------------------------------------- Wikidata fact sheet
PROPS = {  # property -> label
    "P571": "Founded", "P112": "Founded by", "P159": "Headquarters", "P169": "CEO", "P1037": "Director/manager",
    "P1128": "Employees", "P2139": "Revenue", "P2295": "Net income", "P2226": "Market capitalisation",
    "P2403": "Total assets", "P452": "Industry", "P749": "Parent organisation", "P127": "Owned by",
    "P1082": "Population", "P2131": "GDP (nominal)", "P38": "Currency", "P36": "Capital", "P35": "Head of state",
    "P6": "Head of government", "P2218": "Net worth", "P106": "Occupation", "P569": "Born",
}
CURRENCY_UNITS = {"Q4917": "US$", "Q25417": "EUR", "Q25224": "GBP", "Q8146": "JPY", "Q39099": "CNY", "Q203567": "NGN",
                  "Q181130": "ZAR", "Q202040": "KES", "Q4588": "GHS", "Q199857": "EGP", "Q80524": "INR"}


def _wd(params):
    for attempt in range(3):
        try:
            r = requests.get("https://www.wikidata.org/w/api.php", headers=UA, timeout=25, params={**params, "format": "json"})
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(3 * (attempt + 1))
    return {}


def _qid_for(wiki_title):
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php", headers=UA, timeout=25, params={
            "action": "query", "prop": "pageprops", "ppprop": "wikibase_item", "redirects": 1,
            "titles": wiki_title, "format": "json"}).json()
        page = next(iter(r["query"]["pages"].values()))
        return page.get("pageprops", {}).get("wikibase_item")
    except Exception:
        return None


def _fmt_amount(v):
    a = abs(v)
    for div, w in ((1e12, " trillion"), (1e9, " billion"), (1e6, " million")):
        if a >= div:
            return f"{v / div:,.2f}".rstrip("0").rstrip(".") + w
    return f"{v:,.0f}"


def wikidata_facts(wiki_title):
    """Short fact sheet of key numbers/names, each with its date when Wikidata has one. Returns (text, entity)."""
    qid = _qid_for(wiki_title)
    if not qid:
        return "", {}
    ent = _wd({"action": "wbgetentities", "ids": qid, "props": "claims|labels", "languages": "en"}).get("entities", {}).get(qid, {})
    claims = ent.get("claims", {})
    if not claims:
        return "", {}
    rows, need_labels = [], set()
    for pid, label in PROPS.items():
        vals = []
        # prefer the most recent value (by point-in-time qualifier) for numbers like revenue/employees
        cl = sorted(claims.get(pid, []), key=lambda c: str(((c.get("qualifiers") or {}).get("P585") or [{}])[0]
                                                         .get("datavalue", {}).get("value", {}).get("time", "")), reverse=True)
        for c in cl[:3]:
            dv = c.get("mainsnak", {}).get("datavalue", {})
            v, t = dv.get("value"), dv.get("type")
            when = (((c.get("qualifiers") or {}).get("P585") or [{}])[0].get("datavalue", {}).get("value", {}).get("time", "")[1:5])
            if t == "wikibase-entityid":
                need_labels.add(v["id"]); vals.append(("ID", v["id"], when))
            elif t == "quantity":
                unit = CURRENCY_UNITS.get(v.get("unit", "").split("/")[-1], "")
                vals.append(("TXT", f"{unit} {_fmt_amount(float(v['amount']))}".strip(), when))
            elif t == "time":
                vals.append(("TXT", v["time"][1:5].lstrip("0"), ""))
            elif t == "string":
                vals.append(("TXT", v, when))
            if pid in ("P2139", "P2295", "P2226", "P2403", "P1128", "P1082", "P2131", "P2218"):
                break  # one (latest) value is enough for numbers
        if vals:
            rows.append((label, vals))
    labels = {}
    ids = list(need_labels)
    for i in range(0, len(ids), 40):
        for k, e in _wd({"action": "wbgetentities", "ids": "|".join(ids[i:i + 40]), "props": "labels",
                         "languages": "en"}).get("entities", {}).items():
            labels[k] = e.get("labels", {}).get("en", {}).get("value", "")
    lines = []
    for label, vals in rows:
        parts = []
        for kind, v, when in vals:
            txt = labels.get(v, "") if kind == "ID" else v
            if txt:
                parts.append(txt + (f" ({when})" if when else ""))
        if parts:
            lines.append(f"- {label}: {', '.join(parts)}")
    if lines:
        log("wikidata facts:", len(lines), "lines")
    return ("\n".join(lines), {"claims": claims, "qid": qid})


# ---------------------------------------------------------------- World Bank
_WB = None
INDICATORS = {"NY.GDP.MKTP.CD": "GDP (current US$)", "NY.GDP.MKTP.KD.ZG": "GDP growth (%)",
              "FP.CPI.TOTL.ZG": "Inflation, consumer prices (%)", "SP.POP.TOTL": "Population",
              "NY.GDP.PCAP.CD": "GDP per person (US$)", "NE.EXP.GNFS.ZS": "Exports (% of GDP)"}


def _wb_countries():
    global _WB
    if _WB is None:
        _WB = {}
        try:
            r = requests.get("https://api.worldbank.org/v2/country", params={"format": "json", "per_page": 400}, timeout=30)
            for c in r.json()[1]:
                if c.get("region", {}).get("value") != "Aggregates":
                    _WB[c["name"]] = c["id"]
        except Exception as e:
            log("world bank country list failed:", str(e)[:100])
    return _WB


def countries_in(text, top=2, min_mentions=3):
    """The countries a piece of text talks about most (by name mentions)."""
    found = []
    for name, iso in _wb_countries().items():
        short = re.sub(r",.*$", "", name)  # "Egypt, Arab Rep." -> "Egypt"
        n = len(re.findall(rf"\b{re.escape(short)}\b", text))
        if n >= min_mentions:
            found.append((n, short, iso))
    found.sort(reverse=True)
    return [(s, iso) for _, s, iso in found[:top]]


def world_bank(countries):
    lines = []
    for name, iso in countries:
        vals = []
        for code, label in INDICATORS.items():
            try:
                r = requests.get(f"https://api.worldbank.org/v2/country/{iso}/indicator/{code}",
                                 params={"format": "json", "mrnev": 1}, timeout=25).json()
                row = r[1][0] if len(r) > 1 and r[1] else None
                if row and row.get("value") is not None and int(row.get("date", "0")[:4] or 0) >= time.gmtime().tm_year - 8:
                    v = row["value"]
                    txt = _fmt_amount(v) if abs(v) >= 1e6 else f"{v:,.1f}"
                    vals.append(f"{label}: {txt} ({row['date']})")
            except Exception:
                continue
        if vals:
            lines.append(f"- {name}: " + "; ".join(vals))
    if lines:
        log("world bank data:", ", ".join(n for n, _ in countries))
    return "\n".join(lines)


# ---------------------------------------------------------------- bundle
def extra_research(wiki_title, topic, wiki_text, budget=6000):
    """Returns (text block for the scriptwriter, list of source names used)."""
    blocks, names = [], []
    try:
        facts, _ = wikidata_facts(wiki_title) if wiki_title else ("", {})
        if facts:
            blocks.append("[KEY FACTS - Wikidata]\n" + facts); names.append("Wikidata")
    except Exception as e:
        log("wikidata skipped:", str(e)[:100])
    try:
        cs = countries_in(wiki_text or "")
        wb = world_bank(cs) if cs else ""
        if wb:
            blocks.append("[COUNTRY DATA - World Bank, latest available year]\n" + wb); names.append("World Bank")
    except Exception as e:
        log("world bank skipped:", str(e)[:100])
    try:
        arts = news_articles(wiki_title or topic, max_articles=4, chars_each=max(800, budget // 5))
        for a in arts:
            blocks.append(f"[NEWS - {a['source']}, {a['date']}: {a['title']}]\n{a['text']}")
            names.append(a["source"])
    except Exception as e:
        log("news research skipped:", str(e)[:100])
    text = "\n\n".join(blocks)
    return text[:budget], names
