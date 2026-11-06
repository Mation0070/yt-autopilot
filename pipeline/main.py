"""YouTube autopilot: topic -> research -> script -> voice -> footage + motion graphics -> captions -> render -> upload.

Usage:  python pipeline/main.py long|short [--no-upload]
Env:    GEMINI_API_KEY, PIXABAY_API_KEY (or PEXELS_API_KEY), YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN,
        SHEET_CSV_URL (optional: published Google Sheet; put STOP in cell A1 to pause)
"""
import asyncio, csv, io, json, math, os, pathlib, random, re, shutil, subprocess, sys, time
import xml.etree.ElementTree as ET

import requests, yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import graphics  # noqa: E402  animated maps / numbers / charts / timelines (free, Pillow-drawn)
import music  # noqa: E402  original generated soundtrack (no copyright claims)
import thumbs  # noqa: E402  branded thumbnails

ROOT = pathlib.Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(ROOT / os.environ.get("CONFIG", "config.yaml"), encoding="utf-8"))
STATE_PATH = ROOT / os.environ.get("STATE", "state.json")
WORK = ROOT / "work"
UA = {"User-Agent": "yt-autopilot/1.0 (educational channel bot)"}
LEARN = ""  # weekly analytics learnings, filled in main()
BRAND = {**graphics.DEFAULT_BRAND, **(CFG.get("brand") or {})}
STRUCTURES = [
    "Cold open on a surprising moment, then rewind to how it happened.",
    "Start with a question the viewer can't answer, then reveal the answer step by step.",
    "Tell it as a rise-and-fall (or fall-and-rise) story with a clear turning point.",
    "Myth vs reality: open with what people believe, then show what actually happened.",
    "Follow the money: trace one dollar/naira through the system from start to finish.",
    "Countdown of the 3 key decisions or moments that made the outcome inevitable.",
]


def log(*a):
    print("[yt]", *a, flush=True)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(map(str, cmd[:6]))}...\n{r.stderr[-1500:]}")
    return r.stdout


def duration(path):
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "default=nw=1:nk=1", str(path)]).strip())


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


# ---------------------------------------------------------------- state / control
def load_state():
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {"used": [], "uploads": []}


def save_state(st):
    STATE_PATH.write_text(json.dumps(st, indent=1, ensure_ascii=False))


def sheet_rows():
    url = os.environ.get("SHEET_CSV_URL", "").strip()
    if not url:
        return []
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return [row for row in csv.reader(io.StringIO(r.content.decode("utf-8-sig"))) if row and row[0].strip()]
    except Exception as e:
        log("sheet read failed (ignored):", e)
        return []


# ---------------------------------------------------------------- AI (Gemini, free tier)
_MODELS = None
_GOOD = None


def available_models(key):
    """Ask Google which text models this key can use (models get retired over time)."""
    global _MODELS
    if _MODELS is None:
        _MODELS = []
        try:
            r = requests.get("https://generativelanguage.googleapis.com/v1beta/models", timeout=30,
                             headers={"x-goog-api-key": key}, params={"pageSize": 200})
            for m in r.json().get("models", []):
                name = m["name"].split("/")[-1]
                if "generateContent" in m.get("supportedGenerationMethods", []) and "flash" in name \
                        and not re.search(r"image|tts|live|audio|embed|robotics|preview-\d\d-\d\d", name):
                    _MODELS.append(name)
            def rank(n):
                v = re.search(r"(\d+(?:\.\d+)?)", n)
                return ("lite" in n, bool(re.search(r"preview|exp|omni|latest", n)), -float(v.group(1)) if v else 0)
            _MODELS.sort(key=rank)
            log("models available:", ", ".join(_MODELS[:8]))
        except Exception as e:
            log("model list failed:", e)
    return _MODELS


def gemini(prompt, want_json=True, temperature=0.8):
    key = os.environ["GEMINI_API_KEY"]
    models = [os.environ.get("GEMINI_MODEL") or CFG.get("gemini_model", "gemini-3.8-flash"),
              ] + available_models(key) + ["gemini-flash-latest", "gemini-flash-lite-latest"]
    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": temperature}}
    if want_json:
        body["generationConfig"]["responseMimeType"] = "application/json"
    global _GOOD
    if _GOOD:
        models = [_GOOD] + models
    last = None
    for m in dict.fromkeys(models):
        for attempt in range(2):
            try:
                r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
                                  headers={"x-goog-api-key": key}, json=body, timeout=150)
            except requests.RequestException as e:
                last = str(e); log("gemini error", m, last[:200]); time.sleep(15); continue
            if r.status_code == 200:
                try:
                    txt = "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])
                    if not want_json:
                        _GOOD = m
                        return txt
                    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip()).strip()
                    out = json.loads(txt)
                    _GOOD = m
                    return out
                except Exception as e:
                    last = f"bad response from {m}: {e}"; time.sleep(5); continue
            last = f"{m}: HTTP {r.status_code} {r.text[:300]}"
            log("gemini", last)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(10); continue
            break
    if os.environ.get("GROQ_API_KEY"):
        log("Gemini busy -> switching to Groq backup")
        try:
            return groq(prompt, want_json, temperature)
        except Exception as e:
            last = f"{last} | Groq: {e}"
    raise RuntimeError(f"AI failed: {last}")


def groq(prompt, want_json=True, temperature=0.8):
    key = os.environ["GROQ_API_KEY"]
    h = {"Authorization": f"Bearer {key}"}
    prefer = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "moonshotai/kimi-k2-instruct",
              "openai/gpt-oss-20b", "llama-3.1-8b-instant"]
    try:
        have = [m["id"] for m in requests.get("https://api.groq.com/openai/v1/models", headers=h,
                                              timeout=30).json().get("data", [])]
        models = [m for m in prefer if m in have] + [m for m in have if m not in prefer and
                                                      not re.search(r"whisper|tts|guard|vision|orpheus", m)]
    except Exception:
        models = prefer
    last = None
    for m in models[:5]:
        body = {"model": m, "temperature": temperature,
                "messages": [{"role": "user", "content": prompt[:24000]}]}
        if want_json:
            body["response_format"] = {"type": "json_object"}
            body["messages"][0]["content"] += "\n\nRespond with valid JSON only."
        for attempt in range(2):
            try:
                r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=h, json=body, timeout=120)
            except requests.RequestException as e:
                last = str(e); time.sleep(10); continue
            if r.status_code == 200:
                txt = r.json()["choices"][0]["message"]["content"]
                if not want_json:
                    return txt
                try:
                    return json.loads(re.sub(r"^```(?:json)?|```$", "", txt.strip()).strip())
                except Exception as e:
                    last = f"bad json from {m}: {e}"; continue
            last = f"{m}: HTTP {r.status_code} {r.text[:200]}"
            log("groq", last)
            if r.status_code == 429:
                time.sleep(30); continue
            break
    raise RuntimeError(last)


# ---------------------------------------------------------------- research
def wiki_get(params):
    """Wikipedia API call that survives rate limits / bad responses (retries, then gives up quietly)."""
    for attempt in range(4):
        try:
            r = requests.get("https://en.wikipedia.org/w/api.php", headers=UA, timeout=30,
                             params={**params, "format": "json"})
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(4 * (attempt + 1))
    log("wikipedia unavailable for", params.get("titles") or params.get("srsearch"))
    return {}


def wiki_article(title):
    j = wiki_get({"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1, "titles": title})
    pages = j.get("query", {}).get("pages", {})
    if not pages:
        return None, ""
    page = next(iter(pages.values()))
    if "missing" in page:
        s = wiki_get({"action": "query", "list": "search", "srsearch": title, "srlimit": 1})
        hits = s.get("query", {}).get("search", [])
        if not hits:
            return None, ""
        return wiki_article(hits[0]["title"]) if hits[0]["title"] != title else (None, "")
    text = page.get("extract", "")
    text = re.split(r"\n== (See also|References|External links|Notes|Further reading) ==", text)[0]
    return page.get("title"), text


def related_sources(main_title, topic, n=2, related=None):
    """Extra research: the most relevant other Wikipedia articles, so scripts aren't a retelling of one page."""
    out, seen = [], {norm(main_title or "")}
    for rt in (related or [])[:n]:
        t, text = wiki_article(str(rt))
        if t and norm(t) not in seen and len(text) > 1500:
            seen.add(norm(t)); out.append((t, text))
    for q in (f"{topic} history economics", f"{main_title} business"):
        s = wiki_get({"action": "query", "list": "search", "srsearch": q, "srlimit": 6})
        for h in s.get("query", {}).get("search", []):
            if len(out) >= n:
                return out
            if norm(h["title"]) in seen or re.search(r"\(disambiguation\)|^List of", h["title"]):
                continue
            seen.add(norm(h["title"]))
            t, text = wiki_article(h["title"])
            if t and len(text) > 1500:
                out.append((t, text))
    return out


def build_sources(t):
    """Main article + related articles, trimmed to the configured budget."""
    budget = CFG.get("source_chars", 14000)
    main = (t.get("source") or "")
    try:
        extra = related_sources(t.get("wiki_title") or t["topic"], t["topic"], related=t.get("related"))
    except Exception as e:
        log("related research skipped:", e); extra = []
    if not extra:
        return main[:budget], [t.get("wiki_title")]
    share = int(budget * 0.6)
    rest = (budget - min(len(main), share)) // len(extra)
    txt = f"[{t.get('wiki_title')}]\n" + main[:share]
    for title, text in extra:
        txt += f"\n\n[{title}]\n" + text[:rest]
    log("research sources:", t.get("wiki_title"), "+", ", ".join(x[0] for x in extra))
    return txt, [t.get("wiki_title")] + [x[0] for x in extra]


def news_headlines(q, n=8):
    try:
        r = requests.get("https://news.google.com/rss/search", headers=UA, timeout=30,
                         params={"q": q, **CFG.get("news_region", {"hl": "en-NG", "gl": "NG", "ceid": "NG:en"})})
        items = ET.fromstring(r.content).findall(".//item")[:n]
        return [f"- {i.findtext('title')} ({(i.findtext('pubDate') or '')[:16]})" for i in items]
    except Exception:
        return []


def teaser_topic(state):
    """Shorts funnel: the first Short after a new long video teases it (same research, one killer fact)."""
    if CFG.get("teasers", True) is False:
        return None
    teased = set(state.get("teased", []))
    now = time.time()
    for u in reversed(state.get("uploads", [])):
        if u.get("kind") != "long" or not u.get("id") or not u.get("wiki"):
            continue
        try:
            age = now - time.mktime(time.strptime(u["date"], "%Y-%m-%d %H:%M"))
        except Exception:
            age = 0
        if u["id"] in teased or age > 4 * 86400:
            return None
        title, text = wiki_article(u["wiki"])
        if not text:
            return None
        log("teaser Short for long video:", u["title"])
        return {"topic": u.get("topic") or u["title"], "angle": "", "wiki_title": title, "source": text,
                "key": None, "teaser_of": u}
    return None


def pick_topic(kind, state):
    used = {norm(u) for u in state["used"]}
    # never repeat a topic the other channel already covered
    for other in CFG.get("avoid_from", []):
        try:
            state = {**state, "used": state["used"] + json.loads((ROOT / other).read_text()).get("used", [])}
        except Exception:
            pass
    used = {norm(u) for u in state["used"]}
    rows = sheet_rows()
    if rows and rows[0][0].strip().upper() == "STOP":
        log("STOP found in sheet cell A1 -> skipping this run.")
        sys.exit(0)
    for row in rows:
        t = row[0].strip()
        if t.upper() in ("TOPIC", "TOPICS") or norm(t) in used:
            continue
        title, text = wiki_article(t)
        notes = row[1].strip() if len(row) > 1 else ""
        log("topic from sheet:", t)
        return {"topic": t, "angle": notes, "wiki_title": title, "source": text, "key": t}

    fmt = "60-second YouTube Short with one surprising fact or story" if kind == "short" \
        else f"{CFG.get('long_minutes', 7)}-minute documentary-style YouTube video"
    avoid = "; ".join(state["used"][-200:]) or "none yet"
    for _ in range(3):
        data = gemini(f"""You plan videos for a YouTube channel. Niche: {CFG['niche']}
Propose 10 fresh topics for a {fmt}. Each must be about a real subject that has its own
English Wikipedia article (company, person's business empire, industry, economic event, policy, market).
Mix famous topics with surprising lesser-known ones. Prefer topics people actually search for.
Favour stories with conflict, high stakes, big numbers, rivalries, scandals, collapses or comebacks -
those get clicked and watched to the end.
Do NOT repeat or closely overlap these already-made topics: {avoid}
{LEARN}
Return JSON: {{"topics":[{{"topic":"short working title","wikipedia_title":"exact article title","angle":"the hook/story angle",
"related_wikipedia_titles":["2 other exact English Wikipedia article titles with key background for this story"]}}]}}""")
        cands = data.get("topics", [])
        past = [norm(u) for u in state["used"]] + [norm(u.get("title", "")) for u in state.get("uploads", [])]
        scored = []
        for c in cands[:10]:
            if norm(c.get("topic", "")) in used or norm(c.get("wikipedia_title", "")) in used:
                continue
            if too_similar(c.get("topic", ""), past) or too_similar(c.get("wikipedia_title", ""), past):
                continue
            title, text = wiki_article(c.get("wikipedia_title") or c["topic"])
            if not title or norm(title) in used or too_similar(title, past) or len(text) < 2500:
                continue
            score, why = demand_score(title, c["topic"])
            log(f"research: {c['topic'][:50]} | score {score:.1f} ({why})")
            scored.append((score, c, title, text))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            score, c, title, text = scored[0]
            log(f"auto topic (best demand {score:.1f}):", c["topic"], "| wiki:", title)
            return {"topic": c["topic"], "angle": c.get("angle", ""), "wiki_title": title,
                    "source": text, "key": title, "related": c.get("related_wikipedia_titles") or []}
    raise RuntimeError("could not find a fresh topic")


def too_similar(text, past, limit=0.6):
    """True if a topic shares most of its key words with something already made."""
    stop = {"the", "of", "and", "a", "an", "in", "how", "why", "to", "for", "on", "s", "its", "is", "what", "who", "shorts", "s"}
    w = {x for x in norm(text).split() if x not in stop}
    if not w:
        return False
    for p in past:
        q = {x for x in p.split() if x not in stop}
        if q and len(w & q) / min(len(w), len(q)) >= limit:
            return True
    return False


def demand_score(wiki_title, topic):
    """Research how much interest a topic has, using free public signals:
    Wikipedia readers (last 60 days), recent news coverage, and YouTube search suggestions."""
    import datetime as _dt
    import math as _m
    import urllib.parse as _u
    views = news = sugg = 0
    try:
        end = _dt.date.today() - _dt.timedelta(days=1)
        start = end - _dt.timedelta(days=60)
        r = requests.get("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
                         f"{_u.quote(wiki_title.replace(' ', '_'), safe='')}/daily/{start:%Y%m%d}/{end:%Y%m%d}",
                         headers=UA, timeout=20)
        views = sum(i.get("views", 0) for i in r.json().get("items", []))
    except Exception:
        pass
    try:
        r = requests.get("https://news.google.com/rss/search", headers=UA, timeout=20,
                         params={"q": f'"{wiki_title}" when:30d', **CFG.get("news_region", {"hl": "en-NG", "gl": "NG", "ceid": "NG:en"})})
        news = len(ET.fromstring(r.content).findall(".//item"))
    except Exception:
        pass
    try:
        q = " ".join(wiki_title.split()[:4])
        r = requests.get("https://suggestqueries.google.com/complete/search", headers=UA, timeout=20,
                         params={"client": "firefox", "ds": "yt", "q": q})
        sugg = len(r.json()[1])
    except Exception:
        pass
    score = _m.log10(views + 1) * 2 + min(news, 30) / 6 + sugg / 2
    return score, f"{views} wiki views/60d, {news} news/30d, {sugg} YT searches"


# ---------------------------------------------------------------- script
GRAPHICS_GUIDE = """ANIMATED GRAPHICS (this is what makes the channel look premium and original - use them well):
Some segments get a "graphic": an animated on-screen visual shown while that segment is narrated.
Allowed graphic objects (use ONLY numbers/names/dates that appear in the narration AND the sources):
 {"type":"number","value":"<number, e.g. 32000000000>","prefix":"$","suffix":"","label":"<what it is, max 8 words>"}
 {"type":"map","countries":["<country>", "..."],"label":"<2-5 words>"}            (geography, trade routes, rivals)
 {"type":"bars","title":"<max 6 words>","items":[{"label":"<name>","value":<number>}, ...2-5 items],"unit":"%","prefix":""}
 {"type":"timeline","title":"<max 6 words>","events":[{"year":"<year>","label":"<max 6 words>"}, ...2-5 events]}
 {"type":"compare","title":"<max 6 words>","left":{"label":"<max 5 words>","value":"<short value>"},"right":{"label":"...","value":"..."}}
 {"type":"statement","kicker":"<2-3 words e.g. THE CATCH>","text":"<the key insight, max 14 words>"}
Put the graphic on the segment whose narration mentions that number/place/comparison. Vary the types.
Segments without a graphic: "graphic": null (they get stock footage)."""


def write_script(kind, t):
    sources, source_titles = build_sources(t)
    t["source_titles"] = source_titles
    news = "\n".join(news_headlines(t["wiki_title"] or t["topic"]))
    teaser = t.get("teaser_of")
    if kind == "short" and teaser:
        spec = f"""A YouTube Short TEASER of 110-140 words total (about 45 seconds spoken) for our full documentary
"{teaser['title']}". 5-7 segments. Build it around the single most jaw-dropping fact or twist from the sources.
Segment 1 is a scroll-stopping hook in under 12 words. Leave the viewer wanting the full story. The FINAL segment
must say the full documentary is on our channel (e.g. 'The full story is on our channel - and it gets wilder.').
Title must be under 60 characters, different from the documentary title, and end with ' #Shorts'.
Give EXACTLY 3 segments (never the first one) a graphic."""
    elif kind == "short":
        spec = """A YouTube Short of 120-150 words total (about 50 seconds spoken).
5-7 segments. Segment 1 is a scroll-stopping hook in under 12 words. The FINAL segment must end with a short,
natural call to action asking viewers to tap like and subscribe (e.g. 'If that surprised you, hit like and
subscribe for more.'). Title must be under 60 characters and end with ' #Shorts'.
Give EXACTLY 3 segments (never the first one) a graphic."""
    else:
        spec = f"""A documentary-style YouTube video of {CFG.get('long_words', '950-1150')} words (about {CFG.get('long_minutes', 7)} minutes spoken).
{CFG.get('long_segments', '11-15')} segments. The first segment is a strong 2-3 sentence hook that creates curiosity (no 'welcome back',
no 'in this video'). Tell it as a story with tension: origins, turning points, numbers, what it means.
Every 2-3 minutes re-hook the viewer with an open loop ("but that wasn't the real problem...").
Close with a takeaway and a one-line call to subscribe. Title: under 70 characters, curiosity-driven, not clickbait lies.
Titles: specific and intriguing (a number, a twist or a stakes word), no ALL-CAPS label prefixes like "SCAM:" or "EXPOSED:".
CHAPTERS: split the story into 4-6 chapters. The first segment of each chapter (except the hook) gets
"chapter":"<2-5 word chapter title>"; other segments "chapter": null.
Give about 40% of segments a graphic (never two "statement" graphics in a row)."""
    prompt = f"""You are a top YouTube scriptwriter for this channel: {CFG['niche']}
Write {spec}
Topic: {t['topic']}
Angle / creator notes: {t['angle'] or 'your choice'}

STRICT FACT RULES: Use ONLY facts found in the SOURCES below. Never invent numbers, dates, quotes or events.
If a detail is not in the sources, leave it out. Recent headlines may be mentioned only as headlines say.
Use more than one of the source articles so the story has context nobody else tells.
Write for the ear: short sentences, vivid, conversational, no lists, no markdown, no emojis, no stage directions.
STRUCTURE FOR THIS VIDEO: {random.choice(STRUCTURES)}
HUMAN TOUCH (required): include one clearly-framed opinion or analysis line as the channel's own take
("Here's what most people miss..."), one everyday analogy that makes the idea click, and end with a
genuine question for viewers to answer in the comments. Vary sentence rhythm; never sound templated.
{CFG.get('style', '')}
{LEARN}

{GRAPHICS_GUIDE}

Also give exactly 3 'hashtags': specific, searchable topic hashtags (companies, countries, industries,
e.g. #Dangote #Nigeria #Oil or #NVIDIA #AI #Semiconductors), no spaces, not generic ones like #viral.

THUMBNAIL: "thumb_text" = 2-4 punchy words for the thumbnail (NOT the title repeated; e.g. "$32B FOR A BLUEPRINT"),
"thumb_accent" = the one word of thumb_text to colour-highlight, "thumb_badge" = optional short number like "$140B" or "".

For each segment give a 'visual': a 2-4 word generic STOCK FOOTAGE search phrase that fits, and 'visual_alt': a
different second phrase (e.g. 'oil refinery aerial' / 'oil tanker sea'). Never use people's names or brand names in visuals.

Return JSON:
{{"title":"...","description":"2 short paragraphs summarising the video for YouTube SEO",
"tags":["8-15 tags"],"hashtags":["#Topic","#Topic","#Topic"],"hook_text":"2-5 word punchy ON-SCREEN hook for the first 2 seconds, ALL CAPS, must be true",
"thumb_text":"...","thumb_accent":"...","thumb_badge":"...",
"segments":[{{"narration":"...","visual":"...","visual_alt":"...","chapter":null,"graphic":null}}]}}

SOURCES:
{sources}

RECENT HEADLINES:
{news or 'none'}"""
    s = gemini(prompt)
    # second pass: fact check against sources
    try:
        checked = gemini(f"""You are a strict fact-checker. Compare the SCRIPT JSON to the SOURCES.
Remove or fix any sentence containing a claim (number, date, name, event, quote) not supported by the sources.
Check the numbers, names and dates inside every "graphic" object too: fix them to match the sources, or set
"graphic" to null if they can't be supported.
Keep everything else word-for-word, keep the same JSON structure and keys, keep it flowing naturally.
Return only the corrected JSON.
SCRIPT: {json.dumps(s, ensure_ascii=False)}
SOURCES: {sources}
HEADLINES: {news or 'none'}""", temperature=0.2)
    except Exception as e:
        log("fact-check pass skipped:", str(e)[:200]); checked = None
    if isinstance(checked, dict) and checked.get("segments"):
        s = checked
    s["segments"] = [g for g in s["segments"] if g.get("narration", "").strip()]
    if kind == "short" and "#shorts" not in s["title"].lower():
        s["title"] = s["title"][:90] + " #Shorts"
    t["_sources_text"] = sources
    return s


# ---------------------------------------------------------------- YouTube policy check
def policy_review(kind, s, sources):
    """AI compliance pass against YouTube Community Guidelines + monetisation policies.
    Returns a (possibly corrected) script, or None if the video should not be posted."""
    try:
        r = gemini(f"""You are a strict YouTube policy reviewer. Check this {kind} video script against YouTube's
Community Guidelines and advertiser-friendly / monetisation policies. Look for: misleading or clickbait titles
that the video doesn't deliver; false or unsupported claims (compare with SOURCES); defamation or accusing real
people of crimes not established in the sources; hate, harassment, violence, graphic content; dangerous or
harmful advice; personalised financial/investment advice or get-rich-quick claims; medical claims; profanity;
content unsuitable for advertisers; spammy or irrelevant tags/hashtags; anything copied verbatim at length.
Fix every problem you can by rewriting the affected title/description/narration while keeping the same JSON
structure and keys (keep the "graphic", "chapter", "thumb_*" fields). Keep the title compelling and
curiosity-driven: only rewrite it if it is false or misleading, and keep its hook. Set "ok" false ONLY if the topic
itself cannot be made acceptable.
Return JSON: {{"ok": true/false, "issues": ["..."], "script": <the corrected script JSON>}}
SCRIPT: {json.dumps(s, ensure_ascii=False)}
SOURCES: {sources[:8000]}""", temperature=0.2)
    except Exception as e:
        log("policy review unavailable, using script as written:", str(e)[:150])
        return s
    for i in r.get("issues", [])[:8]:
        log("policy fix:", str(i)[:160])
    if not r.get("ok", True):
        log("policy review REJECTED this topic")
        return None
    fixed = r.get("script")
    if isinstance(fixed, dict) and fixed.get("segments") and fixed.get("title"):
        fixed["segments"] = [g for g in fixed["segments"] if str(g.get("narration", "")).strip()]
        for k in ("thumb_text", "thumb_accent", "thumb_badge", "hook_text"):
            fixed.setdefault(k, s.get(k))
        return fixed
    return s


def clean_meta(script, kind):
    """Make metadata satisfy YouTube's technical limits (no < >, title <=100, tags <=450 chars)."""
    strip = lambda x: re.sub(r"[<>]", "", str(x)).strip()
    title = strip(script.get("title", "Untitled"))[:100]
    if title.isupper():
        title = title.title()
    script["title"] = title
    script["description"] = strip(script.get("description", ""))
    tags, total = [], 0
    for t in script.get("tags", []):
        t = strip(t).replace(",", "")[:30]
        if t and t.lower().strip("#") not in ("viral", "shorts", "trending", "fyp") and total + len(t) + 1 <= 450:
            tags.append(t); total += len(t) + 1
    script["tags"] = tags
    script["hook_text"] = " ".join(str(script.get("hook_text") or "").split()[:4])
    # graphics: keep only well-formed ones, at most one per segment, never on two segments in a row for Shorts
    for i, g in enumerate(script.get("segments", [])):
        spec = graphics.validate(g.get("graphic"))
        if spec and spec["type"] == "chapter":
            spec = None
        if kind == "short" and i == 0:
            spec = None  # Short opens on footage + the big hook text
        g["graphic"] = spec
        if kind != "long":
            g["chapter"] = None
        elif g.get("chapter"):
            g["chapter"] = strip(g["chapter"])[:40]
    return script


# ---------------------------------------------------------------- voice
async def _tts(text, out):
    import edge_tts
    for attempt in range(4):
        try:
            await edge_tts.Communicate(text, CFG.get("voice", "en-US-AndrewNeural"),
                                       rate=CFG.get("voice_rate", "+0%")).save(str(out))
            return
        except Exception as e:
            log("tts retry", e); await asyncio.sleep(5 * (attempt + 1))
    raise RuntimeError("TTS failed")


def voice_segments(segs, d):
    outs = []
    for i, g in enumerate(segs):
        mp3, wav = d / f"a{i:02d}.mp3", d / f"a{i:02d}.wav"
        asyncio.run(_tts(g["narration"], mp3))
        run(["ffmpeg", "-y", "-i", mp3, "-af", "apad=pad_dur=0.35", "-ar", "44100", "-ac", "2", wav])
        outs.append((wav, duration(wav)))
    return outs


# ---------------------------------------------------------------- footage
def pexels_search(query, orientation):
    try:
        r = requests.get("https://api.pexels.com/videos/search", timeout=30,
                         headers={"Authorization": os.environ["PEXELS_API_KEY"]},
                         params={"query": query, "orientation": orientation, "per_page": 20})
        vids = r.json().get("videos", []) if r.status_code == 200 else []
        return [{"id": f"px{v['id']}", "files": [{"link": f["link"], "width": f["width"], "height": f["height"]}
                 for f in v.get("video_files", []) if f.get("file_type") == "video/mp4" and f.get("width")]}
                for v in vids]
    except Exception:
        return []


def pixabay_search(query, orientation):
    try:
        r = requests.get("https://pixabay.com/api/videos/", timeout=30, params={
            "key": os.environ["PIXABAY_API_KEY"], "q": query[:99], "per_page": 30, "safesearch": "true"})
        hits = r.json().get("hits", []) if r.status_code == 200 else []
        out = []
        for h in hits:
            files = [{"link": f["url"], "width": f["width"], "height": f["height"]}
                     for f in h.get("videos", {}).values() if f.get("url") and f.get("width")]
            if orientation == "portrait":
                files = [f for f in files if f["width"] >= 1080 or f["height"] >= 1080] or files
            out.append({"id": f"pb{h['id']}", "portrait": any(f["height"] > f["width"] for f in files), "files": files})
        if orientation == "portrait":
            out.sort(key=lambda v: not v["portrait"])
        return out
    except Exception:
        return []


def stock_search(query, orientation):
    # use both libraries: more variety, fewer repeated clips across videos
    res = []
    if os.environ.get("PIXABAY_API_KEY"):
        res += pixabay_search(query, orientation)
    if os.environ.get("PEXELS_API_KEY") and (len(res) < 12 or random.random() < 0.5):
        px = pexels_search(query, orientation)
        res = [v for pair in zip(px, res) for v in pair] + px[len(res):] + res[len(px):] if res else px
    return res


def best_file(video, W, H):
    files = video.get("files", [])
    if not files:
        return None
    big = [f for f in files if min(f["width"], f["height"]) >= min(W, H) * 0.66]
    pool = big or files
    return min(pool, key=lambda f: f["width"] * f["height"])["link"]


def get_clips(queries, n, orientation, W, H, used_ids, d, tag):
    queries = [q for q in queries if q] or [random.choice(CFG["fallback_footage"])]
    vids = []
    for q in queries:
        vids += [v for v in stock_search(q, orientation) if v["id"] not in used_ids and v not in vids]
        if len(vids) >= n * 2:
            break
    if len(vids) < n:
        q2 = " ".join(queries[0].split()[:1]) if len(queries[0].split()) > 1 else random.choice(CFG["fallback_footage"])
        vids += [v for v in stock_search(q2, orientation) if v["id"] not in used_ids and v not in vids]
    if not vids:
        vids = [v for v in stock_search(random.choice(CFG["fallback_footage"]), orientation) if v["id"] not in used_ids]
    head = vids[:10]; random.shuffle(head); vids = head + vids[10:]
    paths = []
    for v in vids:
        if len(paths) >= n:
            break
        link = best_file(v, W, H)
        if not link:
            continue
        p = d / f"c{tag}_{len(paths)}.mp4"
        try:
            with requests.get(link, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(p, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
            used_ids.add(v["id"]); paths.append(p)
        except Exception as e:
            log("clip download failed", e)
    while paths and len(paths) < n:
        paths.append(random.choice(paths))
    return paths


# ---------------------------------------------------------------- captions
def ts(t):
    h, rem = divmod(max(t, 0), 3600); m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def ass_color(hexcol):
    h = hexcol.lstrip("#")
    return f"&H00{h[4:6]}{h[2:4]}{h[0:2]}&".upper()


def make_ass(segs, audio, W, H, short, path, hook=""):
    size, margin, align = (92, int(H * 0.30), 2) if short else (56, 70, 2)
    words_per = 3 if short else 7
    acc = ass_color(BRAND["accent"])
    fontname = "Poppins ExtraBold"
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{fontname},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,"
        f"{5 if short else 3},1,{align},60,60,{margin},1", "",
        f"Style: Hook,Anton,{150 if short else 110},{acc.rstrip('&')},{acc.rstrip('&')},&H00000000,&H96000000,0,0,0,0,100,100,0,0,1,"
        f"8,2,8,70,70,{int(H * 0.13)},1", "",
        "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    hook = re.sub(r"[{}\\]", "", str(hook or "")).strip().upper()
    if hook:
        lines.append(f"Dialogue: 1,{ts(0)},{ts(2.3)},Hook,,0,0,0,,{{\\fad(0,250)\\fscx60\\fscy60\\t(0,180,\\fscx100\\fscy100)}}{hook}")
    t0 = 0.0
    for g, (_, dur) in zip(segs, audio):
        words = g["narration"].split()
        chunks = [" ".join(words[i:i + words_per]) for i in range(0, len(words), words_per)]
        speak = dur - 0.35
        total = sum(len(c) + 3 for c in chunks) or 1
        t = t0
        for c in chunks:
            dt = speak * (len(c) + 3) / total
            if short:
                # word-by-word highlight: current word in brand accent, others white
                ws = c.upper().split()
                wt = sum(len(w) + 2 for w in ws) or 1
                tw = t
                for k, w in enumerate(ws):
                    wd = dt * (len(w) + 2) / wt
                    txt = " ".join(("{\\c" + acc + "}" + x + "{\\c&HFFFFFF&}") if m == k else x for m, x in enumerate(ws))
                    lines.append(f"Dialogue: 0,{ts(tw)},{ts(tw + wd)},Default,,0,0,0,,{txt}")
                    tw += wd
            else:
                lines.append(f"Dialogue: 0,{ts(t)},{ts(t + dt)},Default,,0,0,0,,{c}")
            t += dt
        t0 += dur
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- render
def stock_part(c, pd, W, H, out, zoom_in):
    """One stock clip: brand colour grade + vignette + slow Ken Burns zoom."""
    z = "min(1+0.0011*on,1.15)" if zoom_in else "max(1.15-0.0011*on,1)"
    vf = (f"scale={int(W * 1.2)}:{int(H * 1.2)}:force_original_aspect_ratio=increase,"
          f"crop={int(W * 1.2)}:{int(H * 1.2)},fps=30,"
          f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={W}x{H}:fps=30,setsar=1,"
          f"eq=contrast=1.07:saturation=1.12:brightness=-0.02,vignette=PI/5")
    run(["ffmpeg", "-y", "-stream_loop", "-1", "-i", c, "-t", f"{pd:.3f}", "-vf", vf, "-an",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-r", "30", out])


def render(kind, script, d):
    short = kind == "short"
    W, H = (1080, 1920) if short else (1920, 1080)
    orient = "portrait" if short else "landscape"
    segs = script["segments"]
    log(f"voicing {len(segs)} segments")
    audio = voice_segments(segs, d)
    used_ids, parts, stock_parts, chapters = set(), [], [], []
    clock, chap_no = 0.0, 0
    for i, (g, (_, dur)) in enumerate(zip(segs, audio)):
        left = dur
        k = 0
        # 1) chapter title card (long videos)
        if g.get("chapter") and dur > 6:
            chap_no += 1
            chapters.append((clock, g["chapter"]))
            cd = 2.6
            try:
                out = d / f"p{i:02d}_ch.mp4"
                graphics.render_graphic({"type": "chapter", "number": chap_no, "title": g["chapter"]}, cd, W, H, BRAND, out, seed=i)
                parts.append(out); left -= cd
            except Exception as e:
                log("chapter card skipped:", str(e)[:150])
        # 2) animated graphic
        spec = g.get("graphic")
        if spec and left > 3.0:
            gd = min(left * (0.8 if short else 0.55), 5.5 if short else 8.0)
            if left - gd < 1.5:
                gd = left
            try:
                out = d / f"p{i:02d}_g.mp4"
                graphics.render_graphic(spec, gd, W, H, BRAND, out, seed=i)
                parts.append(out); left -= gd
                log(f"scene {i + 1}: {spec['type']} graphic ({gd:.1f}s)")
            except Exception as e:
                log("graphic skipped:", spec.get("type"), str(e)[:150])
        # 3) stock footage for the rest of the segment
        if left > 0.05:
            n = max(1, math.ceil(left / (2.3 if short else 5.0)))
            clips = get_clips([g.get("visual"), g.get("visual_alt")], n, orient, W, H, used_ids, d, i)
            if not clips:
                raise RuntimeError("no footage found")
            pd = left / n
            for j, c in enumerate(clips):
                out = d / f"p{i:02d}_{j}.mp4"
                stock_part(c, pd, W, H, out, (i + j) % 2 == 0)
                parts.append(out); stock_parts.append(out)
        clock += dur
        log(f"scene {i + 1}/{len(segs)} done")
    (d / "v.txt").write_text("".join(f"file '{p.name}'\n" for p in parts))
    (d / "a.txt").write_text("".join(f"file '{w.name}'\n" for w, _ in audio))
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", d / "v.txt", "-c", "copy", d / "video.mp4"])
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", d / "a.txt", "-c", "copy", d / "narration.wav"])
    T = sum(x for _, x in audio)
    add_music(d, T, CFG.get("music_volume_short", 0.13) if short else CFG.get("music_volume_long", 0.09))
    make_ass(segs, audio, W, H, short, d / "subs.ass", script.get("hook_text", "") if short else "")
    fdir = graphics.fonts_dir()
    subs = f"ass={d / 'subs.ass'}:fontsdir={fdir}"
    final = d / "final.mp4"
    if short:
        # animated like + subscribe button in the last seconds, with a "tap" and a click sound
        a_png, b_png, tap_png = cta_pngs(d, W)
        t0, t1 = max(1.0, T - 4.5), max(1.5, T - 2.6)
        y = int(H * 0.16)
        fc = (f"[0:v]{subs}[v0];"
              f"[2:v]format=rgba[ca];[3:v]format=rgba[cb];[4:v]format=rgba[tp];"
              f"[v0][ca]overlay=x=60:y='{y}+max(0,({t0}+0.25-t))*600':enable='between(t,{t0},{t1})'[v1];"
              f"[v1][cb]overlay=x=60:y={y}:enable='gte(t,{t1})'[v2];"
              f"[v2][tp]overlay=x=80:y={y + 15}:enable='between(t,{t1 - 0.35},{t1 + 0.25})'[v];"
              f"aevalsrc='0.6*sin(2*PI*1400*t)*exp(-40*t)':d=0.15:s=44100,aformat=channel_layouts=stereo,"
              f"adelay={int(t1 * 1000)}|{int(t1 * 1000)}[clk];[1:a][clk]amix=inputs=2:duration=first:normalize=0[a]")
        run(["ffmpeg", "-y", "-i", str(d / "video.mp4"), "-i", str(d / "audio.wav"),
             "-loop", "1", "-i", str(a_png), "-loop", "1", "-i", str(b_png), "-loop", "1", "-i", str(tap_png),
             "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "160k", "-t", f"{T:.2f}", "-movflags", "+faststart", str(final)])
    else:
        run(["ffmpeg", "-y", "-i", str(d / "video.mp4"), "-i", str(d / "audio.wav"), "-vf", subs,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(final)])
    log(f"rendered {final} ({duration(final):.1f}s)")
    script["_chapters"] = chapters
    script["_stock_parts"] = [str(p) for p in stock_parts]
    return final


def add_music(d, T, vol):
    """Original generated soundtrack (new key/tempo/progression every video) ducked under the narration."""
    try:
        music.make_loop(d / "pad.wav", seed=random.randint(0, 10 ** 6))
    except Exception as e:
        log("music generator failed, using simple pad:", str(e)[:150])
        run(["ffmpeg", "-y", "-f", "lavfi", "-i", "aevalsrc='0.2*sin(2*PI*110*t)+0.15*sin(2*PI*164.8*t)':s=44100:d=8",
             str(d / "pad.wav")])
    run(["ffmpeg", "-y", "-i", str(d / "narration.wav"), "-stream_loop", "-1", "-i", str(d / "pad.wav"),
         "-filter_complex",
         f"[1:a]aresample=44100,aformat=channel_layouts=stereo,volume={vol},"
         f"afade=t=in:d=1.5,afade=t=out:st={max(0.0, T - 2.5):.2f}:d=2.5[bg];"
         f"[0:a]asplit=2[n1][n2];"
         f"[bg][n1]sidechaincompress=threshold=0.05:ratio=4:attack=20:release=400[duck];"
         f"[n2][duck]amix=inputs=2:duration=first:normalize=0[a]",
         "-map", "[a]", "-t", f"{T:.2f}", "-ar", "44100", "-ac", "2", str(d / "audio.wav")])


def cta_pngs(d, W):
    """Draw the animated like/subscribe call-to-action: state A (before tap) and state B (after tap)."""
    from PIL import Image, ImageDraw, ImageFont
    f = graphics.font("heavy", 56)

    def thumb(dr, x, y, s, col):  # simple thumbs-up icon
        dr.rounded_rectangle([x, y + 0.42 * s, x + 0.22 * s, y + s], radius=int(0.05 * s), fill=col)
        dr.rounded_rectangle([x + 0.27 * s, y + 0.40 * s, x + s, y + s], radius=int(0.14 * s), fill=col)
        dr.polygon([(x + 0.30 * s, y + 0.45 * s), (x + 0.50 * s, y), (x + 0.66 * s, y + 0.06 * s),
                    (x + 0.60 * s, y + 0.44 * s)], fill=col)

    out = []
    for liked in (False, True):
        img = Image.new("RGBA", (W - 120, 170), (0, 0, 0, 0))
        dr = ImageDraw.Draw(img)
        lw = 400
        dr.rounded_rectangle([0, 0, lw, 170], radius=85, fill=(6, 95, 212, 255) if liked else (255, 255, 255, 235))
        thumb(dr, 45, 38, 90, (255, 255, 255, 255) if liked else (20, 20, 20, 255))
        dr.text((160, 48), "LIKED" if liked else "LIKE", font=f, fill=(255, 255, 255) if liked else (20, 20, 20))
        x2 = lw + 30
        dr.rounded_rectangle([x2, 0, W - 120, 170], radius=85,
                             fill=(90, 90, 90, 245) if liked else (230, 0, 18, 255))
        dr.text((x2 + 45, 48), "SUBSCRIBED" if liked else "SUBSCRIBE", font=f, fill=(255, 255, 255))
        pth = d / f"cta_{int(liked)}.png"
        img.save(pth)
        out.append(pth)
    # tap ripple
    tap = Image.new("RGBA", (140, 140), (0, 0, 0, 0))
    td = ImageDraw.Draw(tap)
    td.ellipse([10, 10, 130, 130], outline=(255, 255, 255, 230), width=10)
    td.ellipse([45, 45, 95, 95], fill=(255, 255, 255, 230))
    tp = d / "cta_tap.png"
    tap.save(tp)
    return out + [tp]


def thumbnail(script, d):
    """Branded thumbnail from the most vivid stock frame + 2-4 bold words."""
    frame = d / "frame.jpg"
    cands = script.get("_stock_parts") or [str(d / "video.mp4")]
    best, best_score = None, -1
    from PIL import Image, ImageStat
    for k, p in enumerate(cands[:12]):
        f = d / f"fr{k}.jpg"
        try:
            run(["ffmpeg", "-y", "-ss", "1", "-i", p, "-frames:v", "1", "-vf", "scale=1280:-2", f])
            st = ImageStat.Stat(Image.open(f).convert("HSV"))
            score = st.mean[1] * 0.6 + st.stddev[2] * 1.0 + min(st.mean[2], 150) * 0.3  # colourful, contrasty, not dark
            if score > best_score:
                best, best_score = f, score
        except Exception:
            continue
    if best is None:
        run(["ffmpeg", "-y", "-ss", "4", "-i", str(d / "video.mp4"), "-frames:v", "1", "-vf", "scale=1280:720", frame])
        best = frame
    text = script.get("thumb_text") or re.sub(r"#\w+", "", script["title"])
    out = d / "thumb.jpg"
    thumbs.make_thumbnail(best, text, out, BRAND, script.get("thumb_accent"), script.get("thumb_badge") or None)
    return out


# ---------------------------------------------------------------- description
def build_description(script, kind, state, topic):
    parts = [script.get("description", "")]
    if kind == "short" and topic.get("teaser_of"):
        parts.insert(0, f"Full documentary: https://youtu.be/{topic['teaser_of']['id']}")
    chs = script.get("_chapters") or []
    if kind == "long" and chs:
        # YouTube chapters need: first at 0:00, at least 3, each at least 10 seconds long
        marks = [(0.0, "Intro")] + [(t, c) for t, c in chs if t >= 10]
        good = [marks[0]]
        for t, c in marks[1:]:
            if t - good[-1][0] >= 10:
                good.append((t, c))
        if len(good) >= 3:
            parts.append("Chapters:\n" + "\n".join(f"{int(t // 60)}:{int(t % 60):02d} {c}" for t, c in good))
    if kind == "long":
        prev = [u for u in state.get("uploads", []) if u.get("kind") == "long" and u.get("id")][-3:]
        if prev:
            parts.append("Watch next:\n" + "\n".join(f"- {u['title']}: https://youtu.be/{u['id']}" for u in reversed(prev)))
    src = ", ".join(x for x in (topic.get("source_titles") or []) if x) or "Wikipedia"
    parts.append(f"Sources: Wikipedia ({src}) and public news reports.\nNarration uses an AI voice; "
                 "research, script and graphics are produced for this channel.")
    return "\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------- upload
def upload(video, thumb, script, kind, desc):
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    creds = Credentials(None, refresh_token=os.environ["YT_REFRESH_TOKEN"], client_id=os.environ["YT_CLIENT_ID"],
                        client_secret=os.environ["YT_CLIENT_SECRET"], token_uri="https://oauth2.googleapis.com/token")
    yt = build("youtube", "v3", credentials=creds, cache_discovery=False)
    hs = []
    for h in script.get("hashtags", []):
        h = "#" + re.sub(r"[^A-Za-z0-9]", "", str(h))
        if len(h) > 2 and h.lower() not in ("#viral", "#shorts", "#trending", "#fyp") and h not in hs:
            hs.append(h)
    hs = hs[:3] + (["#Shorts"] if kind == "short" else [])
    desc = desc + "\n\n" + " ".join(hs)
    tags = [t[:30] for t in script.get("tags", [])][:15]
    body = {"snippet": {"title": script["title"][:100], "description": desc[:4900], "tags": tags,
                        "categoryId": "27", "defaultLanguage": "en", "defaultAudioLanguage": "en"},
            "status": {"privacyStatus": CFG.get("privacy", "public"), "selfDeclaredMadeForKids": False,
                       "containsSyntheticMedia": False}}
    req = yt.videos().insert(part="snippet,status", body=body,
                             media_body=MediaFileUpload(str(video), chunksize=8 * 1024 * 1024, resumable=True))
    resp = None
    while resp is None:
        _, resp = req.next_chunk()
    vid = resp["id"]
    log("uploaded: https://youtu.be/" + vid, "| privacy:", resp.get("status", {}).get("privacyStatus"))
    if thumb:
        try:
            yt.thumbnails().set(videoId=vid, media_body=MediaFileUpload(str(thumb))).execute()
        except Exception as e:
            log("thumbnail skipped (verify your channel by phone to enable custom thumbnails):", str(e)[:200])
    return vid


# ---------------------------------------------------------------- main
def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "long"
    no_upload = "--no-upload" in sys.argv
    state = load_state()
    global LEARN
    ins = state.get("insights") or {}
    if ins.get("learnings") or ins.get("double_down"):
        LEARN = ("WHAT'S WORKING ON THIS CHANNEL (from last week's analytics - follow it):\n- "
                 + "\n- ".join(ins.get("learnings", []))
                 + ("\nMore like: " + "; ".join(ins.get("double_down", [])) if ins.get("double_down") else "")
                 + ("\nAvoid: " + "; ".join(ins.get("avoid", [])) if ins.get("avoid") else ""))
    tune = state.get("tuning") or {}
    if tune.get("long_minutes"):
        m = int(tune["long_minutes"])
        CFG.update(long_minutes=m, long_words=f"{m * 140}-{m * 160}",
                   long_segments=f"{max(8, round(m * 1.3))}-{round(m * 1.8)}")
        log("auto-tuned long video length:", m, "min")
    if kind == "short" and os.environ.get("GITHUB_EVENT_NAME") == "schedule":
        today = time.strftime("%Y-%m-%d")
        done = sum(1 for u in state.get("uploads", []) if u.get("kind") == "short" and str(u.get("date", "")).startswith(today))
        if done >= CFG.get("max_shorts_per_day", 2):
            log(f"already posted {done} Shorts today -> skipping this slot")
            return
    log("starting", kind, "video")
    teaser = teaser_topic(state) if kind == "short" else None
    for attempt in range(3):
        topic = teaser if (teaser and attempt == 0) else pick_topic(kind, state)
        script = write_script(kind, topic)
        script = policy_review(kind, script, topic.get("_sources_text") or topic.get("source") or "")
        if script:
            break
        if topic.get("key"):
            state["used"].append(topic["key"]); save_state(state)  # never retry a rejected topic
    else:
        raise RuntimeError("no policy-safe topic found this run")
    script = clean_meta(script, kind)
    log("title:", script["title"])
    d = WORK / kind
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    (d / "script.json").write_text(json.dumps(script, indent=1, ensure_ascii=False))
    video = render(kind, script, d)
    secs = duration(video)
    has_audio = "audio" in run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(video)])
    if not has_audio or secs < 15 or (kind == "short" and secs > 175) or (kind == "long" and secs < 60):
        raise RuntimeError(f"video failed quality check (length {secs:.0f}s, audio={has_audio})")
    thumb = thumbnail(script, d) if kind == "long" else None
    desc = build_description(script, kind, state, topic)
    (d / "description.txt").write_text(desc, encoding="utf-8")
    vid = None if no_upload else upload(video, thumb, script, kind, desc)
    if topic.get("key"):
        state["used"].append(topic["key"])
    if topic.get("teaser_of") and topic["teaser_of"].get("id"):
        state.setdefault("teased", []).append(topic["teaser_of"]["id"])
        state["teased"] = state["teased"][-50:]
    state["uploads"].append({"kind": kind, "title": script["title"], "id": vid,
                             "date": time.strftime("%Y-%m-%d %H:%M"), "topic": topic.get("topic"),
                             "wiki": topic.get("wiki_title"),
                             "graphics": sum(1 for g in script["segments"] if g.get("graphic"))})
    save_state(state)


if __name__ == "__main__":
    main()
