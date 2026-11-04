"""YouTube autopilot: topic -> research -> script -> voice -> footage -> captions -> render -> upload.

Usage:  python pipeline/main.py long|short [--no-upload]
Env:    GEMINI_API_KEY, PIXABAY_API_KEY (or PEXELS_API_KEY), YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN,
        SHEET_CSV_URL (optional: published Google Sheet; put STOP in cell A1 to pause)
"""
import asyncio, csv, io, json, math, os, pathlib, random, re, shutil, subprocess, sys, time
import xml.etree.ElementTree as ET

import requests, yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(ROOT / os.environ.get("CONFIG", "config.yaml"), encoding="utf-8"))
STATE_PATH = ROOT / os.environ.get("STATE", "state.json")
WORK = ROOT / "work"
UA = {"User-Agent": "yt-autopilot/1.0 (educational channel bot)"}
LEARN = ""  # weekly analytics learnings, filled in main()
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


def news_headlines(q, n=8):
    try:
        r = requests.get("https://news.google.com/rss/search", headers=UA, timeout=30,
                         params={"q": q, **CFG.get("news_region", {"hl": "en-NG", "gl": "NG", "ceid": "NG:en"})})
        items = ET.fromstring(r.content).findall(".//item")[:n]
        return [f"- {i.findtext('title')} ({(i.findtext('pubDate') or '')[:16]})" for i in items]
    except Exception:
        return []


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
Do NOT repeat or closely overlap these already-made topics: {avoid}
{LEARN}
Return JSON: {{"topics":[{{"topic":"short working title","wikipedia_title":"exact article title","angle":"the hook/story angle"}}]}}""")
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
                    "source": text, "key": title}
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
def write_script(kind, t):
    sources = (t["source"] or "")[:CFG.get("source_chars", 14000)]
    news = "\n".join(news_headlines(t["wiki_title"] or t["topic"]))
    if kind == "short":
        spec = """A YouTube Short of 120-150 words total (about 50 seconds spoken).
5-7 segments. Segment 1 is a scroll-stopping hook in under 12 words. The FINAL segment must end with a short,
natural call to action asking viewers to tap like and subscribe (e.g. 'If that surprised you, hit like and
subscribe for more.'). Title must be under 60 characters and end with ' #Shorts'."""
    else:
        spec = f"""A documentary-style YouTube video of {CFG.get('long_words', '950-1150')} words (about {CFG.get('long_minutes', 7)} minutes spoken).
{CFG.get('long_segments', '11-15')} segments. The first segment is a strong 2-3 sentence hook that creates curiosity (no 'welcome back',
no 'in this video'). Tell it as a story with tension: origins, turning points, numbers, what it means.
Close with a takeaway and a one-line call to subscribe. Title: under 70 characters, curiosity-driven, not clickbait lies."""
    prompt = f"""You are a top YouTube scriptwriter for this channel: {CFG['niche']}
Write {spec}
Topic: {t['topic']}
Angle / creator notes: {t['angle'] or 'your choice'}

STRICT FACT RULES: Use ONLY facts found in the SOURCES below. Never invent numbers, dates, quotes or events.
If a detail is not in the sources, leave it out. Recent headlines may be mentioned only as headlines say.
Write for the ear: short sentences, vivid, conversational, no lists, no markdown, no emojis, no stage directions.
STRUCTURE FOR THIS VIDEO: {random.choice(STRUCTURES)}
HUMAN TOUCH (required): include one clearly-framed opinion or analysis line as the channel's own take
("Here's what most people miss..."), one everyday analogy that makes the idea click, and end with a
genuine question for viewers to answer in the comments. Vary sentence rhythm; never sound templated.
{CFG.get('style', '')}
{LEARN}

Also give exactly 3 'hashtags': specific, searchable topic hashtags (companies, countries, industries,
e.g. #Dangote #Nigeria #Oil or #NVIDIA #AI #Semiconductors), no spaces, not generic ones like #viral.

For each segment give a 'visual': a 2-4 word generic STOCK FOOTAGE search phrase that fits
(e.g. 'oil refinery aerial', 'lagos traffic', 'cement factory', 'phone payment app'). Never use people's names or brand names in visuals.

Return JSON:
{{"title":"...","description":"2 short paragraphs summarising the video for YouTube SEO",
"tags":["8-15 tags"],"hashtags":["#Topic","#Topic","#Topic"],"hook_text":"2-5 word punchy ON-SCREEN hook for the first 2 seconds, ALL CAPS, must be true","segments":[{{"narration":"...","visual":"..."}}]}}

SOURCES (Wikipedia: {t['wiki_title']}):
{sources}

RECENT HEADLINES:
{news or 'none'}"""
    s = gemini(prompt)
    # second pass: fact check against sources
    try:
        checked = gemini(f"""You are a strict fact-checker. Compare the SCRIPT JSON to the SOURCES.
Remove or fix any sentence containing a claim (number, date, name, event, quote) not supported by the sources.
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
structure and keys. Set "ok" false ONLY if the topic itself cannot be made acceptable.
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
        if t and total + len(t) + 1 <= 450:
            tags.append(t); total += len(t) + 1
    script["tags"] = tags
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
    if os.environ.get("PIXABAY_API_KEY"):
        res = pixabay_search(query, orientation)
        if res or not os.environ.get("PEXELS_API_KEY"):
            return res
    return pexels_search(query, orientation)


def best_file(video, W, H):
    files = video.get("files", [])
    if not files:
        return None
    big = [f for f in files if min(f["width"], f["height"]) >= min(W, H) * 0.66]
    pool = big or files
    return min(pool, key=lambda f: f["width"] * f["height"])["link"]


def get_clips(query, n, orientation, W, H, used_ids, d, tag):
    vids = [v for v in stock_search(query, orientation) if v["id"] not in used_ids]
    if len(vids) < n:
        q2 = " ".join(query.split()[:1]) if len(query.split()) > 1 else random.choice(CFG["fallback_footage"])
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


def make_ass(segs, audio, W, H, short, path, hook=""):
    size, margin, align = (96, int(H * 0.30), 2) if short else (54, 70, 2)
    words_per = 3 if short else 7
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,DejaVu Sans,{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,"
        f"{5 if short else 3},1,{align},60,60,{margin},1", "",
        f"Style: Hook,DejaVu Sans,{118 if short else 90},&H0000E5FF,&H0000E5FF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,"
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
                # word-by-word highlight: current word yellow, others white
                ws = c.upper().split()
                wt = sum(len(w) + 2 for w in ws) or 1
                tw = t
                for k, w in enumerate(ws):
                    wd = dt * (len(w) + 2) / wt
                    txt = " ".join(("{\\c&H00E5FF&}" + x + "{\\c&HFFFFFF&}") if m == k else x for m, x in enumerate(ws))
                    lines.append(f"Dialogue: 0,{ts(tw)},{ts(tw + wd)},Default,,0,0,0,,{txt}")
                    tw += wd
            else:
                lines.append(f"Dialogue: 0,{ts(t)},{ts(t + dt)},Default,,0,0,0,,{c}")
            t += dt
        t0 += dur
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- render
def render(kind, script, d):
    short = kind == "short"
    W, H = (1080, 1920) if short else (1920, 1080)
    orient = "portrait" if short else "landscape"
    segs = script["segments"]
    log(f"voicing {len(segs)} segments")
    audio = voice_segments(segs, d)
    used_ids, parts = set(), []
    for i, (g, (_, dur)) in enumerate(zip(segs, audio)):
        n = max(1, math.ceil(dur / (2.3 if short else 5.0)))
        clips = get_clips(g.get("visual") or "city", n, orient, W, H, used_ids, d, i)
        if not clips:
            raise RuntimeError("no footage found")
        pd = dur / n
        for j, c in enumerate(clips):
            out = d / f"p{i:02d}_{j}.mp4"
            # slow Ken Burns zoom, alternating in / out
            z = "min(1+0.0011*on,1.15)" if (i + j) % 2 == 0 else "max(1.15-0.0011*on,1)"
            vf = (f"scale={int(W * 1.2)}:{int(H * 1.2)}:force_original_aspect_ratio=increase,"
                  f"crop={int(W * 1.2)}:{int(H * 1.2)},fps=30,"
                  f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={W}x{H}:fps=30,setsar=1")
            run(["ffmpeg", "-y", "-stream_loop", "-1", "-i", c, "-t", f"{pd:.3f}", "-vf", vf, "-an",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-r", "30", out])
            parts.append(out)
        log(f"scene {i + 1}/{len(segs)} done")
    (d / "v.txt").write_text("".join(f"file '{p.name}'\n" for p in parts))
    (d / "a.txt").write_text("".join(f"file '{w.name}'\n" for w, _ in audio))
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", d / "v.txt", "-c", "copy", d / "video.mp4"])
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", d / "a.txt", "-c", "copy", d / "narration.wav"])
    add_music(d, sum(x for _, x in audio), 0.11 if short else 0.07)
    make_ass(segs, audio, W, H, short, d / "subs.ass", script.get("hook_text", "") if short else "")
    final = d / "final.mp4"
    if short:
        # animated like + subscribe button in the last seconds, with a "tap" and a click sound
        a_png, b_png, tap_png = cta_pngs(d, W)
        T = sum(x for _, x in audio)
        t0, t1 = max(1.0, T - 4.5), max(1.5, T - 2.6)
        y = int(H * 0.16)
        fc = (f"[0:v]ass={d / 'subs.ass'}[v0];"
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
        run(["ffmpeg", "-y", "-i", str(d / "video.mp4"), "-i", str(d / "audio.wav"), "-vf", f"ass={d / 'subs.ass'}",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(final)])
    log(f"rendered {final} ({duration(final):.1f}s)")
    return final


def add_music(d, T, vol):
    """Soft, original background pad (generated, so no copyright issues) mixed under the narration."""
    progs = [[(220, 261.63, 329.63), (174.61, 220, 261.63), (130.81, 164.81, 196), (196, 246.94, 293.66)],
             [(146.83, 174.61, 220), (116.54, 146.83, 174.61), (174.61, 220, 261.63), (130.81, 164.81, 196)]]
    chords = random.choice(progs)
    ins, parts = [], []
    for k, (f1, f2, f3) in enumerate(chords):
        expr = (f"(0.22*sin(2*PI*{f1}*t)+0.18*sin(2*PI*{f2}*t)+0.16*sin(2*PI*{f3}*t)+0.14*sin(2*PI*{f1 / 2}*t))"
                f"*(0.85+0.15*sin(2*PI*0.25*t))")
        ins += ["-f", "lavfi", "-i", f"aevalsrc='{expr}':s=44100:d=4"]
        parts.append(f"[{k}:a]afade=t=in:d=1.2,afade=t=out:st=2.6:d=1.4[c{k}]")
    fc = ";".join(parts) + ";" + "".join(f"[c{k}]" for k in range(4)) + "concat=n=4:v=0:a=1,aformat=channel_layouts=stereo[m]"
    run(["ffmpeg", "-y", *ins, "-filter_complex", fc, "-map", "[m]", str(d / "pad.wav")])
    run(["ffmpeg", "-y", "-i", str(d / "narration.wav"), "-stream_loop", "-1", "-i", str(d / "pad.wav"),
         "-filter_complex", f"[1:a]lowpass=f=1400,aecho=0.8:0.6:90|160:0.25|0.15,volume={vol}[bg];"
                            f"[0:a][bg]amix=inputs=2:duration=first:normalize=0[a]",
         "-map", "[a]", "-t", f"{T:.2f}", "-ar", "44100", "-ac", "2", str(d / "audio.wav")])


def cta_pngs(d, W):
    """Draw the animated like/subscribe call-to-action: state A (before tap) and state B (after tap)."""
    from PIL import Image, ImageDraw, ImageFont
    font_path = run(["fc-match", "-f", "%{file}", "DejaVu Sans:bold"]).strip()
    f = ImageFont.truetype(font_path, 58)

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
        dr.text((160, 52), "LIKED" if liked else "LIKE", font=f, fill=(255, 255, 255) if liked else (20, 20, 20))
        x2 = lw + 30
        dr.rounded_rectangle([x2, 0, W - 120, 170], radius=85,
                             fill=(90, 90, 90, 245) if liked else (230, 0, 18, 255))
        dr.text((x2 + 45, 52), "SUBSCRIBED \u2713" if liked else "SUBSCRIBE", font=f, fill=(255, 255, 255))
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


def thumbnail(video, title, d):
    from PIL import Image, ImageDraw, ImageEnhance, ImageFont
    frame = d / "frame.jpg"
    run(["ffmpeg", "-y", "-ss", "4", "-i", video, "-frames:v", "1", "-vf", "scale=1280:720", frame])
    img = ImageEnhance.Brightness(Image.open(frame).convert("RGB")).enhance(0.55)
    font_path = run(["fc-match", "-f", "%{file}", "DejaVu Sans:bold"]).strip()
    words = re.sub(r"#\w+", "", title).upper().split()
    text = " ".join(words[:7])
    dr = ImageDraw.Draw(img)
    for size in range(120, 40, -6):
        font = ImageFont.truetype(font_path, size)
        lines, cur = [], ""
        for w in text.split():
            test = (cur + " " + w).strip()
            if dr.textlength(test, font=font) <= 1160:
                cur = test
            else:
                lines.append(cur); cur = w
        lines.append(cur)
        if len(lines) <= 3:
            break
    y = 720 - 60 - len(lines) * (size + 10)
    for k, ln in enumerate(lines):
        dr.text((60, y), ln, font=font, fill=(255, 214, 0) if k == 0 else (255, 255, 255),
                stroke_width=6, stroke_fill=(0, 0, 0))
        y += size + 10
    out = d / "thumb.jpg"
    img.save(out, quality=90)
    return out


# ---------------------------------------------------------------- upload
def upload(video, thumb, script, kind):
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    creds = Credentials(None, refresh_token=os.environ["YT_REFRESH_TOKEN"], client_id=os.environ["YT_CLIENT_ID"],
                        client_secret=os.environ["YT_CLIENT_SECRET"], token_uri="https://oauth2.googleapis.com/token")
    yt = build("youtube", "v3", credentials=creds, cache_discovery=False)
    hs = []
    for h in script.get("hashtags", []):
        h = "#" + re.sub(r"[^A-Za-z0-9]", "", str(h))
        if len(h) > 2 and h.lower() not in ("#viral", "#shorts") and h not in hs:
            hs.append(h)
    hs = hs[:3] + (["#Shorts"] if kind == "short" else []) + ["#viral"]
    desc = (script.get("description", "") + "\n\nSources: Wikipedia and public news reports."
            "\nNarration uses an AI voice.\n\n" + " ".join(hs))
    tags = ["viral"] + [t[:30] for t in script.get("tags", []) if t.lower().strip("#") != "viral"][:14]
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
    for _ in range(3):
        topic = pick_topic(kind, state)
        script = write_script(kind, topic)
        script = policy_review(kind, script, topic.get("source") or "")
        if script:
            break
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
    thumb = thumbnail(d / "video.mp4", script["title"], d) if kind == "long" else None
    vid = None if no_upload else upload(video, thumb, script, kind)
    state["used"].append(topic["key"])
    state["uploads"].append({"kind": kind, "title": script["title"], "id": vid,
                             "date": time.strftime("%Y-%m-%d %H:%M")})
    save_state(state)


if __name__ == "__main__":
    main()
