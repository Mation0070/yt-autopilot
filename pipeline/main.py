"""YouTube autopilot: topic -> research -> script -> voice -> footage -> captions -> render -> upload.

Usage:  python pipeline/main.py long|short [--no-upload]
Env:    GEMINI_API_KEY, PIXABAY_API_KEY (or PEXELS_API_KEY), YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN,
        SHEET_CSV_URL (optional: published Google Sheet; put STOP in cell A1 to pause)
"""
import asyncio, csv, io, json, math, os, pathlib, random, re, shutil, subprocess, sys, time
import xml.etree.ElementTree as ET

import requests, yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
STATE_PATH = ROOT / "state.json"
WORK = ROOT / "work"
UA = {"User-Agent": "yt-autopilot/1.0 (educational channel bot)"}


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
def wiki_article(title):
    r = requests.get("https://en.wikipedia.org/w/api.php", headers=UA, timeout=30, params={
        "action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
        "titles": title, "format": "json"})
    page = next(iter(r.json()["query"]["pages"].values()))
    if "missing" in page:
        s = requests.get("https://en.wikipedia.org/w/api.php", headers=UA, timeout=30, params={
            "action": "query", "list": "search", "srsearch": title, "format": "json", "srlimit": 1}).json()
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
                         params={"q": q, "hl": "en-NG", "gl": "NG", "ceid": "NG:en"})
        items = ET.fromstring(r.content).findall(".//item")[:n]
        return [f"- {i.findtext('title')} ({(i.findtext('pubDate') or '')[:16]})" for i in items]
    except Exception:
        return []


def pick_topic(kind, state):
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
        else "7-minute documentary-style YouTube video"
    avoid = "; ".join(state["used"][-200:]) or "none yet"
    for _ in range(3):
        data = gemini(f"""You plan videos for a YouTube channel. Niche: {CFG['niche']}
Propose 10 fresh topics for a {fmt}. Each must be about a real subject that has its own
English Wikipedia article (company, person's business empire, industry, economic event, policy, market).
Mix famous topics with surprising lesser-known ones. Prefer topics people actually search for.
Do NOT repeat or closely overlap these already-made topics: {avoid}
Return JSON: {{"topics":[{{"topic":"short working title","wikipedia_title":"exact article title","angle":"the hook/story angle"}}]}}""")
        cands = data.get("topics", [])
        random.shuffle(cands)
        for c in cands:
            if norm(c.get("topic", "")) in used or norm(c.get("wikipedia_title", "")) in used:
                continue
            title, text = wiki_article(c.get("wikipedia_title") or c["topic"])
            if title and norm(title) not in used and len(text) > 2500:
                log("auto topic:", c["topic"], "| wiki:", title)
                return {"topic": c["topic"], "angle": c.get("angle", ""), "wiki_title": title,
                        "source": text, "key": title}
    raise RuntimeError("could not find a fresh topic")


# ---------------------------------------------------------------- script
def write_script(kind, t):
    sources = (t["source"] or "")[:14000]
    news = "\n".join(news_headlines(t["wiki_title"] or t["topic"]))
    if kind == "short":
        spec = """A YouTube Short of 120-150 words total (about 50 seconds spoken).
5-7 segments. Segment 1 is a scroll-stopping hook in under 12 words. End with a line that makes people
want to rewatch or follow. Title must be under 60 characters and end with ' #Shorts'."""
    else:
        spec = """A documentary-style YouTube video of 950-1150 words (about 7 minutes spoken).
11-15 segments. The first segment is a strong 2-3 sentence hook that creates curiosity (no 'welcome back',
no 'in this video'). Tell it as a story with tension: origins, turning points, numbers, what it means.
Close with a takeaway and a one-line call to subscribe. Title: under 70 characters, curiosity-driven, not clickbait lies."""
    prompt = f"""You are a top YouTube scriptwriter for this channel: {CFG['niche']}
Write {spec}
Topic: {t['topic']}
Angle / creator notes: {t['angle'] or 'your choice'}

STRICT FACT RULES: Use ONLY facts found in the SOURCES below. Never invent numbers, dates, quotes or events.
If a detail is not in the sources, leave it out. Recent headlines may be mentioned only as headlines say.
Write for the ear: short sentences, vivid, conversational, no lists, no markdown, no emojis, no stage directions.

For each segment give a 'visual': a 2-4 word generic STOCK FOOTAGE search phrase that fits
(e.g. 'oil refinery aerial', 'lagos traffic', 'cement factory', 'phone payment app'). Never use people's names or brand names in visuals.

Return JSON:
{{"title":"...","description":"2 short paragraphs summarising the video for YouTube SEO",
"tags":["8-15 tags"],"segments":[{{"narration":"...","visual":"..."}}]}}

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


def make_ass(segs, audio, W, H, short, path):
    size, margin, align = (96, int(H * 0.30), 2) if short else (54, 70, 2)
    words_per = 3 if short else 7
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,DejaVu Sans,{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,"
        f"{5 if short else 3},1,{align},60,60,{margin},1", "",
        "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    t0 = 0.0
    for g, (_, dur) in zip(segs, audio):
        words = g["narration"].split()
        chunks = [" ".join(words[i:i + words_per]) for i in range(0, len(words), words_per)]
        speak = dur - 0.35
        total = sum(len(c) + 3 for c in chunks) or 1
        t = t0
        for c in chunks:
            dt = speak * (len(c) + 3) / total
            txt = c.upper() if short else c
            if short:
                txt = r"{\fad(40,0)}" + txt
            lines.append(f"Dialogue: 0,{ts(t)},{ts(t + dt)},Default,,0,0,0,,{txt}")
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
    vf = (f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps=30,setsar=1")
    for i, (g, (_, dur)) in enumerate(zip(segs, audio)):
        n = max(1, math.ceil(dur / (4.0 if short else 6.0)))
        clips = get_clips(g.get("visual") or "city", n, orient, W, H, used_ids, d, i)
        if not clips:
            raise RuntimeError("no footage found")
        pd = dur / n
        for j, c in enumerate(clips):
            out = d / f"p{i:02d}_{j}.mp4"
            run(["ffmpeg", "-y", "-stream_loop", "-1", "-i", c, "-t", f"{pd:.3f}", "-vf", vf, "-an",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-r", "30", out])
            parts.append(out)
        log(f"scene {i + 1}/{len(segs)} done")
    (d / "v.txt").write_text("".join(f"file '{p.name}'\n" for p in parts))
    (d / "a.txt").write_text("".join(f"file '{w.name}'\n" for w, _ in audio))
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", d / "v.txt", "-c", "copy", d / "video.mp4"])
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", d / "a.txt", "-c", "copy", d / "audio.wav"])
    make_ass(segs, audio, W, H, short, d / "subs.ass")
    final = d / "final.mp4"
    run(["ffmpeg", "-y", "-i", str(d / "video.mp4"), "-i", str(d / "audio.wav"), "-vf", f"ass={d / 'subs.ass'}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(final)])
    log(f"rendered {final} ({duration(final):.1f}s)")
    return final


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
    desc = (script.get("description", "") + "\n\nSources: Wikipedia and public news reports."
            "\nNarration uses an AI voice." + ("\n\n#Shorts" if kind == "short" else ""))
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
    log("starting", kind, "video")
    topic = pick_topic(kind, state)
    script = write_script(kind, topic)
    log("title:", script["title"])
    d = WORK / kind
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    (d / "script.json").write_text(json.dumps(script, indent=1, ensure_ascii=False))
    video = render(kind, script, d)
    thumb = thumbnail(d / "video.mp4", script["title"], d) if kind == "long" else None
    vid = None if no_upload else upload(video, thumb, script, kind)
    state["used"].append(topic["key"])
    state["uploads"].append({"kind": kind, "title": script["title"], "id": vid,
                             "date": time.strftime("%Y-%m-%d %H:%M")})
    save_state(state)


if __name__ == "__main__":
    main()
