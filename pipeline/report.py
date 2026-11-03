"""Weekly performance report: pulls YouTube stats, asks the AI what's working, saves learnings
that the video generator uses for future topics, titles and hooks.

Usage: python pipeline/report.py "<Channel name>"   (CONFIG / STATE env pick the channel)
Writes: reports/<slug>-<date>.md  and  report.md (for the GitHub issue), updates STATE["insights"].
"""
import datetime as dt
import json
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from main import ROOT, gemini, load_state, log, save_state  # noqa: E402


def creds():
    from google.oauth2.credentials import Credentials
    return Credentials(None, refresh_token=os.environ["YT_REFRESH_TOKEN"], client_id=os.environ["YT_CLIENT_ID"],
                       client_secret=os.environ["YT_CLIENT_SECRET"], token_uri="https://oauth2.googleapis.com/token")


def fmt_dur(s):
    s = int(s or 0)
    return f"{s // 60}:{s % 60:02d}"


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "Channel"
    from googleapiclient.discovery import build
    c = creds()
    yt = build("youtube", "v3", credentials=c, cache_discovery=False)
    state = load_state()
    ups = [u for u in state.get("uploads", []) if u.get("id")]
    kinds = {u["id"]: u["kind"] for u in ups}

    ch = yt.channels().list(part="statistics,snippet", mine=True).execute()["items"][0]
    cs = ch["statistics"]

    # basic stats for every uploaded video (works with the normal upload permission)
    rows = {}
    ids = list(kinds)
    for i in range(0, len(ids), 50):
        for v in yt.videos().list(part="statistics,snippet", id=",".join(ids[i:i + 50])).execute().get("items", []):
            st = v["statistics"]
            rows[v["id"]] = {"title": v["snippet"]["title"], "kind": kinds.get(v["id"], "?"),
                             "published": v["snippet"]["publishedAt"][:10],
                             "views": int(st.get("viewCount", 0)), "likes": int(st.get("likeCount", 0)),
                             "comments": int(st.get("commentCount", 0))}

    # watch-time metrics (needs the yt-analytics.readonly permission)
    analytics_ok = False
    try:
        ya = build("youtubeAnalytics", "v2", credentials=c, cache_discovery=False)
        end = dt.date.today()
        start = end - dt.timedelta(days=28)
        r = ya.reports().query(ids="channel==MINE", startDate=str(start), endDate=str(end),
                               metrics="views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,subscribersGained",
                               dimensions="video", sort="-views", maxResults=200).execute()
        for vid, views28, mins, avd, avp, subs in r.get("rows", []):
            if vid in rows:
                rows[vid].update(views_28d=int(views28), minutes=round(mins), avg_view_duration=int(avd),
                                 avg_view_pct=round(avp, 1), subs_gained=int(subs))
        analytics_ok = True
    except Exception as e:
        log("analytics not available (re-authorise with yt-analytics.readonly to get watch time):", str(e)[:200])

    vids = sorted(rows.values(), key=lambda x: x["views"], reverse=True)
    if not vids:
        log("no videos yet"); return

    # AI analysis -> learnings fed back into the generator
    data = json.dumps(vids[:60], ensure_ascii=False)
    try:
        ai = gemini(f"""You are a YouTube growth strategist reviewing the channel "{name}".
Channel totals: {cs.get('subscriberCount')} subscribers, {cs.get('viewCount')} views, {cs.get('videoCount')} videos.
Per-video data (views; if present: views_28d, avg_view_duration seconds, avg_view_pct = % of video watched,
subs_gained): {data}
Analyse what is working and what is not: which topics, title styles and formats (short vs long) get views,
retention and subscribers. Be specific and practical; small samples are noisy, say so when relevant.
Return JSON: {{"summary":"3-5 sentence plain-English summary for the owner",
"learnings":["5-8 short concrete rules the AI scriptwriter should follow for future topics, titles and hooks"],
"double_down":["topic ideas similar to winners"],"avoid":["topics/styles that underperformed"]}}""", temperature=0.4)
    except Exception as e:
        log("analysis failed:", e)
        ai = {"summary": "AI analysis unavailable this week.", "learnings": [], "double_down": [], "avoid": []}

    # self-tuning: adjust long-video length from retention (bounded, one step per week)
    tune_note = ""
    from main import CFG
    longs = [v for v in vids if v["kind"] == "long" and v.get("avg_view_pct") is not None and v["views"] >= 20]
    if len(longs) >= 3:
        pct = sum(v["avg_view_pct"] for v in longs) / len(longs)
        cur = int((state.get("tuning") or {}).get("long_minutes") or CFG.get("long_minutes", 7))
        lo, hi = CFG.get("tune_min_minutes", 5), CFG.get("tune_max_minutes", 15)
        new = cur - 1 if pct < 30 else cur + 1 if pct > 50 else cur
        new = max(lo, min(hi, new))
        state["tuning"] = {**(state.get("tuning") or {}), "long_minutes": new}
        tune_note = (f"Long videos were watched {pct:.0f}% on average -> length "
                     f"{'kept at' if new == cur else 'changed to'} {new} min.")
    state["insights"] = {"date": str(dt.date.today()), "learnings": ai.get("learnings", [])[:8],
                         "double_down": ai.get("double_down", [])[:8], "avoid": ai.get("avoid", [])[:8]}
    save_state(state)

    # markdown report
    L = [f"# {name} - weekly report ({dt.date.today()})", "",
         f"**Subscribers:** {cs.get('subscriberCount')} | **Total views:** {cs.get('viewCount')} | "
         f"**Videos:** {cs.get('videoCount')}", "", "## Summary", ai.get("summary", ""), "",
         "## Top videos", "", "| Video | Type | Views | Avg watched | % watched | Subs |", "|---|---|---|---|---|---|"]
    for v in vids[:10]:
        L.append(f"| {v['title'][:60]} | {v['kind']} | {v['views']} | {fmt_dur(v.get('avg_view_duration'))} | "
                 f"{v.get('avg_view_pct', '-')} | {v.get('subs_gained', '-')} |")
    for head, key in (("What the AI will do more of", "double_down"), ("What it will avoid", "avoid"),
                      ("New rules for scripts & titles", "learnings")):
        L += ["", f"## {head}"] + [f"- {x}" for x in ai.get(key, [])]
    if tune_note:
        L += ["", "## Auto-tuning", tune_note]
    if not analytics_ok:
        L += ["", "_Watch-time data missing: the channel needs the analytics permission re-authorised._"]
    L += ["", "_Click-through rate (CTR) isn't available through YouTube's API - check it in the YouTube Studio app "
              "(Analytics > Reach)._"]
    md = "\n".join(L)
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    (out / f"{slug}-{dt.date.today()}.md").write_text(md, encoding="utf-8")
    (ROOT / "report.md").write_text(md, encoding="utf-8")
    log("report written")


if __name__ == "__main__":
    main()
