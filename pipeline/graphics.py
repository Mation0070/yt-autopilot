"""Animated motion-graphics for the autopilot (free: Pillow + ffmpeg only).

Each graphic is drawn frame-by-frame with Pillow and piped into ffmpeg, giving the channel a
recognisable animated "explainer" look instead of a pure stock-footage montage.

Graphic types the scriptwriter can ask for (segment["graphic"]):
  {"type":"number",   "value":5000000000, "prefix":"$", "suffix":"", "label":"spent on ..."}
  {"type":"map",      "countries":["Taiwan","China"], "label":"..."}
  {"type":"bars",     "title":"...", "items":[{"label":"TSMC","value":61},{"label":"Samsung","value":11}], "unit":"%"}
  {"type":"timeline", "title":"...", "events":[{"year":"1987","label":"Founded"}, ...]}
  {"type":"compare",  "left":{"label":"...","value":"$12B"}, "right":{"label":"...","value":"$3B"}, "title":"..."}
  {"type":"statement","text":"one punchy sentence", "kicker":"THE CATCH"}
  {"type":"chapter",  "number":2, "title":"..."}
"""
import json, math, pathlib, random, re, subprocess

import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FPS = 30
FONT_URLS = {
    "display": "https://raw.githubusercontent.com/google/fonts/main/ofl/anton/Anton-Regular.ttf",
    "heavy": "https://raw.githubusercontent.com/google/fonts/main/ofl/poppins/Poppins-ExtraBold.ttf",
    "semi": "https://raw.githubusercontent.com/google/fonts/main/ofl/poppins/Poppins-SemiBold.ttf",
}
MAP_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
           "ne_110m_admin_0_countries.geojson")
ASSETS = pathlib.Path(__file__).resolve().parent.parent / "work" / "_assets"
DEFAULT_BRAND = {"bg": "#0A0F1C", "bg2": "#16213A", "accent": "#FFB627", "accent2": "#3DA5FF",
                 "text": "#FFFFFF", "muted": "#9AA6BF", "name": "", "land": "#1F2A44"}
_FONT_CACHE = {}


# ---------------------------------------------------------------- assets
def _download(url, path):
    if path.exists() and path.stat().st_size > 1000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path


def font_path(kind):
    """Brand fonts (Google Fonts, free/OFL). Falls back to DejaVu if the download fails."""
    try:
        return str(_download(FONT_URLS[kind], ASSETS / pathlib.Path(FONT_URLS[kind]).name))
    except Exception:
        try:
            return subprocess.run(["fc-match", "-f", "%{file}", "DejaVu Sans:bold"],
                                  capture_output=True, text=True).stdout.strip()
        except Exception:
            return "DejaVuSans-Bold.ttf"


def font(kind, size):
    key = (kind, int(size))
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(font_path(kind), int(size))
    return _FONT_CACHE[key]


def fonts_dir():
    for k in FONT_URLS:
        font_path(k)
    return str(ASSETS)


def hexrgb(h, a=255):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a)


# ---------------------------------------------------------------- easing / helpers
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(t):
    t = clamp(t)
    return 1 - (1 - t) ** 3


def ease_in_out(t):
    t = clamp(t)
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def prog(t, start, dur):
    return ease_out((t - start) / dur) if dur > 0 else 1.0


def fmt_number(v, decimals=None):
    """5_000_000_000 -> '5 billion' style short form used in counters."""
    a = abs(v)
    for div, word in ((1e12, " trillion"), (1e9, " billion"), (1e6, " million")):
        if a >= div:
            x = v / div
            d = decimals if decimals is not None else (0 if x >= 100 else 1)
            s = f"{x:,.{d}f}"
            if s.endswith(".0"):
                s = s[:-2]
            return s + word
    d = decimals if decimals is not None else (0 if a >= 100 or float(v).is_integer() else 1)
    return f"{v:,.{d}f}"


def wrap(draw, text, fnt, max_w):
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if draw.textlength(test, font=fnt) <= max_w or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def fit_font(draw, text, kind, max_w, max_lines, start, minimum=24):
    size = start
    while size > minimum:
        f = font(kind, size)
        lines = wrap(draw, text, f, max_w)
        if len(lines) <= max_lines and all(draw.textlength(l, font=f) <= max_w for l in lines):
            return f, lines
        size -= 4
    f = font(kind, minimum)
    return f, wrap(draw, text, f, max_w)[:max_lines]


def text_center(draw, cx, y, text, fnt, fill, stroke=0, stroke_fill=(0, 0, 0, 255)):
    w = draw.textlength(text, font=fnt)
    draw.text((cx - w / 2, y), text, font=fnt, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)


# ---------------------------------------------------------------- background
class Backdrop:
    """Animated brand background: deep gradient, slow drifting light, faint grid."""

    def __init__(self, W, H, brand, seed=0):
        self.W, self.H, self.b = W, H, brand
        rnd = random.Random(seed)
        base = Image.new("RGB", (W, H), hexrgb(brand["bg"])[:3])
        g = Image.linear_gradient("L").rotate(rnd.choice([0, 90, 180, 270])).resize((W, H))
        top = Image.new("RGB", (W, H), hexrgb(brand["bg2"])[:3])
        base = Image.composite(top, base, g.point(lambda v: int(v * 0.85)))
        grid = ImageDraw.Draw(base, "RGBA")
        step = max(W, H) // 18
        for x in range(0, W, step):
            grid.line([(x, 0), (x, H)], fill=(255, 255, 255, 10), width=1)
        for y in range(0, H, step):
            grid.line([(0, y), (W, y)], fill=(255, 255, 255, 10), width=1)
        self.base = base
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        r = int(min(W, H) * 0.55)
        ImageDraw.Draw(glow).ellipse([W / 2 - r, H / 2 - r, W / 2 + r, H / 2 + r], fill=hexrgb(brand["accent2"], 60))
        self.glow = glow.filter(ImageFilter.GaussianBlur(r // 3))
        self.phase = rnd.random() * 6.28

    def frame(self, t):
        img = self.base.copy()
        dx = int(math.sin(t * 0.35 + self.phase) * self.W * 0.18)
        dy = int(math.cos(t * 0.27 + self.phase) * self.H * 0.12)
        img.paste(self.glow, (dx, dy), self.glow)
        return img.convert("RGBA")


def vignette(img):
    W, H = img.size
    v = Image.new("L", (W, H), 0)
    ImageDraw.Draw(v).ellipse([-W * 0.25, -H * 0.25, W * 1.25, H * 1.25], fill=255)
    v = v.filter(ImageFilter.GaussianBlur(min(W, H) // 6))
    dark = Image.new("RGBA", (W, H), (0, 0, 0, 150))
    img.paste(dark, (0, 0), Image.eval(v, lambda x: 255 - x))
    return img


# ---------------------------------------------------------------- world map
_MAP = None


def _load_map():
    global _MAP
    if _MAP is None:
        p = _download(MAP_URL, ASSETS / "countries.geojson")
        _MAP = json.loads(p.read_text(encoding="utf-8"))["features"]
    return _MAP


ALIASES = {"usa": "united states of america", "us": "united states of america", "united states": "united states of america",
           "america": "united states of america", "uk": "united kingdom", "britain": "united kingdom",
           "great britain": "united kingdom", "england": "united kingdom", "russia": "russia",
           "south korea": "south korea", "korea": "south korea", "north korea": "north korea",
           "uae": "united arab emirates", "drc": "democratic republic of the congo",
           "dr congo": "democratic republic of the congo", "congo": "republic of the congo",
           "ivory coast": "côte d'ivoire", "cote d'ivoire": "côte d'ivoire", "czech republic": "czechia",
           "holland": "netherlands", "the netherlands": "netherlands", "eswatini": "eswatini",
           "swaziland": "eswatini", "burma": "myanmar", "turkey": "turkey", "türkiye": "turkey"}


def _match(feature, wanted):
    p = feature["properties"]
    names = {str(p.get(k, "")).lower() for k in ("NAME", "NAME_LONG", "ADMIN", "NAME_EN", "SOVEREIGNT", "FORMAL_EN", "ISO_A3", "ADM0_A3")}
    return any(w in names for w in wanted)


def _rings(geom):
    if geom["type"] == "Polygon":
        return [geom["coordinates"][0]]
    if geom["type"] == "MultiPolygon":
        return [poly[0] for poly in geom["coordinates"]]
    return []


def map_shapes(wanted):
    """Country outlines as lon/lat rings, plus which are highlighted and the box to zoom to."""
    shapes, hits, box = [], [], [999, 999, -999, -999]
    for f in _load_map():
        hit = _match(f, wanted)
        rings = [r for r in _rings(f["geometry"]) if len(r) >= 3]
        for ring in rings:
            shapes.append((hit, ring))
        if hit:
            hits.append(f["properties"].get("NAME"))
            # frame on the largest piece (ignores far-flung islands/overseas territories)
            big = max(rings, key=lambda r: (max(p[0] for p in r) - min(p[0] for p in r)) *
                                           (max(p[1] for p in r) - min(p[1] for p in r)))
            xs, ys = [p[0] for p in big], [p[1] for p in big]
            box = [min(box[0], min(xs)), min(box[1], min(ys)), max(box[2], max(xs)), max(box[3], max(ys))]
    return shapes, hits, (box if hits else None)


# ---------------------------------------------------------------- graphic renderers
class G:
    def __init__(self, W, H, brand, spec, dur, seed=0):
        self.W, self.H, self.b, self.s, self.dur = W, H, brand, spec, dur
        self.short = H > W
        self.bg = Backdrop(W, H, brand, seed)
        self.u = min(W, H) / 1080  # unit scale
        self.cx = W / 2

    def col(self, k, a=255):
        return hexrgb(self.b[k], a)

    def brand_tag(self, dr, t):
        name = self.b.get("name")
        if not name:
            return
        f = font("semi", 26 * self.u)
        a = int(200 * prog(t, 0.2, 0.6))
        x, y = 48 * self.u, (self.H - 70 * self.u) if not self.short else 120 * self.u
        dr.rectangle([x, y + 6 * self.u, x + 8 * self.u, y + 34 * self.u], fill=self.col("accent", a))
        dr.text((x + 20 * self.u, y), name.upper(), font=f, fill=self.col("muted", a))

    def label(self, dr, text, y, t, start=0.5, size=54, maxw=0.82, lines=2, kind="semi", key="text"):
        if not text:
            return y
        f, ls = fit_font(dr, text, kind, self.W * maxw, lines, int(size * self.u))
        a = int(255 * prog(t, start, 0.5))
        off = (1 - prog(t, start, 0.5)) * 30 * self.u
        for ln in ls:
            text_center(dr, self.cx, y + off, ln, f, self.col(key, a))
            y += f.size * 1.25
        return y

    def frame(self, t):
        img = self.bg.frame(t)
        dr = ImageDraw.Draw(img, "RGBA")
        self.draw(img, dr, t)
        self.brand_tag(dr, t)
        # gentle fade in/out at the edges of the graphic
        fade = min(prog(t, 0, 0.25), clamp((self.dur - t) / 0.25))
        if fade < 1:
            img = Image.blend(Image.new("RGBA", img.size, (0, 0, 0, 255)), img, fade)
        return img

    def draw(self, img, dr, t):
        raise NotImplementedError


class Number(G):
    def draw(self, img, dr, t):
        s = self.s
        v = float(s.get("value", 0))
        k = ease_in_out(clamp(t / max(1.2, self.dur * 0.45)))
        txt = f"{s.get('prefix', '')}{fmt_number(v * k, s.get('decimals'))}{s.get('suffix', '')}"
        final = f"{s.get('prefix', '')}{fmt_number(v, s.get('decimals'))}{s.get('suffix', '')}"
        f, _ = fit_font(dr, final, "display", self.W * 0.88, 1, int((260 if not self.short else 220) * self.u), 60)
        y = self.H * (0.30 if not self.short else 0.34)
        pop = 1 + 0.06 * math.sin(clamp((t - self.dur * 0.45) / 0.3) * math.pi)
        f2 = font("display", f.size * pop) if pop > 1.001 else f
        text_center(dr, self.cx, y - (f2.size - f.size) / 2, txt, f2, self.col("accent"))
        # underline sweep
        w = dr.textlength(final, font=f) * prog(t, 0.3, 0.9)
        yy = y + f.size * 1.18
        dr.rounded_rectangle([self.cx - w / 2, yy, self.cx + w / 2, yy + 10 * self.u], radius=5, fill=self.col("accent2"))
        self.label(dr, s.get("label", ""), yy + 50 * self.u, t, 0.6, 58)


class Statement(G):
    def draw(self, img, dr, t):
        s = self.s
        kick = str(s.get("kicker", "")).upper()
        y = self.H * (0.30 if not self.short else 0.32)
        if kick:
            f = font("heavy", 40 * self.u)
            w = dr.textlength(kick, font=f) + 50 * self.u
            a = int(255 * prog(t, 0.1, 0.4))
            dr.rounded_rectangle([self.cx - w / 2, y, self.cx + w / 2, y + 66 * self.u], radius=int(10 * self.u),
                                 fill=self.col("accent", a))
            text_center(dr, self.cx, y + 6 * self.u, kick, f, self.col("bg", a))
            y += 110 * self.u
        f, lines = fit_font(dr, s.get("text", ""), "heavy", self.W * 0.84, 4 if not self.short else 6,
                            int(84 * self.u), 40)
        words = sum(len(l.split()) for l in lines)
        shown = int(words * clamp((t - 0.3) / max(0.8, self.dur * 0.5))) + 1
        n = 0
        for ln in lines:
            ws = ln.split()
            vis = ws[:max(0, shown - n)]
            n += len(ws)
            full_w = dr.textlength(ln, font=f)
            x = self.cx - full_w / 2
            for w in vis:
                dr.text((x, y), w, font=f, fill=self.col("text"))
                x += dr.textlength(w + " ", font=f)
            y += f.size * 1.22


class Chapter(G):
    def draw(self, img, dr, t):
        s = self.s
        num = s.get("number")
        a = int(255 * prog(t, 0.0, 0.5))
        y = self.H * 0.36
        if num:
            f = font("semi", 44 * self.u)
            text_center(dr, self.cx, y, f"CHAPTER {num}", f, self.col("accent", a))
            y += 80 * self.u
        f, lines = fit_font(dr, str(s.get("title", "")).upper(), "display", self.W * 0.86, 2, int(130 * self.u), 50)
        off = (1 - prog(t, 0.15, 0.6)) * 50 * self.u
        for ln in lines:
            text_center(dr, self.cx, y + off, ln, f, self.col("text", int(255 * prog(t, 0.15, 0.6))))
            y += f.size * 1.1
        w = self.W * 0.3 * prog(t, 0.4, 0.8)
        dr.rectangle([self.cx - w / 2, y + 30 * self.u, self.cx + w / 2, y + 38 * self.u], fill=self.col("accent"))


class Bars(G):
    def draw(self, img, dr, t):
        s = self.s
        items = [i for i in s.get("items", []) if isinstance(i, dict)][:6]
        vals = [float(i.get("value", 0) or 0) for i in items]
        mx = max(vals + [1e-9])
        unit = s.get("unit", "")
        prefix = s.get("prefix", "")
        top = self.label(dr, s.get("title", ""), self.H * (0.12 if not self.short else 0.22), t, 0.1, 64,
                         lines=2, kind="heavy") + (30 if not self.short else 110) * self.u
        bottom = self.H * (0.88 if not self.short else 0.78)
        n = max(1, len(items))
        gap = (bottom - top) / n
        bh = min(gap * 0.55, 90 * self.u)
        x0 = self.W * (0.30 if not self.short else 0.08)
        maxw = self.W * (0.55 if not self.short else 0.70)
        fl = font("semi", (40 if not self.short else 38) * self.u)
        fv = font("heavy", 44 * self.u)
        for k, (it, v) in enumerate(zip(items, vals)):
            p = prog(t, 0.4 + k * 0.25, 0.9)
            y = top + k * gap
            if not self.short:
                lab = str(it.get("label", ""))[:28]
                dr.text((x0 - 30 * self.u - dr.textlength(lab, font=fl), y + bh / 2 - fl.size * 0.6), lab, font=fl,
                        fill=self.col("text", int(255 * prog(t, 0.2 + k * 0.25, 0.4))))
            else:
                dr.text((x0, y - fl.size * 1.3), str(it.get("label", ""))[:30], font=fl,
                        fill=self.col("text", int(255 * prog(t, 0.2 + k * 0.25, 0.4))))
            w = max(6 * self.u, maxw * (v / mx) * p)
            colr = self.col("accent") if v == mx else self.col("accent2")
            dr.rounded_rectangle([x0, y, x0 + w, y + bh], radius=int(bh / 4), fill=colr)
            if p > 0.05:
                txt = f"{prefix}{fmt_number(v * p, 0 if float(v).is_integer() and v < 1e6 else None)}{unit}"
                dr.text((x0 + w + 20 * self.u, y + bh / 2 - fv.size * 0.62), txt, font=fv, fill=self.col("text", int(255 * p)))


class Timeline(G):
    def draw(self, img, dr, t):
        s = self.s
        ev = [e for e in s.get("events", []) if isinstance(e, dict)][:5]
        n = max(1, len(ev))
        self.label(dr, s.get("title", ""), self.H * (0.12 if not self.short else 0.16), t, 0.1, 64, kind="heavy")
        fy, fl = font("display", (78 if not self.short else 70) * self.u), font("semi", 34 * self.u)
        if not self.short:
            y = self.H * 0.55
            x0, x1 = self.W * 0.1, self.W * 0.9
            p = prog(t, 0.3, max(1.0, self.dur * 0.6))
            dr.line([(x0, y), (x0 + (x1 - x0) * p, y)], fill=self.col("muted", 180), width=int(6 * self.u))
            for k, e in enumerate(ev):
                x = x0 + (x1 - x0) * (k + 0.5) / n
                a = prog(t, 0.3 + (k + 0.5) / n * max(1.0, self.dur * 0.6), 0.4)
                if a <= 0:
                    continue
                r = 16 * self.u * a
                dr.ellipse([x - r, y - r, x + r, y + r], fill=self.col("accent"))
                text_center(dr, x, y - 130 * self.u, str(e.get("year", ""))[:9], fy, self.col("accent", int(255 * a)))
                ly = y + 40 * self.u
                for ln in wrap(dr, str(e.get("label", ""))[:60], fl, (x1 - x0) / n * 0.92)[:3]:
                    text_center(dr, x, ly, ln, fl, self.col("text", int(255 * a)))
                    ly += fl.size * 1.25
        else:
            x = self.W * 0.16
            y0, y1 = self.H * 0.28, self.H * 0.82
            p = prog(t, 0.3, max(1.0, self.dur * 0.6))
            dr.line([(x, y0), (x, y0 + (y1 - y0) * p)], fill=self.col("muted", 180), width=int(6 * self.u))
            for k, e in enumerate(ev):
                y = y0 + (y1 - y0) * (k + 0.3) / n
                a = prog(t, 0.3 + (k + 0.3) / n * max(1.0, self.dur * 0.6), 0.4)
                if a <= 0:
                    continue
                r = 16 * self.u * a
                dr.ellipse([x - r, y - r, x + r, y + r], fill=self.col("accent"))
                dr.text((x + 50 * self.u, y - fy.size * 0.7), str(e.get("year", ""))[:9], font=fy, fill=self.col("accent", int(255 * a)))
                ly = y + fy.size * 0.45
                for ln in wrap(dr, str(e.get("label", ""))[:60], fl, self.W * 0.7)[:2]:
                    dr.text((x + 50 * self.u, ly), ln, font=fl, fill=self.col("text", int(255 * a)))
                    ly += fl.size * 1.2


class Compare(G):
    def draw(self, img, dr, t):
        s = self.s
        self.label(dr, s.get("title", ""), self.H * (0.12 if not self.short else 0.18), t, 0.1, 64, kind="heavy")
        sides = [s.get("left") or {}, s.get("right") or {}]
        for k, side in enumerate(sides):
            p = prog(t, 0.4 + k * 0.5, 0.6)
            if self.short:
                cx, cy, bw, bh = self.cx, self.H * (0.42 + k * 0.25), self.W * 0.8, self.H * 0.19
            else:
                cx, cy, bw, bh = self.W * (0.28 + k * 0.44), self.H * 0.56, self.W * 0.38, self.H * 0.46
            off = (1 - p) * 60 * self.u
            box = [cx - bw / 2, cy - bh / 2 + off, cx + bw / 2, cy + bh / 2 + off]
            dr.rounded_rectangle(box, radius=int(24 * self.u), fill=self.col("bg2", int(230 * p)),
                                 outline=self.col("accent" if k == 0 else "accent2", int(255 * p)), width=int(5 * self.u))
            fv, _ = fit_font(dr, str(side.get("value", "")), "display", bw * 0.85, 1, int(150 * self.u), 40)
            fl, ls = fit_font(dr, str(side.get("label", "")), "semi", bw * 0.85, 2, int(40 * self.u), 22)
            content = fv.size * 1.25 + len(ls) * fl.size * 1.2
            vy = (box[1] + box[3]) / 2 - content / 2 - fv.size * 0.1
            text_center(dr, cx, vy, str(side.get("value", "")), fv,
                        self.col("accent" if k == 0 else "accent2", int(255 * p)))
            ly = vy + fv.size * 1.25
            for ln in ls:
                text_center(dr, cx, ly, ln, fl, self.col("text", int(255 * p)))
                ly += fl.size * 1.2
        if not self.short:
            text_center(dr, self.cx, self.H * 0.52, "VS", font("display", 70 * self.u), self.col("muted", int(255 * prog(t, 0.9, 0.4))))


class Map(G):
    """World map that flies in to the highlighted countries (vector-drawn every frame, so it stays sharp)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        wanted = []
        for c in self.s.get("countries", [])[:6]:
            c = str(c).strip().lower()
            wanted.append(ALIASES.get(c, c))
        self.shapes, self.hits, box = map_shapes(wanted)
        if not self.hits:
            raise ValueError("no countries matched")
        self.bboxes = [(min(p[0] for p in r), min(p[1] for p in r), max(p[0] for p in r), max(p[1] for p in r))
                       for _, r in self.shapes]
        x0, y0, x1, y1 = box
        aspect = self.W / self.H
        w = max((x1 - x0) * 2.2, (y1 - y0) * 2.2 * aspect, 28 if aspect >= 1 else 16)
        self.end = ((x0 + x1) / 2, (y0 + y1) / 2, min(w, 360))
        self.start = ((x0 + x1) / 2 * 0.3, 15, 360 if aspect >= 1 else 150)

    def view(self, t):
        k = ease_in_out(clamp((t - 0.2) / max(1.5, self.dur * 0.55)))
        (sx, sy, sw), (ex, ey, ew) = self.start, self.end
        return sx + (ex - sx) * k, sy + (ey - sy) * k, sw * (ew / sw) ** k, k

    def draw(self, img, dr, t):
        cx, cy, w, k = self.view(t)
        h = w * self.H / self.W
        sc = self.W / w
        X = lambda lon: (lon - (cx - w / 2)) * sc
        Y = lambda lat: ((cy + h / 2) - lat) * sc
        lo0, lo1, la0, la1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
        out = self.col("bg")
        for (hit, ring), (bx0, by0, bx1, by1) in zip(self.shapes, self.bboxes):
            if bx1 < lo0 or bx0 > lo1 or by1 < la0 or by0 > la1:
                continue
            pts = [(X(x), Y(y)) for x, y in ring]
            dr.polygon(pts, fill=self.col("accent") if hit else self.col("land"), outline=out)
        ex, ey, _ = self.end
        if k > 0.85:
            pulse = 0.5 + 0.5 * math.sin(t * 4)
            r = (40 + 14 * pulse) * self.u
            px, py = X(ex), Y(ey)
            dr.ellipse([px - r, py - r, px + r, py + r], outline=self.col("text", int(220 * (k - 0.85) / 0.15)),
                       width=int(4 * self.u))
        names = self.s.get("label") or " · ".join(self.hits[:3])
        f, lines = fit_font(dr, str(names).upper(), "heavy", self.W * 0.8, 2, int(60 * self.u), 28)
        y = self.H * (0.80 if not self.short else 0.70)
        a = int(255 * prog(t, 0.8, 0.5))
        bw = max(dr.textlength(l, font=f) for l in lines) + 60 * self.u
        bh = len(lines) * f.size * 1.2 + 30 * self.u
        dr.rounded_rectangle([self.cx - bw / 2, y - 15 * self.u, self.cx + bw / 2, y - 15 * self.u + bh],
                             radius=int(14 * self.u), fill=self.col("bg", int(a * 0.85)))
        for ln in lines:
            text_center(dr, self.cx, y, ln, f, self.col("accent", a))
            y += f.size * 1.2


TYPES = {"number": Number, "statement": Statement, "chapter": Chapter, "bars": Bars,
         "timeline": Timeline, "compare": Compare, "map": Map}


# ---------------------------------------------------------------- validation + render
def _num(x):
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r"-?\d[\d,]*\.?\d*", str(x or ""))
    if not m:
        raise ValueError("no number")
    v = float(m.group(0).replace(",", ""))
    s = str(x).lower()
    for word, mul in (("trillion", 1e12), ("billion", 1e9), ("million", 1e6), ("thousand", 1e3)):
        if word in s:
            v *= mul
    return v


def validate(spec):
    """Return a cleaned spec, or None if the AI's graphic is unusable (we then fall back to footage)."""
    if not isinstance(spec, dict):
        return None
    t = str(spec.get("type", "")).lower().strip()
    if t not in TYPES:
        return None
    s = dict(spec, type=t)
    try:
        if t == "number":
            s["value"] = _num(s.get("value"))
            if s["value"] == 0:
                return None
        elif t == "bars":
            items = []
            for i in s.get("items", []):
                items.append({"label": str(i.get("label", ""))[:30], "value": _num(i.get("value"))})
            if len(items) < 2:
                return None
            s["items"] = items[:6]
        elif t == "timeline":
            ev = [e for e in s.get("events", []) if isinstance(e, dict) and e.get("year")]
            if len(ev) < 2:
                return None
            s["events"] = ev[:5]
        elif t == "compare":
            if not (isinstance(s.get("left"), dict) and isinstance(s.get("right"), dict)):
                return None
        elif t == "map":
            if not s.get("countries"):
                return None
            if isinstance(s["countries"], str):
                s["countries"] = [s["countries"]]
        elif t == "statement":
            if not str(s.get("text", "")).strip():
                return None
            s["text"] = str(s["text"])[:160]
        elif t == "chapter":
            if not str(s.get("title", "")).strip():
                return None
    except Exception:
        return None
    return s


def render_graphic(spec, dur, W, H, brand, out, seed=0):
    """Draw one animated graphic to an mp4 of exactly `dur` seconds. Raises on failure."""
    b = {**DEFAULT_BRAND, **(brand or {})}
    g = TYPES[spec["type"]](W, H, b, spec, dur, seed)
    n = max(1, int(round(dur * FPS)))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-r", str(FPS), str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        last = None
        for i in range(n):
            t = i / FPS
            # draw at 15fps and repeat frames: halves the CPU time, still smooth for motion graphics
            if i % 2 == 0 or last is None:
                last = g.frame(t).convert("RGB").tobytes()
            p.stdin.write(last)
        p.stdin.close()
    except BrokenPipeError:
        pass
    if p.wait() != 0:
        raise RuntimeError("graphic render failed: " + p.stderr.read().decode()[-500:])
    return out
