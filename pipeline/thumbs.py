"""Click-worthy branded thumbnails (1280x720): bold 2-4 word text panel + punchy photo, one accent word."""
import re

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from graphics import DEFAULT_BRAND, font, hexrgb


def make_thumbnail(frame_path, text, out, brand=None, accent_word=None, badge=None):
    b = {**DEFAULT_BRAND, **(brand or {})}
    W, H = 1280, 720
    photo = Image.open(frame_path).convert("RGB")
    photo = ImageOps.fit(photo, (W, H), Image.LANCZOS)
    photo = ImageEnhance.Contrast(photo).enhance(1.25)
    photo = ImageEnhance.Color(photo).enhance(1.3)
    photo = ImageEnhance.Brightness(photo).enhance(0.9)
    img = photo.convert("RGBA")

    # left text panel with a diagonal edge, fading into the photo
    panel = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    pd = ImageDraw.Draw(panel)
    edge = int(W * 0.60)
    pd.polygon([(0, 0), (edge, 0), (edge - 140, H), (0, H)], fill=hexrgb(b["bg"], 240))
    panel = panel.filter(ImageFilter.GaussianBlur(3))
    img.alpha_composite(panel)
    dr = ImageDraw.Draw(img)
    dr.line([(edge, 0), (edge - 140, H)], fill=hexrgb(b["accent"]), width=10)

    words = re.sub(r"#\w+", "", str(text or "")).upper().split()[:5] or ["WATCH"]
    acc = (accent_word or "").upper().strip()
    if not acc:  # highlight the number/money word if there is one, else the last word
        acc = next((w for w in words if re.search(r"\d|\$|%", w)), words[-1])
    maxw = edge - 150
    # pick the largest size where the text fits in <= 3 lines
    for size in range(210, 60, -6):
        f = font("display", size)
        lines, cur = [], []
        for w in words:
            test = " ".join(cur + [w])
            if dr.textlength(test, font=f) <= maxw or not cur:
                cur.append(w)
            else:
                lines.append(cur)
                cur = [w]
        lines.append(cur)
        if len(lines) <= 3 and all(dr.textlength(" ".join(l), font=f) <= maxw for l in lines):
            break
    hot = set()
    for i, w in enumerate(words):
        if acc and acc in w:
            hot.add(i)
            if re.search(r"\d", w) and i + 1 < len(words) and re.match(r"(TRILLION|BILLION|MILLION|PERCENT)", words[i + 1]):
                hot.add(i + 1)
    if not hot:  # accent word not found as written (e.g. "$15.3T" vs "$15.3 TRILLION"): highlight the number
        for i, w in enumerate(words):
            if re.search(r"\d|\$|%", w):
                hot.add(i)
                if i + 1 < len(words) and re.match(r"(TRILLION|BILLION|MILLION|PERCENT)", words[i + 1]):
                    hot.add(i + 1)
                break
    lh = int(f.size * 1.02)
    y = (H - lh * len(lines)) // 2 - 10
    idx = 0
    for ln in lines:
        x = 60
        for w in ln:
            col = hexrgb(b["accent"]) if idx in hot else (255, 255, 255, 255)
            idx += 1
            dr.text((x, y), w, font=f, fill=col, stroke_width=4, stroke_fill=(0, 0, 0, 255))
            x += dr.textlength(w + " ", font=f)
        y += lh

    if badge:
        fb = font("heavy", 44)
        tw = dr.textlength(str(badge), font=fb)
        bx, by = W - tw - 90, 40
        dr.rounded_rectangle([bx, by, bx + tw + 50, by + 76], radius=16, fill=hexrgb(b["accent"]))
        dr.text((bx + 25, by + 8), str(badge), font=fb, fill=hexrgb(b["bg"]))
    if b.get("name"):
        fn = font("semi", 26)
        dr.rectangle([60, H - 62, 68, H - 30], fill=hexrgb(b["accent"]))
        dr.text((80, H - 66), b["name"].upper(), font=fn, fill=hexrgb(b["muted"]))
    img.convert("RGB").save(out, quality=92)
    return out
