"""Original background music, generated from scratch with numpy (no licences, no Content ID claims).

Makes a seamless ~1.5-minute loop in a cinematic "documentary" style: warm detuned pad chords,
a sub-bass root, a soft heartbeat pulse and a delayed plucked arpeggio, all through a small reverb.
Every video gets a different key, tempo, progression and texture (seeded), so tracks don't repeat.
"""
import random
import wave

import numpy as np

SR = 22050
PROGRESSIONS = [  # scale degrees (minor key) - moody, "something big is happening" progressions
    [0, 5, 2, 6], [0, 3, 5, 4], [0, 5, 3, 4], [5, 3, 0, 4], [0, 2, 5, 6], [0, 6, 5, 6],
]
MINOR = [0, 2, 3, 5, 7, 8, 10]


def _midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def _triad(root_deg, key):
    notes = []
    for step in (0, 2, 4):
        d = root_deg + step
        notes.append(key + MINOR[d % 7] + 12 * (d // 7))
    return notes


def _lowpass(x, cutoff):
    a = np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x)
    from itertools import accumulate
    y[:] = np.fromiter(accumulate(x * (1 - a), lambda p, v: p * a + v), dtype=np.float32, count=len(x))
    return y


def _saw(f, t, phase=0.0):
    return 2 * ((f * t + phase) % 1.0) - 1


def _reverb(x, mix=0.35):
    out = x.copy()
    for d, g in ((0.0297, 0.78), (0.0371, 0.76), (0.0411, 0.74), (0.0437, 0.72), (0.113, 0.5), (0.171, 0.4)):
        n = int(d * SR)
        y = x.copy()
        for k in range(1, 7):  # feedback comb approximated by a few taps
            if n * k >= len(x):
                break
            y[n * k:] += x[:-n * k] * (g ** k)
        out += y * (mix / 6)
    return out


def make_loop(path, seed=None, mood="documentary"):
    rnd = random.Random(seed)
    key = rnd.choice([45, 46, 48, 50, 52, 53])  # A2..F3 minor keys
    prog = rnd.choice(PROGRESSIONS)
    bpm = rnd.choice([72, 76, 80, 84, 88])
    beat = 60 / bpm
    bars_per_chord = 2
    chord_len = beat * 4 * bars_per_chord
    total = chord_len * len(prog) * 2  # two passes of the progression
    n = int(total * SR)
    t = np.arange(n, dtype=np.float32) / SR
    pad = np.zeros(n, np.float32)
    bass = np.zeros(n, np.float32)
    arp = np.zeros(n, np.float32)
    pulse = np.zeros(n, np.float32)
    clen = int(chord_len * SR)
    env_c = np.ones(clen, np.float32)
    fade = int(0.12 * SR)
    env_c[:fade] = np.linspace(0, 1, fade)
    env_c[-fade:] = np.linspace(1, 0, fade)
    tc = np.arange(clen, dtype=np.float32) / SR
    arp_pattern = rnd.choice([[0, 1, 2, 1], [0, 2, 1, 2], [0, 1, 2, 3], [2, 1, 0, 1]])
    arp_step = beat / 2
    for i in range(len(prog) * 2):
        deg = prog[i % len(prog)]
        notes = _triad(deg, key)
        s = i * clen
        seg = np.zeros(clen, np.float32)
        for m in notes:
            f = _midi(m + 12)
            for det in (-0.12, 0.0, 0.11):
                seg += _saw(f * 2 ** (det / 12), tc, rnd.random()) * 0.06
        pad[s:s + clen] += seg * env_c
        fb = _midi(notes[0] - 12)
        bass[s:s + clen] += np.sin(2 * np.pi * fb * tc) * 0.30 * env_c
        # plucked arpeggio, one octave up
        steps = int(chord_len / arp_step)
        ar_notes = notes + [notes[0] + 12]
        for k in range(steps):
            m = ar_notes[arp_pattern[k % len(arp_pattern)]] + 24
            st = s + int(k * arp_step * SR)
            ln = int(min(1.2, chord_len) * SR)
            ln = min(ln, n - st)
            if ln <= 0:
                continue
            tt = np.arange(ln, dtype=np.float32) / SR
            vel = 0.10 if k % 2 == 0 else 0.07
            arp[st:st + ln] += (np.sin(2 * np.pi * _midi(m) * tt) + 0.3 * np.sin(4 * np.pi * _midi(m) * tt)) \
                * np.exp(-tt * 5.5) * vel
    # soft heartbeat pulse on beats 1 and 3
    kick_len = int(0.35 * SR)
    tk = np.arange(kick_len, dtype=np.float32) / SR
    kick = np.sin(2 * np.pi * (48 + 60 * np.exp(-tk * 30)) * tk) * np.exp(-tk * 9) * 0.35
    b = 0.0
    while b < total:
        st = int(b * SR)
        ln = min(kick_len, n - st)
        pulse[st:st + ln] += kick[:ln]
        b += beat * 2
    cut = rnd.choice([700, 850, 1000])
    pad = _lowpass(_lowpass(_lowpass(pad, cut), cut), cut)  # 18 dB/oct: warm, no buzz
    # slow filter-like swell on the pad
    pad *= (0.75 + 0.25 * np.sin(2 * np.pi * t / (chord_len * 2))).astype(np.float32)
    # delay on arp
    dl = int(beat * 0.75 * SR)
    arp[dl:] += arp[:-dl] * 0.35
    mix = pad + bass + arp * rnd.choice([0.8, 1.0]) + pulse * rnd.choice([0.6, 0.9])
    # seamless loop: process two copies, keep the second (reverb tail wraps around)
    two = np.concatenate([mix, mix])
    two = _reverb(two)
    loop = two[n:]
    loop = loop / (np.max(np.abs(loop)) + 1e-9) * 0.8
    pcm = (loop * 32767).astype(np.int16)
    stereo = np.stack([pcm, np.roll(pcm, int(0.012 * SR))], axis=1)  # tiny Haas widening
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(stereo.tobytes())
    return path
