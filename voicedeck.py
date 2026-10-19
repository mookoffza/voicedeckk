"""VoiceDeck - real-time voice changer + soundboard.

Mic -> voice effect -> (+ soundboard sounds) -> virtual cable -> Discord / games.
Run:  pip install -r requirements.txt   then   python voicedeck.py
"""
import json, os, queue, shutil, threading, webbrowser
from math import gcd
import numpy as np
from scipy import signal
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox

try:
    import sounddevice as sd
except Exception:  # no PortAudio available
    sd = None
try:
    import soundfile as sf
except Exception:
    sf = None
try:
    import keyboard  # optional: global hotkeys that work inside games
except Exception:
    keyboard = None

SR, BLOCK = 48000, 512
HOME = os.path.join(os.path.expanduser("~"), "VoiceDeck")
SND_DIR, CFG = os.path.join(HOME, "sounds"), os.path.join(HOME, "config.json")

EFFECTS = {
    "Normal": {},
    "Deep": dict(pitch=.72),
    "Demon": dict(pitch=.55, dist=4, echo=(.09, .35)),
    "Chipmunk": dict(pitch=1.7),
    "Baby": dict(pitch=1.9, band=(300, 6000)),
    "Girl": dict(pitch=1.3),
    "Robot": dict(ring=55, pitch=.95),
    "Alien": dict(pitch=1.25, ring=14),
    "Megaphone": dict(band=(500, 3500), dist=6),
    "Telephone": dict(band=(400, 3000)),
    "Cave": dict(echo=(.14, .55)),
    "Echo": dict(echo=(.3, .45)),
}


# ----------------------------------------------------------------- DSP
class PitchShifter:
    """Two crossfaded delay taps sweeping through a ring buffer (low latency)."""
    W = 2048

    def __init__(s):
        s.buf = np.zeros(16384, np.float32); s.w = 0; s.ph = 0.0

    def process(s, x, r):
        n, N = len(x), len(s.buf)
        base = s.w + np.arange(n)
        s.buf[base % N] = x
        ph = s.ph + np.arange(n) * (1 - r) / s.W
        out = np.zeros(n, np.float32)
        for p in (ph % 1.0, (ph + .5) % 1.0):
            pos = base - (2 + p * s.W)
            i0 = np.floor(pos).astype(np.int64)
            fr = (pos - i0).astype(np.float32)
            a, b = s.buf[i0 % N], s.buf[(i0 + 1) % N]
            out += (a + (b - a) * fr) * (1 - np.abs(2 * p - 1)).astype(np.float32)
        s.ph = (s.ph + n * (1 - r) / s.W) % 1.0
        s.w = (s.w + n) % N
        return out


class Voice:
    def __init__(s):
        s.ps = PitchShifter(); s.t = 0
        s.eb = np.zeros(SR, np.float32); s.ew = 0
        s.set("Normal")

    def set(s, name):
        s.p = EFFECTS[name]; s.sos = None
        if "band" in s.p:
            s.sos = signal.butter(2, s.p["band"], "bandpass", fs=SR, output="sos")
            s.zi = np.zeros((s.sos.shape[0], 2))

    def process(s, x):
        p, n = s.p, len(x)
        if not p:
            return x
        if p.get("pitch", 1) != 1:
            x = s.ps.process(x, p["pitch"])
        if "ring" in p:
            t = (s.t + np.arange(n)) / SR
            x = x * np.sin(2 * np.pi * p["ring"] * t).astype(np.float32) * 1.4
            s.t = (s.t + n) % SR
        if "dist" in p:
            x = np.tanh(x * p["dist"]) * .8
        if s.sos is not None:
            x, s.zi = signal.sosfilt(s.sos, x, zi=s.zi)
        if "echo" in p:
            d, fb = p["echo"]; D, L = int(d * SR), len(s.eb)
            idx = (s.ew + np.arange(n)) % L
            x = x + fb * s.eb[(idx - D) % L]
            s.eb[idx] = x; s.ew = (s.ew + n) % L
        return np.clip(x, -1, 1).astype(np.float32)


# -------------------------------------------------------------- engine
def _ex(dev):
    """On Windows WASAPI let Windows convert sample rates for us."""
    try:
        api = sd.query_hostapis(sd.query_devices(dev)["hostapi"])["name"]
        return sd.WasapiSettings(auto_convert=True) if "WASAPI" in api else None
    except Exception:
        return None


class Engine:
    def __init__(s):
        s.voice = Voice(); s.lock = threading.Lock(); s.active = []; s.cache = {}
        s.vq, s.mq = queue.Queue(4), queue.Queue(4)
        s.mic_vol, s.snd_vol = 1.0, .8
        s.mute = False; s.hear_voice = False; s.hear_sounds = True
        s.streams = []; s.running = False; s.monitor_on = False

    # callbacks -------------------------------------------------------
    def _in(s, indata, frames, t, status):
        try:
            s.vq.put_nowait(s.voice.process(indata[:, 0].copy()))
        except queue.Full:
            pass

    def _mix(s, n):
        out = np.zeros((n, 2), np.float32)
        with s.lock:
            keep = []
            for pl in s.active:
                seg = pl[0][pl[1]:pl[1] + n]
                out[:len(seg)] += seg; pl[1] += n
                if pl[1] < len(pl[0]):
                    keep.append(pl)
            s.active = keep
        return out

    def _out(s, outdata, frames, t, status):
        try:
            v = s.vq.get_nowait()
        except queue.Empty:
            v = np.zeros(frames, np.float32)
        if len(v) != frames:
            v = np.resize(v, frames)
        v = np.zeros(frames, np.float32) if s.mute else v * s.mic_vol
        snd = s._mix(frames) * s.snd_vol
        out = np.clip(v[:, None] + snd, -1, 1)
        outdata[:] = out[:, :outdata.shape[1]]
        if s.monitor_on:
            m = (snd if s.hear_sounds else snd * 0) + (v[:, None] if s.hear_voice else 0)
            try:
                s.mq.put_nowait(np.clip(m, -1, 1).astype(np.float32))
            except queue.Full:
                pass

    def _mon(s, outdata, frames, t, status):
        try:
            m = s.mq.get_nowait()
            outdata[:] = np.resize(m, outdata.shape) if m.shape[0] != frames else m[:, :outdata.shape[1]]
        except queue.Empty:
            outdata.fill(0)

    # control ---------------------------------------------------------
    def start(s, mic, out, mon):
        s.stop()
        kw = dict(samplerate=SR, blocksize=BLOCK, dtype="float32")
        try:
            s.streams = [
                sd.InputStream(device=mic, channels=1, callback=s._in, extra_settings=_ex(mic), **kw),
                sd.OutputStream(device=out, channels=2, callback=s._out, extra_settings=_ex(out), **kw)]
            s.monitor_on = mon is not None
            if s.monitor_on:
                s.streams.append(sd.OutputStream(device=mon, channels=2, callback=s._mon,
                                                 extra_settings=_ex(mon), **kw))
            for st in s.streams:
                st.start()
        except Exception:
            s.stop(); raise
        s.running = True

    def stop(s):
        for st in s.streams:
            try: st.stop(); st.close()
            except Exception: pass
        s.streams = []; s.running = False
        for q in (s.vq, s.mq):
            while not q.empty():
                try: q.get_nowait()
                except queue.Empty: break

    def load(s, path):
        if path not in s.cache:
            d, rate = sf.read(path, dtype="float32", always_2d=True)
            d = np.repeat(d, 2, 1) if d.shape[1] == 1 else d[:, :2]
            if rate != SR:
                g = gcd(SR, rate)
                d = signal.resample_poly(d, SR // g, rate // g, axis=0).astype(np.float32)
            s.cache[path] = np.ascontiguousarray(d)
        return s.cache[path]

    def play(s, path):
        def go():
            try:
                d = s.load(path)
                with s.lock: s.active.append([d, 0])
            except Exception as e:
                print("Could not play", path, e)
        threading.Thread(target=go, daemon=True).start()

    def stop_sounds(s):
        with s.lock: s.active = []


# ------------------------------------------------- built-in starter sounds
def make_defaults():
    if sf is None or os.listdir(SND_DIR):
        return
    def synth(freqs, vib=0):
        t = np.arange(len(freqs)) / SR
        ph = np.cumsum(freqs * (1 + vib * np.sin(2 * np.pi * 6 * t))) / SR
        return 2 * (ph % 1) - 1
    def env(x, a=.01, r=.08):
        n = len(x); e = np.ones(n)
        A, R = int(a * SR), int(r * SR)
        e[:A] = np.linspace(0, 1, A); e[-R:] = np.linspace(1, 0, R)
        return x * e
    horn = sum(synth(np.full(int(.9 * SR), f), .01) for f in (440, 554, 659)) / 3
    trom = np.concatenate([synth(np.full(int(d * SR), f), v)
                           for f, d, v in ((233, .45, 0), (220, .45, 0), (208, .45, 0), (196, 1.2, .02))])
    n = int(.5 * SR); tt = np.arange(n) / SR
    boing = np.sin(2 * np.pi * np.cumsum(200 + 600 * np.exp(-6 * tt) * np.cos(2 * np.pi * 14 * tt)) / SR) * np.exp(-4 * tt)
    laser = np.sin(2 * np.pi * np.cumsum(np.linspace(1800, 200, int(.3 * SR))) / SR)
    for name, x in (("Air Horn", env(horn)), ("Sad Trombone", env(trom, .05, .2)),
                    ("Boing", env(boing)), ("Laser", env(laser))):
        sf.write(os.path.join(SND_DIR, name + ".wav"), (x * .6).astype(np.float32), SR, subtype="PCM_16")


# ------------------------------------------------------------------ UI
BG, PANEL, FG, ACC, PAD = "#16161d", "#22222d", "#eaeaf2", "#7c5cff", "#2f2f3f"


class App:
    def __init__(s, root):
        s.r, s.eng = root, Engine()
        root.title("VoiceDeck"); root.geometry("1000x720"); root.configure(bg=BG)
        os.makedirs(SND_DIR, exist_ok=True); make_defaults()
        s.sounds = s.load_cfg()
        s.style()
        s.build_routing(); s.build_voices(); s.build_board()
        s.refresh_pads()
        root.after(600, s.check_cable)

    def check_cable(s):
        if sd and not any("CABLE" in d["name"].upper() for d in sd.query_devices()):
            if messagebox.askyesno("One-time setup",
                    "No virtual audio cable found. Discord and games need one to hear VoiceDeck.\n\n"
                    "Open the free VB-Cable download page now? (Install it, reboot, then reopen VoiceDeck.)"):
                webbrowser.open("https://vb-audio.com/Cable/")

    def style(s):
        st = ttk.Style(); st.theme_use("clam")
        st.configure("TFrame", background=BG); st.configure("TLabel", background=BG, foreground=FG)
        st.configure("TCheckbutton", background=BG, foreground=FG)
        st.configure("TCombobox", fieldbackground=PANEL, background=PANEL, foreground=FG)

    def load_cfg(s):
        if os.path.exists(CFG):
            try: return json.load(open(CFG))["sounds"]
            except Exception: pass
        return [{"name": os.path.splitext(f)[0], "path": os.path.join(SND_DIR, f), "hotkey": ""}
                for f in sorted(os.listdir(SND_DIR))]

    def save_cfg(s):
        json.dump({"sounds": s.sounds}, open(CFG, "w"), indent=1)

    # routing ----------------------------------------------------------
    def build_routing(s):
        f = ttk.Frame(s.r); f.pack(fill="x", padx=12, pady=8)
        s.v = {k: tk.StringVar() for k in ("api", "mic", "out", "mon")}
        s.cb = {}
        labels = (("api", "Audio API"), ("mic", "Your microphone"),
                  ("out", "Send to (virtual cable)"), ("mon", "Hear it on (headphones)"))
        for i, (k, lab) in enumerate(labels):
            ttk.Label(f, text=lab).grid(row=0, column=i, sticky="w", padx=4)
            s.cb[k] = ttk.Combobox(f, textvariable=s.v[k], state="readonly", width=30)
            s.cb[k].grid(row=1, column=i, padx=4, pady=2)
        apis = [a["name"] for a in sd.query_hostapis()] if sd else []
        s.cb["api"]["values"] = apis
        pref = next((a for a in apis if "WASAPI" in a), apis[0] if apis else "")
        s.v["api"].set(pref); s.cb["api"].bind("<<ComboboxSelected>>", lambda e: s.fill_devices())
        s.fill_devices()

        g = ttk.Frame(s.r); g.pack(fill="x", padx=12)
        s.btn = tk.Button(g, text="START", bg=ACC, fg="white", bd=0, font=("Segoe UI", 12, "bold"),
                          width=10, command=s.toggle); s.btn.pack(side="left", padx=4, pady=6)
        s.status = ttk.Label(g, text="Stopped"); s.status.pack(side="left", padx=8)
        ttk.Button(g, text="Get VB-Cable (virtual mic)",
                   command=lambda: webbrowser.open("https://vb-audio.com/Cable/")).pack(side="right")
        h = ttk.Frame(s.r); h.pack(fill="x", padx=12)
        s.mute, s.hv, s.hs = tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(value=True)
        for var, text, attr in ((s.mute, "Mute mic", "mute"), (s.hv, "Hear myself", "hear_voice"),
                                (s.hs, "Hear sounds", "hear_sounds")):
            ttk.Checkbutton(h, text=text, variable=var,
                            command=lambda v=var, a=attr: setattr(s.eng, a, v.get())).pack(side="left", padx=6)
        for text, attr, val in (("Mic vol", "mic_vol", 1.0), ("Sound vol", "snd_vol", .8)):
            ttk.Label(h, text=text).pack(side="left", padx=(16, 2))
            sc = tk.Scale(h, from_=0, to=2 if attr == "mic_vol" else 1.5, resolution=.05, orient="horizontal",
                          length=130, bg=BG, fg=FG, highlightthickness=0, troughcolor=PANEL,
                          command=lambda v, a=attr: setattr(s.eng, a, float(v)))
            sc.set(val); sc.pack(side="left")

    def fill_devices(s):
        if sd is None:
            return
        apis = [a["name"] for a in sd.query_hostapis()]
        api = apis.index(s.v["api"].get())
        ins, outs = [], []
        for i, d in enumerate(sd.query_devices()):
            if d["hostapi"] != api: continue
            lab = f"{i}: {d['name']}"
            if d["max_input_channels"] > 0: ins.append(lab)
            if d["max_output_channels"] > 0: outs.append(lab)
        s.cb["mic"]["values"] = ins; s.cb["out"]["values"] = outs
        s.cb["mon"]["values"] = ["None"] + outs
        s.v["mic"].set(next((x for x in ins if "CABLE" not in x.upper()), ins[0] if ins else ""))
        cable = next((x for x in outs if "CABLE INPUT" in x.upper()), None)
        s.v["out"].set(cable or (outs[0] if outs else ""))
        s.v["mon"].set(next((x for x in outs if "CABLE" not in x.upper()), "None"))

    def toggle(s):
        if s.eng.running:
            s.eng.stop(); s.btn.config(text="START", bg=ACC); s.status.config(text="Stopped"); return
        idx = lambda k: None if s.v[k].get() in ("", "None") else int(s.v[k].get().split(":")[0])
        if "CABLE" not in s.v["out"].get().upper():
            if not messagebox.askyesno("No virtual cable selected",
                    "Discord/games can only hear VoiceDeck if 'Send to' is a virtual cable "
                    "(e.g. 'CABLE Input'). Start anyway?"):
                return
        try:
            s.eng.start(idx("mic"), idx("out"), idx("mon"))
        except Exception as e:
            messagebox.showerror("Audio error", f"{e}\n\nTry another Audio API (WASAPI/MME), close apps "
                                 "using the mic exclusively, or pick matching devices.")
            return
        s.btn.config(text="STOP", bg="#d9444f"); s.status.config(text="Live")

    # voices -----------------------------------------------------------
    def build_voices(s):
        ttk.Label(s.r, text="VOICES", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=14, pady=(10, 0))
        f = ttk.Frame(s.r); f.pack(fill="x", padx=12)
        s.vb = {}
        for i, name in enumerate(EFFECTS):
            b = tk.Button(f, text=name, bg=PAD, fg=FG, bd=0, width=12, height=2,
                          command=lambda n=name: s.pick(n)); b.grid(row=i // 6, column=i % 6, padx=3, pady=3)
            s.vb[name] = b
        s.pick("Normal")

    def pick(s, name):
        s.eng.voice.set(name)
        for n, b in s.vb.items(): b.config(bg=ACC if n == name else PAD)

    # soundboard -------------------------------------------------------
    def build_board(s):
        top = ttk.Frame(s.r); top.pack(fill="x", padx=12, pady=(12, 2))
        ttk.Label(top, text="SOUNDBOARD", font=("Segoe UI", 11, "bold")).pack(side="left")
        s.q = tk.StringVar(); s.q.trace_add("write", lambda *a: s.refresh_pads())
        tk.Entry(top, textvariable=s.q, bg=PANEL, fg=FG, insertbackground=FG, bd=0, width=28).pack(side="left", padx=10)
        ttk.Label(top, text="<- search").pack(side="left")
        ttk.Button(top, text="Stop all sounds", command=s.eng.stop_sounds).pack(side="right")
        ttk.Button(top, text="Find free sounds", command=lambda: webbrowser.open(
            "https://www.myinstants.com")).pack(side="right", padx=6)
        ttk.Button(top, text="+ Add sounds / songs", command=s.add).pack(side="right")
        wrap = ttk.Frame(s.r); wrap.pack(fill="both", expand=True, padx=12, pady=6)
        s.cv = tk.Canvas(wrap, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(wrap, command=s.cv.yview); s.cv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); s.cv.pack(side="left", fill="both", expand=True)
        s.grid = ttk.Frame(s.cv); s.cv.create_window((0, 0), window=s.grid, anchor="nw")
        s.grid.bind("<Configure>", lambda e: s.cv.configure(scrollregion=s.cv.bbox("all")))
        s.cv.bind_all("<MouseWheel>", lambda e: s.cv.yview_scroll(-1 * (e.delta // 120), "units"))

    def add(s):
        files = filedialog.askopenfilenames(title="Pick audio files",
                filetypes=[("Audio", "*.mp3 *.wav *.ogg *.flac *.m4a *.aiff"), ("All", "*.*")])
        for p in files:
            dst = os.path.join(SND_DIR, os.path.basename(p))
            try: shutil.copy(p, dst)
            except Exception: dst = p
            s.sounds.append({"name": os.path.splitext(os.path.basename(p))[0], "path": dst, "hotkey": ""})
        s.save_cfg(); s.refresh_pads()

    def refresh_pads(s):
        for w in s.grid.winfo_children(): w.destroy()
        q = s.q.get().lower() if hasattr(s, "q") else ""
        shown = [x for x in s.sounds if q in x["name"].lower()]
        for i, snd in enumerate(shown):
            txt = snd["name"][:22] + (f"\n[{snd['hotkey']}]" if snd["hotkey"] else "")
            b = tk.Button(s.grid, text=txt, bg=PAD, fg=FG, bd=0, width=16, height=3, wraplength=120,
                          activebackground=ACC, command=lambda x=snd: s.eng.play(x["path"]))
            b.grid(row=i // 6, column=i % 6, padx=4, pady=4)
            b.bind("<Button-3>", lambda e, x=snd: s.menu(e, x))
        s.bind_hotkeys()

    def menu(s, e, snd):
        m = tk.Menu(s.r, tearoff=0)
        m.add_command(label="Rename", command=lambda: s.edit(snd, "name"))
        m.add_command(label="Set hotkey (e.g. ctrl+1)", command=lambda: s.edit(snd, "hotkey"))
        m.add_command(label="Remove", command=lambda: (s.sounds.remove(snd), s.save_cfg(), s.refresh_pads()))
        m.tk_popup(e.x_root, e.y_root)

    def edit(s, snd, key):
        v = simpledialog.askstring(key.title(), f"New {key}:", initialvalue=snd[key], parent=s.r)
        if v is not None:
            snd[key] = v.strip(); s.save_cfg(); s.refresh_pads()

    def bind_hotkeys(s):
        if keyboard is None: return
        try: keyboard.unhook_all_hotkeys()
        except Exception: pass
        for x in s.sounds:
            if x["hotkey"]:
                try: keyboard.add_hotkey(x["hotkey"], lambda p=x["path"]: s.eng.play(p))
                except Exception: pass


if __name__ == "__main__":
    if sd is None or sf is None:
        raise SystemExit("Run: pip install -r requirements.txt")
    root = tk.Tk(); App(root); root.mainloop()
