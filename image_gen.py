#!/usr/bin/env python3
"""
Image Gen Local · Story Mode - one-file, fully local image generation UI (Bootstrap front end).

SDXL GGUF models run through the native Windows stable-diffusion.cpp CUDA executable.
The Python web app runs in its private Windows virtual environment. On first use the
app clones/builds the official sd.cpp CUDA image (compute capability 7.5) and downloads
RealVisXL Lightning GGUF.
  gpu       everything on the GPU
  balanced  transformer stays on the GPU while sampling, text encoder / VAE swap in and out
  partial   transformer is split block-by-block; only the blocks that fit stay on the GPU
  auto      picks the lightest of the above that fits your free VRAM, and steps up
            automatically if a run still hits out-of-memory

Models
  * Ranked public, ungated OpenRAIL safetensors models for 6 GB GPUs, fastest first.
  * SD 1.5 checkpoints for speed and SDXL / distilled SDXL checkpoints for higher detail.
  * Local .safetensors checkpoints can also be placed in ./localgen_data/models.

Run:   python image_gen_better.py

Options:
  --offline      never contact the internet (models + Bootstrap must already be downloaded)
  --reinstall    wipe and rebuild the private environment
  --setup-only   install everything, don't launch
  --port N       UI port (default 7860)
  --no-browser   don't auto-open the browser
  --theme X      default look: dark (default), light, auto. Colour themes are in the UI.

Everything lives in ./localgen_data. Delete that folder to uninstall. No telemetry, no cloud.
"""
import argparse
import base64
import binascii
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


SETUP_VERSION = "9"  # no environment rebuild required for UI/backend-only updates
ROOT = Path(__file__).resolve().parent / "localgen_data"
SDCPP_DIR = ROOT / "sd_cpp"
VENV_DIR = ROOT / "venv"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
STATIC_DIR = ROOT / "static"
STORY_REF_DIR = ROOT / "story_refs"
MARKER = ROOT / ".setup_done"
BOOTSTRAP_VERSION = "5.3.3"
BOOTSTRAP_URL = f"https://cdn.jsdelivr.net/npm/bootstrap@{BOOTSTRAP_VERSION}/dist/css/bootstrap.min.css"
TORCH_INDEXES = [
    "https://download.pytorch.org/whl/cu126",
    "https://download.pytorch.org/whl/cu124",
    "https://download.pytorch.org/whl/cu121",
]
SUPPORTED_PY = [(3, 12), (3, 11), (3, 10)]  # PyTorch wheels exist for these
PACKAGES = [
    "diffusers", "transformers", "accelerate", "safetensors", "pillow",
    "huggingface_hub", "gguf", "sentencepiece", "protobuf", "spacy",
]

# --------------------------------------------------------------------------- #
# Hugging Face authentication
# Never embed a personal token in source code. Prefer HF_TOKEN in the process
# environment, or use `hf auth login` so huggingface_hub reads its stored credential.
# Public, ungated model downloads work without any token or licence click-through.
# --------------------------------------------------------------------------- #
HF_TOKEN = ""


def resolve_hf_token() -> str:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = (os.environ.get(key) or "").strip()
        if value.startswith("hf_"):
            return value
    try:
        from huggingface_hub import get_token
        value = (get_token() or "").strip()
        if value.startswith("hf_"):
            return value
    except Exception:
        pass
    return ""


try:  # avoid Windows console encoding crashes
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# --------------------------------------------------------------------------- #
# Bootstrap CSS (downloaded once, served locally so the UI works offline)
# --------------------------------------------------------------------------- #
def ensure_bootstrap(offline: bool = False) -> bool:
    target = STATIC_DIR / "bootstrap.min.css"
    if target.exists() and target.stat().st_size > 10_000:
        return True
    if offline:
        return False
    try:
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(BOOTSTRAP_URL, headers={"User-Agent": "image-gen-local"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        if len(data) < 10_000:
            return False
        target.write_bytes(data)
        return True
    except Exception as e:
        print(f"  (couldn't download Bootstrap CSS: {e}; the page will fall back to the CDN link)")
        return False




def _load_spacy_pipeline():
    """Load spaCy's English parser when installed; return None for the offline fallback."""
    try:
        import spacy
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            return None
    except Exception:
        return None


def parse_story_scenes(story: str) -> list[str]:
    """Split explicit scene headings, otherwise split into sentence-sized story beats."""
    text = (story or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    heading = re.compile(r"(?im)^\s*(?:scene\s*#?\s*\d+\s*[:.\-]?\s*|scene\s+[ivxlcdm]+\s*[:.\-]?\s*)$")
    heading_inline = re.compile(r"(?im)^\s*scene\s*(?:#?\d+|[ivxlcdm]+)\s*[:.\-]\s*(.*)$")
    lines = text.splitlines()
    headed, current = [], []
    found_heading = False
    for line in lines:
        if heading.match(line.strip()):
            found_heading = True
            if "\n".join(current).strip():
                headed.append("\n".join(current).strip())
            current = []
            continue
        match = heading_inline.match(line)
        if match:
            found_heading = True
            if "\n".join(current).strip():
                headed.append("\n".join(current).strip())
            current = [match.group(1).strip()] if match.group(1).strip() else []
            continue
        current.append(line)
    if "\n".join(current).strip():
        headed.append("\n".join(current).strip())
    if found_heading:
        return headed[:40]

    # For unheaded stories, each sentence is a potential visual beat. This is
    # important when one paragraph contains several distinct events.
    nlp = _load_spacy_pipeline()
    if nlp:
        doc = nlp(text)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    else:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return sentences[:40]


def _clean_visual_phrase(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip(" ,.;:-")
    value = re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.I)
    return value


def simplify_scene_for_image(scene: str, nlp=None) -> str:
    """Turn narrative prose into a short, concrete image prompt.

    spaCy supplies sentence/dependency/entity analysis when its English model is
    available. A deterministic visual-language fallback keeps the feature usable
    offline and avoids requiring a cloud LLM.
    """
    raw = re.sub(r"\s+", " ", (scene or "")).strip()
    if not raw:
        return ""
    if raw.startswith(('"', "'", "“", "‘")):
        return ""

    # Dialogue is usually not drawable. Keep the surrounding visual narration.
    raw = re.sub(r'["“”][^"“”]*["“”]', " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip(" ,.;:-")
    if not raw:
        return ""

    low = raw.lower()

    # Discard dialogue-only and abstract personification. These lines do not
    # provide a stable visual subject/action for an image model.
    if scene.strip().startswith(('"', "'", "“", "‘")):
        return ""
    if re.search(r"\b(universe|darkness|void)\b", low) and re.search(
        r"wrap(?:ped|s)? around|blanket|reply|replied|wait(?:ing)?|keep you company|slept|sleeping", low
    ):
        return ""
    if re.search(r"\blight went out\b", low) or re.search(r"\buniverse (?:finally )?slept\b", low):
        return ""

    # Concrete visual transformations for common story events. These rules also
    # prevent metaphorical prose from becoming literal, confusing SDXL prompts.
    if re.search(r"\bstar\b", low) and re.search(r"flicker|shine|glow|light|twinkl|dark|universe", low):
        return "One tiny bright star shining alone in a vast, dark universe."
    if re.search(r"\bearth\b", low) and re.search(r"born|form|appear|emerge|creat", low):
        return "A new planet Earth appearing in space."
    if re.search(r"\b(tree|trees|forest|woodland)\b", low) and re.search(r"earth|grow|populate|cover|fill|spread", low):
        return "Green trees growing across the surface of planet Earth."
    if re.search(r"\b(sun|sunrise|sunset)\b", low):
        return "The bright sun in the sky, illuminating the surrounding landscape."
    if re.search(r"\b(moon)\b", low):
        return "The moon glowing in the night sky."
    if re.search(r"\b(baby|newborn|child)\b", low) and re.search(r"born|sleep|lie|cradl", low):
        return "A newborn baby resting peacefully."
    if re.search(r"\b(city|village|town)\b", low) and re.search(r"appear|build|rise|grow|emerge", low):
        return "A town with buildings and streets viewed as a clear wide scene."

    # spaCy dependency-based extraction: subject + visible action + object/location.
    if nlp:
        try:
            doc = nlp(raw)
            sent = next(iter(doc.sents), doc)
            root = next((t for t in sent if t.dep_ == "ROOT"), None)
            if root is not None:
                subjects = [t for t in root.children if t.dep_ in ("nsubj", "nsubjpass", "csubj")]
                objects = [t for t in root.children if t.dep_ in ("dobj", "obj", "attr", "oprd")]
                subject = ""
                if subjects:
                    subject = _clean_visual_phrase(" ".join(t.text for t in subjects[0].subtree))
                obj = ""
                if objects:
                    obj = _clean_visual_phrase(" ".join(t.text for t in objects[0].subtree))
                # Retain useful place phrases (in/on/under/above/near) but not
                # abstract clauses such as "waiting to keep you company".
                locations = []
                for token in sent:
                    if token.dep_ == "prep" and token.text.lower() in {
                        "in", "on", "under", "above", "near", "beside", "across", "around"
                    }:
                        pobj = next((c for c in token.children if c.dep_ == "pobj"), None)
                        if pobj:
                            locations.append(f"{token.text} {_clean_visual_phrase(' '.join(t.text for t in pobj.subtree))}")
                if subject:
                    verb = root.lemma_.lower()
                    action_map = {
                        "be": "standing", "appear": "appearing", "emerge": "emerging",
                        "grow": "growing", "bloom": "blooming", "shine": "shining",
                        "flicker": "glowing", "glow": "glowing", "rise": "rising",
                        "float": "floating", "fly": "flying", "run": "running",
                        "walk": "walking", "sit": "sitting", "stand": "standing",
                        "sleep": "sleeping", "dance": "dancing", "fall": "falling",
                        "move": "moving", "flow": "flowing", "burn": "burning",
                        "spread": "spreading", "cover": "covering",
                    }
                    action = action_map.get(verb, verb if verb.endswith("ing") else verb + "ing")
                    phrase = subject
                    if obj and obj.lower() not in subject.lower():
                        phrase += " with " + obj
                    if locations:
                        phrase += " " + locations[0]
                    return f"{phrase[:1].upper() + phrase[1:]} {action}. Keep the scene clear and visually focused."
        except Exception:
            pass

    # Fallback: remove dialogue, abstract/emotional clauses and simplify common
    # narrative verbs while preserving the main concrete nouns.
    raw = re.sub(r"\b(struggling|struggled|whispered|replied|said|thought|felt|wondered|remembered)\b[^,.!?]*",
                 "", raw, flags=re.I)
    raw = re.sub(r"\b(absolute|vast|gentle|finally|suddenly|quietly|softly)\b", "", raw, flags=re.I)
    raw = re.sub(r"\s+", " ", raw).strip(" ,.;:-")
    if not raw:
        return ""
    raw = re.sub(r"\bwas born\b", "appeared", raw, flags=re.I)
    raw = re.sub(r"\bwere born\b", "appeared", raw, flags=re.I)
    raw = re.sub(r"\bwas populated with\b", "filled with", raw, flags=re.I)
    raw = re.sub(r"\bwere populated with\b", "filled with", raw, flags=re.I)
    return raw[:1].upper() + raw[1:] + ("" if raw.endswith(".") else ".")


def simplify_story_scenes(scenes: list[str]) -> list[str]:
    """Simplify extracted story beats into concise prompts suitable for image models."""
    nlp = _load_spacy_pipeline()
    simplified = []
    for scene in scenes:
        prompt = simplify_scene_for_image(scene, nlp)
        if prompt:
            simplified.append(prompt)
    return simplified[:40]


def extract_story_entities(story: str) -> list[dict[str, str]]:
    """Extract named entities with spaCy when available; use a lightweight fallback."""
    text = (story or "").strip()
    if not text:
        return []
    nlp = _load_spacy_pipeline()
    names = []
    if nlp:
        try:
            doc = nlp(text)
            names = [ent.text.strip() for ent in doc.ents
                     if ent.label_ in {"PERSON", "GPE", "LOC", "ORG", "FAC", "PRODUCT"}]
        except Exception:
            names = []
    if not names:
        stop = {"The", "A", "An", "When", "While", "After", "Before", "Then", "One",
                "Suddenly", "But", "And", "He", "She", "They", "It", "In", "On", "At"}
        candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", text)
        names = [name for name in candidates if name.split()[0] not in stop]
    counts = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    results = []
    for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if count >= 1:
            results.append({"name": name, "description": "Named character or place. Add its fixed visual appearance or design here."})
    return results[:30]


# --------------------------------------------------------------------------- #
# Bootstrap / setup
# --------------------------------------------------------------------------- #
def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def is_setup() -> bool:
    return (MARKER.exists() and MARKER.read_text().strip() == SETUP_VERSION and venv_python().exists())


def run(cmd):
    print("  >", " ".join(str(c) for c in cmd), flush=True)
    subprocess.check_call([str(c) for c in cmd])


def py_version_of(cmd):
    try:
        out = subprocess.run([*cmd, "-c", "import sys;print(sys.version_info[0], sys.version_info[1])"],
                             capture_output=True, text=True, timeout=30).stdout.split()
        return (int(out[0]), int(out[1]))
    except Exception:
        return None


def venv_py_version():
    return py_version_of([str(venv_python())])


def find_compatible_python():
    if sys.version_info[:2] in SUPPORTED_PY:
        return [sys.executable]

    def probe():
        local = os.environ.get("LOCALAPPDATA")
        for major, minor in SUPPORTED_PY:
            cands = [["py", f"-{major}.{minor}"], [f"python{major}.{minor}"]]
            if local:
                cands.append([str(Path(local) / "Programs" / "Python" / f"Python{major}{minor}" / "python.exe")])
            for cmd in cands:
                if py_version_of(cmd) == (major, minor):
                    return cmd
        return None

    found = probe()
    if found:
        return found
    print(f"\nYour Python is {sys.version_info[0]}.{sys.version_info[1]}, but PyTorch only ships for "
          "Python 3.10-3.12.")
    if os.name == "nt" and shutil.which("winget"):
        ans = input("Install Python 3.12 next to it now using winget? [Y/n] ").strip().lower()
        if ans in ("", "y", "yes"):
            subprocess.call(["winget", "install", "-e", "--id", "Python.Python.3.12", "--scope", "user",
                             "--accept-package-agreements", "--accept-source-agreements"])
            found = probe()
            if found:
                return found
    sys.exit("\nNo compatible Python found. Install Python 3.12 from https://www.python.org/downloads/ "
             f"then run:\n    py -3.12 \"{Path(__file__).resolve()}\"")


def setup(force: bool = False):
    print("\n=== Image Gen Local setup (one time only) ===\n")
    for d in (ROOT, MODELS_DIR, OUTPUT_DIR, STATIC_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if VENV_DIR.exists() and venv_python().exists() and venv_py_version() not in SUPPORTED_PY:
        print("Existing environment uses an unsupported Python version; rebuilding it...")
        force = True
    if force:
        MARKER.unlink(missing_ok=True)
        if VENV_DIR.exists():
            print("Removing old environment...")
            shutil.rmtree(VENV_DIR, ignore_errors=True)

    if not venv_python().exists():
        base = find_compatible_python()
        print(f"[1/5] Creating private virtual environment using {' '.join(base)} ...")
        subprocess.check_call([*base, "-m", "venv", str(VENV_DIR)])
    else:
        print("[1/5] Reusing existing virtual environment.")

    py = venv_python()
    print("[2/5] Updating pip...")
    run([py, "-m", "pip", "install", "--upgrade", "pip"])

    print("[3/5] Installing PyTorch with CUDA (large download the first time)...")
    installed = False
    for index in TORCH_INDEXES:
        try:
            run([py, "-m", "pip", "install", "torch", "torchvision", "--index-url", index])
            installed = True
            break
        except subprocess.CalledProcessError:
            print(f"  (no luck with {index}, trying the next one)")
    if not installed:
        sys.exit("\nCould not install PyTorch from any CUDA index. Check your connection and that the "
                 "venv's Python is 3.10-3.12.")

    print("[4/5] Installing/upgrading diffusers, transformers, gguf, bitsandbytes...")
    run([py, "-m", "pip", "install", "--upgrade", *PACKAGES])
    try:
        run([py, "-m", "spacy", "download", "en_core_web_sm"])
    except subprocess.CalledProcessError:
        print("  (spaCy English model unavailable; story simplification will use its offline rules)")

    print("[5/5] Fetching Bootstrap CSS...")
    ensure_bootstrap()

    print("\nChecking GPU access...")
    res = subprocess.run([str(py), "-c", "import torch;print(torch.cuda.is_available());"
                          "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"],
                         capture_output=True, text=True)
    lines = res.stdout.strip().splitlines()
    if not lines or lines[0] != "True":
        sys.exit("\nPyTorch can't see your NVIDIA GPU. Update your NVIDIA driver, then run again with "
                 "--reinstall.\n" + res.stderr[-500:])
    print("GPU found:", lines[1] if len(lines) > 1 else "")

    for mod, note in (("bitsandbytes", "4-bit text encoders will not load"),
                      ("gguf", "GGUF models will not load")):
        chk = subprocess.run([str(py), "-c", f"import {mod}"], capture_output=True, text=True)
        if chk.returncode != 0:
            print(f"\nWARNING: {mod} failed to import; {note}.\n" + chk.stderr[-400:])

    MARKER.write_text(SETUP_VERSION)
    print("\nSetup complete.\n")


# --------------------------------------------------------------------------- #
# Front end (Bootstrap 5.3). Placeholders: __BSLINK__ __GPU__ __DEFAULT_THEME__
# --------------------------------------------------------------------------- #
INDEX_HTML = r"""<!doctype html>
<html lang="en" data-bs-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image Gen Local</title>
__BSLINK__
<script>
(function () {
  var t = "__DEFAULT_THEME__";
  try { t = localStorage.getItem("igl-theme") || t; } catch (e) {}
  if (t === "auto") t = matchMedia("(prefers-color-scheme: light)").matches ? "light" : "abyssal";
  document.documentElement.setAttribute("data-theme", t);
  document.documentElement.setAttribute("data-bs-theme", t === "light" ? "light" : "dark");
})();
</script>
<style>
html[data-theme="abyssal"] { color-scheme: dark; --solid:#141820; --accent:#7ed3bf; --accent-strong:#a3e6d4; --accent-soft:rgba(126,211,191,.16);
  --bg:#0a0d13; --card:rgba(159,178,204,.07); --card2:rgba(159,178,204,.12); --line:rgba(159,178,204,.16);
  --text:#eef1f5; --dim:#a8b4c4; --faint:#7a8898; --btn-fg:#0d1117; }
html[data-theme="forest"] { color-scheme: dark; --solid:#141c14; --accent:#9fd08a; --accent-strong:#bde3ab; --accent-soft:rgba(159,208,138,.16);
  --bg:#0c120c; --card:rgba(150,190,140,.07); --card2:rgba(150,190,140,.12); --line:rgba(150,190,140,.16);
  --text:#eef2ea; --dim:#a8bca3; --faint:#7a8f76; --btn-fg:#0c120c; }
html[data-theme="cocoa"] { color-scheme: dark; --solid:#1c1713; --accent:#e0b48f; --accent-strong:#f0cdb0; --accent-soft:rgba(224,180,143,.16);
  --bg:#120e0c; --card:rgba(205,165,135,.08); --card2:rgba(205,165,135,.13); --line:rgba(205,165,135,.18);
  --text:#f2ece5; --dim:#c0a892; --faint:#8a7060; --btn-fg:#120e0c; }
html[data-theme="dark"] { color-scheme: dark; --solid:#111111; --accent:#7ed3bf; --accent-strong:#a3e6d4; --accent-soft:rgba(126,211,191,.16);
  --bg:#000; --card:rgba(255,255,255,.06); --card2:rgba(255,255,255,.10); --line:rgba(255,255,255,.14);
  --text:#f1f1ee; --dim:#a8acb8; --faint:#7a7e8a; --btn-fg:#0d1117; }
html[data-theme="light"] { color-scheme: light; --solid:#f0ede6; --accent:#2fa889; --accent-strong:#1f8a70; --accent-soft:rgba(47,168,137,.12);
  --bg:#f5f4f0; --card:rgba(255,255,255,.85); --card2:rgba(240,238,232,.95); --line:rgba(0,0,0,.12);
  --text:#1a1d24; --dim:#4a5060; --faint:#727880; --btn-fg:#fff; }

html[data-theme] {
  --bs-body-bg: var(--bg); --bs-body-color: var(--text); --bs-emphasis-color: var(--text);
  --bs-secondary-color: var(--dim); --bs-tertiary-color: var(--faint);
  --bs-border-color: var(--line); --bs-link-color: var(--accent-strong); --bs-link-hover-color: var(--accent);
  --bs-secondary-bg: var(--card2); --bs-tertiary-bg: var(--card);
}
body { background: var(--bg); color: var(--text); font-weight: 300; }
::selection { background: var(--accent-soft); color: var(--accent-strong); }
h1 { font-weight: 400; letter-spacing: -.01em; }
.subtitle { color: var(--dim); font-style: normal; }
.card { --bs-card-bg: var(--card); --bs-card-border-color: var(--line); --bs-card-cap-bg: transparent; }
.form-label { font-size: .7rem; font-weight: 500; letter-spacing: .12em; text-transform: uppercase; color: var(--faint); margin-bottom: .25rem; }
.form-control, .form-select { background-color: var(--card2); border-color: var(--line); color: var(--text); font-weight: 300; }
.form-control:focus, .form-select:focus { background-color: var(--card2); color: var(--text);
  border-color: var(--accent); box-shadow: 0 0 0 .2rem var(--accent-soft); }
.form-control::placeholder { color: var(--faint); }
/* native dropdown lists ignore translucent backgrounds, so give them a solid themed one */
select.form-select, select.form-select option, select.form-select optgroup {
  background-color: var(--solid); color: var(--text); }
select.form-select { background-color: var(--card2); }
select.form-select option:checked, select.form-select option:hover { background-color: var(--accent-soft); color: var(--accent-strong); }
.form-text, .small-note { color: var(--faint) !important; font-size: .78rem; }
.form-range::-webkit-slider-thumb { background: var(--accent); }
.form-range::-moz-range-thumb { background: var(--accent); }
.form-check-input:checked { background-color: var(--accent); border-color: var(--accent); }
.btn-primary { --bs-btn-bg: var(--accent); --bs-btn-border-color: var(--accent); --bs-btn-color: var(--btn-fg);
  --bs-btn-hover-bg: var(--accent-strong); --bs-btn-hover-border-color: var(--accent-strong); --bs-btn-hover-color: var(--btn-fg);
  --bs-btn-active-bg: var(--accent-strong); --bs-btn-active-border-color: var(--accent-strong); --bs-btn-active-color: var(--btn-fg);
  --bs-btn-disabled-bg: var(--accent); --bs-btn-disabled-border-color: var(--accent); --bs-btn-disabled-color: var(--btn-fg); font-weight: 500; }
.btn-outline-secondary { --bs-btn-color: var(--dim); --bs-btn-border-color: var(--line);
  --bs-btn-hover-bg: var(--card2); --bs-btn-hover-color: var(--text); --bs-btn-hover-border-color: var(--line);
  --bs-btn-active-bg: var(--accent-soft); --bs-btn-active-color: var(--accent-strong); --bs-btn-active-border-color: var(--accent); }
.nav-tabs { --bs-nav-tabs-border-color: var(--line); --bs-nav-link-color: var(--dim); --bs-nav-link-hover-color: var(--text);
  --bs-nav-tabs-link-active-bg: transparent; --bs-nav-tabs-link-active-color: var(--accent-strong);
  --bs-nav-tabs-link-active-border-color: var(--line) var(--line) var(--bg); --bs-nav-tabs-link-hover-border-color: transparent; }
.progress { --bs-progress-bg: var(--card2); --bs-progress-bar-bg: var(--accent); height: .6rem; }
.badge-gpu { background: var(--accent-soft); border: 1px solid var(--accent); color: var(--accent-strong); font-weight: 600; }
.alert-info-soft { background: var(--card); border: 1px solid var(--line); color: var(--dim); font-size: .82rem; }
.sticky-panel { position: sticky; top: 1rem; max-height: calc(100vh - 2rem); overflow-y: auto; }
.result-img { cursor: zoom-in; aspect-ratio: 1 / 1; object-fit: contain; background: var(--card2); }
.empty { border: 1px dashed var(--line); border-radius: var(--bs-border-radius-lg); color: var(--faint); min-height: 40vh; }
#lightbox { position: fixed; inset: 0; z-index: 2000; background: rgba(0,0,0,.88); display: none; cursor: zoom-out; }
#lightbox img { max-width: 96vw; max-height: 96vh; }
.d-gguf { display: none; }
.is-gguf .d-gguf { display: block; }
.d-localgguf { display: none; }
.is-localgguf .d-localgguf { display: block; }
.is-sdcpp #memory, .is-sdcpp label[for="memory"] { display: none !important; }
.hist-thumb { aspect-ratio: 1 / 1; object-fit: cover; cursor: zoom-in; }

.model-ratings { padding: .65rem .75rem; border: 1px solid var(--bs-border-color); border-radius: .65rem; background: var(--bs-tertiary-bg); }
.rating-line { display: grid; grid-template-columns: minmax(105px, 1fr) auto minmax(72px, .9fr); align-items: center; gap: .55rem; margin: .2rem 0; }
.rating-label { font-size: .82rem; color: var(--bs-secondary-color); }
.rating-value { min-width: 3.8rem; text-align: center; font-size: .78rem; font-weight: 750; padding: .16rem .4rem; border-radius: 999px; }
.rating-good { color: #146c43; background: #d1e7dd; }
.rating-mid { color: #805500; background: #fff3cd; }
.rating-low { color: #842029; background: #f8d7da; }
.rating-meter { height: .42rem; border-radius: 99px; background: var(--bs-secondary-bg); overflow: hidden; }
.rating-meter > span { display: block; height: 100%; width: 0; border-radius: inherit; background: #198754; transition: width .2s ease; }
.rating-meter > span.mid { background: #d39e00; }
.rating-meter > span.low { background: #dc3545; }
.rating-note { margin-top: .4rem; font-size: .7rem; color: var(--bs-secondary-color); line-height: 1.35; }
.shape-options { display: grid !important; grid-template-columns: repeat(2, minmax(0, 1fr)); width: 100%; }
.shape-options > label { min-width: 0; width: 100%; white-space: nowrap; }
#panel, #panel .card-body, #panel .row, #panel [class*="col-"] { min-width: 0; }
#panel .form-select, #panel .form-control { max-width: 100%; }

</style>
</head>
<body>
<div class="container-fluid px-3 px-lg-4 py-4">

  <header class="d-flex flex-wrap align-items-end justify-content-between gap-3 mb-4">
    <div>
      <div class="d-flex flex-wrap align-items-center gap-2">
        <h1 class="h2 mb-0">Image Gen Local</h1>
        <span class="badge rounded-pill badge-gpu">__GPU__</span>
      </div>
      <p class="subtitle mb-0 mt-1">Turn a complete story into a consistent sequence of images.</p>
    </div>
    <div style="min-width: 190px;">
      <select id="theme" class="form-select form-select-sm" aria-label="Colour theme"></select>
    </div>
  </header>

  <div class="row g-4">
    <!-- results -->
    <div class="col-12">
      <ul class="nav nav-tabs mb-3" role="tablist">
        <li class="nav-item"><button class="nav-link" id="tab-res" type="button">Results</button></li>
        <li class="nav-item"><button class="nav-link active" id="tab-story" type="button">Story Mode</button></li>
        <li class="nav-item"><button class="nav-link" id="tab-his" type="button">History</button></li>
        <li class="nav-item"><button class="nav-link" id="tab-logs" type="button">Terminal logs</button></li>
      </ul>

      <div id="pane-res" class="d-none">
        <div id="empty" class="empty d-flex align-items-center justify-content-center text-center p-4">Your images will appear here.</div>
        <div id="grid" class="row row-cols-1 row-cols-md-2 row-cols-xxl-3 g-3"></div>
      </div>

      <div id="pane-story">
        <div id="progress-wrap" class="mb-2 d-none">
          <div class="progress" role="progressbar" aria-label="Progress">
            <div id="bar" class="progress-bar progress-bar-striped progress-bar-animated" style="width:0%"></div>
          </div>
        </div>
        <div id="status" class="small-note mb-3">Ready.</div>
        <div class="row g-3">
    
    <div class="col-12">
      <div class="card mb-3" id="story-options">
        <div class="card-body d-flex flex-column gap-3">
          <div>
            <label class="form-label" for="style">Art style</label>
            <select id="style" class="form-select">
              <option value="none">No style preset</option>
              <option value="watercolor">Watercolor painting</option>
              <option value="loose-watercolor">Loose watercolor wash</option>
              <option value="ink-wash">Japanese ink &amp; watercolor</option>
              <option value="pencil">Pencil sketch</option>
              <option value="charcoal">Charcoal drawing</option>
              <option value="ink">Fine-line ink illustration</option>
              <option value="gouache">Gouache painting</option>
              <option value="pastel">Soft pastel painting</option>
              <option value="dreamy">Dreamy ethereal art</option>
              <option value="surreal">Surreal dreamscape</option>
              <option value="storybook">Whimsical storybook illustration</option>
              <option value="botanical">Vintage botanical illustration</option>
              <option value="oil">Expressive oil painting</option>
              <option value="concept">Fantasy concept art</option>
            </select>
            <div class="form-text">Style recipes guide the selected checkpoint; they do not download separate model weights. For painterly results, start with DreamShaper XL Turbo.</div>
          </div>

          <div>
            <label class="form-label" for="model">Model</label>
            <select id="model" class="form-select"></select>
            <div id="model-ratings" class="model-ratings mt-2" aria-live="polite">
              <div class="rating-line">
                <span class="rating-label">Image quality</span>
                <span id="quality-rating" class="rating-value rating-good">—/10</span>
                <span id="quality-meter" class="rating-meter"><span></span></span>
              </div>
              <div class="rating-line">
                <span class="rating-label">Generation speed</span>
                <span id="speed-rating" class="rating-value rating-good">—/10</span>
                <span id="speed-meter" class="rating-meter"><span></span></span>
              </div>
              <div id="rating-note" class="rating-note">Estimated ratings for your GTX 1660 6GB setup; higher speed score means faster generation.</div>
            </div>
            <div id="info" class="alert alert-info-soft py-2 px-3 mt-2 mb-0"></div>
            <div class="small-note mt-2">All listed models are GGUF builds for native Windows stable-diffusion.cpp. CUDA does the compute; model weights can be offloaded to system RAM automatically when VRAM is tight.</div>
          </div>

          <div class="d-gguf">
            <div class="row g-2">
              <div class="col-6">
                <label class="form-label" for="quant">GGUF quantization</label>
                <select id="quant" class="form-select"></select>
              </div>
            </div>
            <div class="form-text">Auto selects Q4_K_S for 6 GB GPUs. Q8_0 is available for better fidelity and uses RAM offload as required.</div>
          </div>

          <div class="row g-2 align-items-end">
            <div class="col-12 col-sm-7">
              <label class="form-label d-block">Shape</label>
              <div class="btn-group shape-options" role="group" aria-label="Output shape">
                <input type="radio" class="btn-check" name="shape" id="sh-sq" value="Square" checked>
                <label class="btn btn-outline-secondary btn-sm" for="sh-sq">Square</label>
                <input type="radio" class="btn-check" name="shape" id="sh-po" value="Portrait">
                <label class="btn btn-outline-secondary btn-sm" for="sh-po">Portrait</label>
                <input type="radio" class="btn-check" name="shape" id="sh-la" value="Landscape">
                <label class="btn btn-outline-secondary btn-sm" for="sh-la">Landscape</label>
                <input type="radio" class="btn-check" name="shape" id="sh-yt" value="YouTube">
                <label class="btn btn-outline-secondary btn-sm" for="sh-yt">YouTube 16:9</label>
              </div>
            </div>
            <div class="col-12 col-sm-5">
              <label class="form-label" for="count">Images: <span id="count-v">1</span></label>
              <input id="count" type="range" class="form-range" min="1" max="8" step="1" value="1">
            </div>
          </div>

          <div>
            <button class="btn btn-sm btn-outline-secondary w-100" type="button" id="adv-toggle">Advanced options</button>
            <div class="collapse mt-3" id="adv">
              <div class="d-flex flex-column gap-3">
                <div>
                  <label class="form-label" for="negative">Negative prompt</label>
                  <textarea id="negative" class="form-control" rows="2"></textarea>
                  <div class="form-text">Used by SD models, and Z-Image only when guidance is above 1.</div>
                </div>
                <div class="row g-2">
                  <div class="col-6"><label class="form-label" for="steps">Steps: <span id="steps-v"></span></label>
                    <input id="steps" type="range" class="form-range" min="1" max="60" step="1"></div>
                  <div class="col-6"><label class="form-label" for="cfg">Guidance: <span id="cfg-v"></span></label>
                    <input id="cfg" type="range" class="form-range" min="0" max="15" step="0.5"></div>
                  <div class="col-6"><label class="form-label" for="width">Width: <span id="width-v"></span></label>
                    <input id="width" type="range" class="form-range" min="256" max="1024" step="64"></div>
                  <div class="col-6"><label class="form-label" for="height">Height: <span id="height-v"></span></label>
                    <input id="height" type="range" class="form-range" min="256" max="1024" step="64"></div>
                </div>
                <div class="row g-2">
                  <div class="col-5"><label class="form-label" for="seed">Seed (-1 random)</label>
                    <input id="seed" type="number" class="form-control" value="-1"></div>
                  <div class="col-7"><label class="form-label" for="sampler">Sampler (SD models)</label>
                    <select id="sampler" class="form-select"></select></div>
                </div>
                <div>
                  <label class="form-label" for="precision">Precision</label>
                  <select id="precision" class="form-select">
                    <option>auto</option><option>fp16</option><option>fp32</option>
                  </select>
                  <div class="form-text">auto: FP16 on GTX 1660 (BF16 is unsupported on this GPU).</div>
                </div>
                <div class="d-localgguf">
                  <label class="form-label" for="family">GGUF family (local file)</label>
                  <select id="family" class="form-select"></select>
                  <div class="form-text">Auto guesses from the file name. Set it if the guess is wrong.</div>
                </div>
                <div class="form-check form-switch">
                  <input class="form-check-input" type="checkbox" id="force-sdxl">
                  <label class="form-check-label small" for="force-sdxl">Treat local checkpoint as SDXL</label>
                </div>
                <div>
                  <label class="form-label" for="token">Hugging Face token</label>
                  <input id="token" type="password" class="form-control" placeholder="hf_... (optional, this session only)" autocomplete="off">
                  <div id="token-note" class="form-text"></div>
                </div>
                <div class="d-flex flex-wrap gap-2">
                  <button class="btn btn-sm btn-outline-secondary" type="button" id="refresh">Refresh models</button>
                  <button class="btn btn-sm btn-outline-secondary" type="button" id="open">Open outputs</button>
                  <button class="btn btn-sm btn-outline-secondary" type="button" id="free">Free VRAM</button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

        </div>
        <div class="row g-3">
          <div class="col-12">
            <div class="card"><div class="card-body">
              <label class="form-label fw-semibold" for="story-input">Your complete story</label>
              <textarea id="story-input" class="form-control" rows="10" placeholder="Paste or write the full story here. Separate scenes with headings such as Scene 1, Scene 2, Scene 3..."></textarea>
              <div class="d-flex flex-wrap gap-2 mt-3">
                <button id="story-extract" class="btn btn-primary" type="button">Extract scenes &amp; characters</button>
                <button id="story-clear" class="btn btn-outline-secondary" type="button">Clear</button>
              </div>
              <div class="form-text mt-2">Use headings like “Scene 1”, “Scene 2”, etc. If no headings are found, the story is split into visual beats. Each beat is rewritten as a short, concrete image prompt; dialogue and abstract metaphors are filtered out. Review and edit every prompt before generating.</div>
            </div></div>
          </div>
          <div class="col-12">
            <div class="card"><div class="card-body">
              <div class="d-flex flex-wrap justify-content-between align-items-start gap-2">
                <div>
                  <h3 class="h6 mb-1">Story reference image <span class="badge text-bg-secondary">Optional</span></h3>
                  <div class="form-text">Upload one visual example to guide the look of every scene. Scene prompts still control the action and composition.</div>
                </div>
                <button id="story-ref-remove" class="btn btn-sm btn-outline-secondary" type="button" disabled>Remove image</button>
              </div>
              <div class="row g-3 align-items-center mt-1">
                <div class="col-12 col-md-7">
                  <label class="form-label" for="story-ref-file">Reference image</label>
                  <input id="story-ref-file" class="form-control" type="file" accept="image/png,image/jpeg,image/webp">
                  <div class="form-text">PNG, JPG or WebP. The browser resizes it before local upload. Nothing is sent to a cloud image service.</div>
                </div>
                <div class="col-12 col-md-5">
                  <label class="form-label" for="story-ref-strength">Photo transformation: <span id="story-ref-strength-v">0.55</span></label>
                  <input id="story-ref-strength" class="form-range" type="range" min="0.20" max="1" step="0.05" value="0.55">
                  <div class="d-flex justify-content-between small-note"><span>Keep photo more</span><span>Change photo more</span></div>
                </div>
              </div>
              <div id="story-ref-preview-wrap" class="d-none mt-3 d-flex align-items-center gap-3">
                <img id="story-ref-preview" alt="Story reference preview" class="rounded border" style="width:112px;height:112px;object-fit:cover;background:var(--card2)">
                <div class="small"><div id="story-ref-name" class="fw-semibold"></div><div id="story-ref-status" class="text-body-secondary">Ready to use for all scenes.</div></div>
              </div>
              <div class="alert alert-info-soft py-2 px-3 mt-3 mb-0 small">CyberRealistic v9 uses image-to-image conditioning: your photo becomes the starting image, and the scene prompt transforms it. Lower strength preserves more of the original photo; higher strength lets the scene prompt change it more. SDXL GGUF models use IP-Adapter Plus. Classic Diffusers models are not supported by this Story reference control.</div>
            </div></div>
          </div>
          <div class="col-12">
            <div class="card"><div class="card-body">
              <h3 class="h6">Characters &amp; recurring details</h3>
              <div class="form-text mb-2">Review or edit the descriptions before generating. These details are added to every scene prompt to help maintain visual consistency.</div>
              <textarea id="story-entities" class="form-control" rows="7" placeholder="Extracted characters and recurring details will appear here."></textarea>
            </div></div>
          </div>
          <div class="col-12">
            <div class="card"><div class="card-body">
              <h3 class="h6">Simplified image prompts</h3>
              <div id="story-scenes" class="d-flex flex-column gap-3">
                <div class="text-body-secondary small">Extract your story to see editable scene prompts here.</div>
              </div>
            </div></div>
          </div>
          <div class="col-12">
            <div class="card"><div class="card-body">
              <div class="d-flex gap-2 mt-3"><button id="story-generate" class="btn btn-primary flex-grow-1 py-2" type="button">Generate all scene images</button><button id="stop" class="btn btn-outline-secondary" type="button" disabled>Stop</button></div>
              <div id="story-progress" class="small text-body-secondary mt-2"></div>
              <div id="story-grid" class="row row-cols-2 row-cols-md-3 g-2 mt-2"></div>
            </div></div>
          </div>
        </div>
      </div>

      <div id="pane-his" class="d-none">
        <div id="history-edit-panel" class="card mb-3 d-none"><div class="card-body">
          <div class="d-flex justify-content-between align-items-start gap-2 mb-3">
            <div><h3 class="h6 mb-1">Edit selected image</h3><div class="form-text">CyberRealistic v9 will use the selected image as its starting point. Keep the edit subtle to preserve its style.</div></div>
            <button id="history-edit-cancel" class="btn btn-sm btn-outline-secondary" type="button">Close</button>
          </div>
          <div class="row g-3 align-items-start">
            <div class="col-12 col-md-4"><img id="history-edit-preview" class="img-fluid rounded border w-100" alt="Selected history image" style="max-height:280px;object-fit:contain;background:var(--card2)"><div id="history-edit-name" class="small text-body-secondary mt-2"></div></div>
            <div class="col-12 col-md-8">
              <label class="form-label" for="history-edit-prompt">What should change?</label>
              <textarea id="history-edit-prompt" class="form-control" rows="4" placeholder="Example: Change the man's jacket to a dark blue coat. Keep his face, pose, background, lighting and overall style unchanged."></textarea>
              <div class="form-text mb-3">Describe only the changes. Unmentioned elements should remain as they are.</div>
              <label class="form-label" for="history-edit-strength">Change strength: <span id="history-edit-strength-v">0.28</span></label>
              <input id="history-edit-strength" class="form-range" type="range" min="0.12" max="0.65" step="0.01" value="0.28">
              <div class="d-flex justify-content-between small-note"><span>Preserve original</span><span>Allow more changes</span></div>
              <div class="d-flex flex-wrap gap-2 mt-3"><button id="history-edit-run" class="btn btn-primary" type="button">Generate edited image</button><span id="history-edit-status" class="small text-body-secondary align-self-center"></span></div>
            </div>
          </div>
        </div></div>
        <div id="hist-empty" class="empty d-flex align-items-center justify-content-center text-center p-4 d-none">Nothing generated yet.</div>
        <div id="hist" class="row row-cols-2 row-cols-md-3 row-cols-xl-4 row-cols-xxl-6 g-2"></div>
      </div>

      <div id="pane-logs" class="d-none">
        <div class="d-flex flex-wrap align-items-center justify-content-between gap-2 mb-2">
          <div class="small-note">Live Python stdout and stderr, including model downloads, warnings and tracebacks. Also continues printing in the terminal.</div>
          <div class="d-flex gap-2">
            <button class="btn btn-sm btn-outline-secondary" type="button" id="logs-refresh">Refresh</button>
            <button class="btn btn-sm btn-outline-secondary" type="button" id="logs-copy">Copy logs</button>
          </div>
        </div>
        <pre id="terminal-logs" class="p-3 rounded border" style="height: min(68vh, 720px); overflow:auto; white-space:pre-wrap; overflow-wrap:anywhere; background:var(--solid); color:var(--text); font:12px/1.5 Consolas, 'Cascadia Mono', monospace;">Waiting for Python output...</pre>
      </div>
    </div>
  </div>
</div>

<div id="lightbox"><div class="h-100 d-flex align-items-center justify-content-center"><img id="lb-img" alt=""></div></div>

<script>
const $ = (id) => document.getElementById(id);
let CFG = null, timer = null, curModel = null;

async function api(path, body) {
  const opt = body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)};
  const r = await fetch(path, opt);
  return r.json();
}
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }

function dims(size, shape) {
  const r = (v) => Math.max(256, Math.min(1024, Math.round(v / 64) * 64));
  if (shape === "Portrait") return [size, r(size * 1.5)];
  if (shape === "Landscape") return [r(size * 1.5), size];
  if (shape === "YouTube") return [r(size * (4 / 3)), r(size * 0.75)];
  return [size, size];
}
const shape = () => document.querySelector("input[name=shape]:checked").value;
function bindRange(id) { const i = $(id), v = $(id + "-v"); const u = () => { v.textContent = i.value; }; i.addEventListener("input", u); u(); }
["count", "steps", "cfg", "width", "height"].forEach(bindRange);
function setRange(id, val) { $(id).value = val; $(id + "-v").textContent = val; }

function applyInfo(m, keepSize) {
  $("info").innerHTML = "";
  $("info").appendChild(document.createTextNode(m.info));
  const root = document.documentElement;
  document.body.classList.toggle("is-gguf", !!m.gguf);
  document.body.classList.toggle("is-localgguf", !!m.local_gguf);
  document.body.classList.toggle("is-sdcpp", m.kind === "sdcpp");
  setRange("steps", m.steps); setRange("cfg", m.cfg);
  const [w, h] = dims(m.size, shape());
  setRange("width", w); setRange("height", h);
  const paintRating = (id, meterId, value) => {
    const score = Number(value);
    const valid = Number.isFinite(score);
    const safe = valid ? Math.max(0, Math.min(10, score)) : 0;
    const level = safe >= 8 ? "good" : (safe >= 6 ? "mid" : "low");
    const badge = $(id), fill = $(meterId).firstElementChild;
    badge.textContent = valid ? safe.toFixed(1) + "/10" : "N/A";
    badge.className = "rating-value rating-" + level;
    fill.style.width = safe * 10 + "%";
    fill.className = level === "good" ? "" : level;
  };
  paintRating("quality-rating", "quality-meter", m.quality_rating);
  paintRating("speed-rating", "speed-meter", m.speed_rating);
  $("rating-note").textContent = m.rating_note ||
    "Estimated for GTX 1660 6GB at 768×768, Q4_K_S, one image. Higher generation-speed score means faster; actual time varies with CPU, RAM and offloading.";
  curModel = m;
}
async function onModel() {
  const m = await api("/api/modelinfo?model=" + encodeURIComponent($("model").value));
  if (m.error) { $("status").textContent = m.error; return; }
  if (Array.isArray(m.quants) && m.quants.length) {
    const quant = $("quant");
    const previous = quant.value;
    quant.innerHTML = "";
    m.quants.forEach((q) => { const option = el("option", null, q); option.value = q; quant.appendChild(option); });
    quant.value = m.quants.includes(previous) ? previous : "Auto";
  }
  applyInfo(m);
}
function onShape() { if (!curModel) return; const [w, h] = dims(curModel.size, shape()); setRange("width", w); setRange("height", h); }

function fillModels(list, keep) {
  const sel = $("model"); sel.innerHTML = "";
  const groups = {};
  list.forEach((m) => {
    if (!groups[m.group]) { groups[m.group] = el("optgroup"); groups[m.group].label = m.group; sel.appendChild(groups[m.group]); }
    const o = el("option", null, m.label); o.value = m.value; groups[m.group].appendChild(o);
  });
  if (keep && list.some((m) => m.value === keep)) sel.value = keep;
}

function showImages(images) {
  const grid = $("grid"); grid.innerHTML = "";
  const storyGrid = $("story-grid");
  if (storyGrid) storyGrid.innerHTML = "";
  $("empty").classList.toggle("d-none", images.length > 0);
  images.forEach((im) => {
    const col = el("div", "col"), card = el("div", "card h-100 overflow-hidden");
    const img = el("img", "card-img-top result-img"); img.src = im.url; img.alt = "seed " + im.seed; img.loading = "lazy";
    img.addEventListener("click", () => openLb(im.url));
    const foot = el("div", "card-footer d-flex justify-content-between align-items-center small");
    foot.appendChild(el("span", "text-body-secondary", "seed " + im.seed));
    const a = el("a", "link-primary text-decoration-none", "Download"); a.href = im.url; a.download = im.name;
    foot.appendChild(a); card.appendChild(img); card.appendChild(foot); col.appendChild(card); grid.appendChild(col);
    if (storyGrid) storyGrid.appendChild(col.cloneNode(true));
  });
}
function openLb(url) { $("lb-img").src = url; $("lightbox").style.display = "block"; }
$("lightbox").addEventListener("click", () => { $("lightbox").style.display = "none"; });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("lightbox").style.display = "none"; });

function setRunning(r) {
  if ($("go")) $("go").disabled = r;
  if ($("stop")) $("stop").disabled = !r;
  if ($("go-spin")) $("go-spin").classList.toggle("d-none", !r);
  if ($("go-text")) $("go-text").textContent = r ? "Generating..." : "Generate";
  if ($("story-generate")) $("story-generate").disabled = r;
  $("progress-wrap").classList.toggle("d-none", !r);
}
async function poll() {
  let s;
  try { s = await api("/api/status"); } catch (e) { return; }
  $("bar").style.width = Math.round((s.progress || 0) * 100) + "%";
  $("status").textContent = s.error ? s.error : s.message;
  $("status").classList.toggle("text-danger", !!s.error);
  showImages(s.images || []);
  setRunning(s.running);
  if (!s.running) {
    clearInterval(timer); timer = null;
    if (!$("pane-his").classList.contains("d-none")) loadHistory();
  }
}
function startPolling() { if (!timer) timer = setInterval(poll, 500); poll(); }

async function generate() {
  const p = {
    prompt: $("prompt").value, negative: $("negative").value, model: $("model").value,
    style: $("style").value, family: $("family").value, quant: $("quant").value,
    force_sdxl: $("force-sdxl").checked, sampler: $("sampler").value,
    steps: +$("steps").value, cfg: +$("cfg").value, width: +$("width").value, height: +$("height").value,
    count: +$("count").value, seed: $("seed").value === "" ? -1 : +$("seed").value,
    precision: $("precision").value, token: $("token").value,
  };
  const r = await api("/api/generate", p);
  if (r.error) { $("status").textContent = r.error; $("status").classList.add("text-danger"); return; }
  $("status").classList.remove("text-danger");
  setRunning(true); startPolling();
}
$("stop").addEventListener("click", async () => { const r = await api("/api/stop", {}); $("status").textContent = r.message; });
$("free").addEventListener("click", async () => { const r = await api("/api/free", {}); $("status").textContent = r.message; });
$("open").addEventListener("click", async () => { const r = await api("/api/open", {}); $("status").textContent = r.message; });
$("refresh").addEventListener("click", async () => { const c = await api("/api/models"); fillModels(c.models, $("model").value); onModel(); });
$("model").addEventListener("change", onModel);
document.querySelectorAll("input[name=shape]").forEach((r) => r.addEventListener("change", onShape));
$("adv-toggle").addEventListener("click", () => { $("adv").classList.toggle("show"); });

let logsTimer = null;
function selectTab(which) {
  for (const [key, id] of [["res", "res"], ["story", "story"], ["his", "his"], ["logs", "logs"]]) {
    $("tab-" + id).classList.toggle("active", which === key);
    $("pane-" + id).classList.toggle("d-none", which !== key);
  }
  if (logsTimer) { clearInterval(logsTimer); logsTimer = null; }
  if (which === "his") loadHistory();
  if (which === "logs") {
    loadTerminalLogs();
    logsTimer = setInterval(loadTerminalLogs, 1000);
  }
}

let storyScenes = [];
let storyEntities = [];
let storyReferenceData = "";
let storyReferenceName = "";

function clearStoryReference() {
  storyReferenceData = "";
  storyReferenceName = "";
  $("story-ref-file").value = "";
  $("story-ref-preview").removeAttribute("src");
  $("story-ref-preview-wrap").classList.add("d-none");
  $("story-ref-remove").disabled = true;
  $("story-ref-status").textContent = "";
}
function prepareStoryReference(file) {
  if (!file) return;
  if (!/^image\/(png|jpeg|webp)$/.test(file.type)) {
    $("status").textContent = "Choose a PNG, JPG or WebP image.";
    $("story-ref-file").value = "";
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    $("status").textContent = "Reference image must be 20 MB or smaller.";
    $("story-ref-file").value = "";
    return;
  }
  const reader = new FileReader();
  reader.onerror = () => { $("status").textContent = "Could not read the reference image."; };
  reader.onload = () => {
    const image = new Image();
    image.onerror = () => { $("status").textContent = "This image could not be decoded."; };
    image.onload = () => {
      const maxSide = 1024;
      const scale = Math.min(1, maxSide / Math.max(image.naturalWidth, image.naturalHeight));
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
      canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
      const ctx = canvas.getContext("2d", {alpha:false});
      ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      storyReferenceData = canvas.toDataURL("image/jpeg", 0.88);
      storyReferenceName = file.name;
      $("story-ref-preview").src = storyReferenceData;
      $("story-ref-name").textContent = file.name + " · " + canvas.width + " × " + canvas.height;
      $("story-ref-status").textContent = "Ready. This reference will guide every scene in this story.";
      $("story-ref-preview-wrap").classList.remove("d-none");
      $("story-ref-remove").disabled = false;
      $("status").textContent = "Story reference image ready.";
    };
    image.src = reader.result;
  };
  reader.readAsDataURL(file);
}
$("story-ref-file").addEventListener("change", (e) => prepareStoryReference(e.target.files && e.target.files[0]));
$("story-ref-remove").addEventListener("click", clearStoryReference);
$("story-ref-strength").addEventListener("input", () => {
  $("story-ref-strength-v").textContent = Number($("story-ref-strength").value).toFixed(2);
});

function renderStoryScenes() {
  const root = $("story-scenes");
  root.innerHTML = "";
  if (!storyScenes.length) {
    root.appendChild(el("div", "text-body-secondary small", "No scenes extracted yet."));
    return;
  }
  storyScenes.forEach((scene, index) => {
    const wrap = el("div", "border rounded p-3");
    const head = el("div", "fw-semibold mb-2", "Scene " + (index + 1));
    const area = document.createElement("textarea");
    area.className = "form-control";
    area.rows = 4;
    area.value = scene;
    area.addEventListener("input", () => { storyScenes[index] = area.value; });
    wrap.appendChild(head);
    wrap.appendChild(area);
    root.appendChild(wrap);
  });
}
async function extractStory() {
  const story = $("story-input").value.trim();
  if (!story) { $("status").textContent = "Enter your story first."; return; }
  const result = await api("/api/story/parse", {story});
  if (result.error) { $("status").textContent = result.error; return; }
  storyScenes = result.scenes || [];
  storyEntities = result.entities || [];
  $("story-entities").value = storyEntities.map(x => x.name + ": " + x.description).join("\n");
  renderStoryScenes();
  $("status").textContent = `Extracted ${storyScenes.length} scenes and ${storyEntities.length} recurring details. Review and edit them before generating.`;
}
$("story-extract").addEventListener("click", extractStory);
$("story-clear").addEventListener("click", () => {
  $("story-input").value = "";
  $("story-entities").value = "";
  clearStoryReference();
  storyScenes = [];
  storyEntities = [];
  renderStoryScenes();
  $("story-progress").textContent = "";
  $("status").textContent = "Story cleared.";
});
$("story-generate").addEventListener("click", async () => {
  if (!storyScenes.length) { $("status").textContent = "Extract your story scenes first."; return; }
  const entities = $("story-entities").value.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
  const result = await api("/api/story/generate", {
    scenes: storyScenes,
    entities,
    model: $("model").value,
    style: $("style").value,
    negative: $("negative").value,
    width: +$("width").value,
    height: +$("height").value,
    steps: +$("steps").value,
    cfg: +$("cfg").value,
    count: +$("count").value,
    seed: $("seed").value === "" ? -1 : +$("seed").value,
    quant: $("quant").value,
    precision: $("precision").value,
    token: $("token").value,
    family: $("family").value,
    force_sdxl: $("force-sdxl").checked,
    sampler: $("sampler").value,
    reference_image: storyReferenceData,
    reference_strength: +$("story-ref-strength").value
  });
  if (result.error) { $("status").textContent = result.error; return; }
  $("story-progress").textContent = storyReferenceData
    ? "Generating scene images with your visual reference…"
    : "Generating scene images in order…";
  setRunning(true);
  startPolling();
});
$("tab-res").addEventListener("click", () => selectTab("res"));
$("tab-story").addEventListener("click", () => selectTab("story"));
$("tab-his").addEventListener("click", () => selectTab("his"));
$("tab-logs").addEventListener("click", () => selectTab("logs"));
async function loadTerminalLogs() {
  const box = $("terminal-logs");
  const wasAtBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 48;
  try {
    const r = await api("/api/logs");
    if (box.textContent !== r.logs) {
      box.textContent = r.logs || "No Python output captured yet.";
      if (wasAtBottom) box.scrollTop = box.scrollHeight;
    }
  } catch (e) {
    box.textContent += "\n[Could not fetch terminal logs: " + e + "]";
  }
}
$("logs-refresh").addEventListener("click", loadTerminalLogs);
$("logs-copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("terminal-logs").textContent);
    $("logs-copy").textContent = "Copied";
    setTimeout(() => { $("logs-copy").textContent = "Copy logs"; }, 1200);
  } catch (e) { $("logs-copy").textContent = "Copy unavailable"; }
});
let selectedHistoryImage = null;
async function loadHistory() {
  const h = await api("/api/history"); const box = $("hist"); box.innerHTML = "";
  $("hist-empty").classList.toggle("d-none", h.items.length > 0);
  h.items.forEach((it) => {
    const col = el("div", "col"), card = el("div", "card h-100 p-2");
    const img = el("img", "img-fluid rounded hist-thumb w-100");
    img.src = it.url; img.loading = "lazy"; img.title = (it.prompt || "") + "\nseed " + it.seed;
    img.addEventListener("click", () => openLb(it.url));
    const foot = el("div", "d-flex justify-content-between align-items-center gap-1 mt-2");
    const caption = el("span", "small text-body-secondary text-truncate", it.model || it.name);
    const edit = el("button", "btn btn-sm btn-outline-primary flex-shrink-0", "Edit");
    edit.type = "button"; edit.addEventListener("click", () => chooseHistoryImage(it));
    foot.appendChild(caption); foot.appendChild(edit); card.appendChild(img); card.appendChild(foot);
    col.appendChild(card); box.appendChild(col);
  });
}
function chooseHistoryImage(item) {
  selectedHistoryImage = item;
  $("history-edit-preview").src = item.url;
  $("history-edit-name").textContent = item.name + (item.model ? " · " + item.model : "");
  $("history-edit-prompt").value = "";
  $("history-edit-strength").value = "0.28";
  $("history-edit-strength-v").textContent = "0.28";
  $("history-edit-status").textContent = "Ready. Describe the specific changes you want.";
  $("history-edit-panel").classList.remove("d-none");
  $("history-edit-panel").scrollIntoView({behavior:"smooth", block:"start"});
  $("history-edit-prompt").focus();
}
$("history-edit-cancel").addEventListener("click", () => {
  selectedHistoryImage = null;
  $("history-edit-panel").classList.add("d-none");
});
$("history-edit-strength").addEventListener("input", () => {
  $("history-edit-strength-v").textContent = Number($("history-edit-strength").value).toFixed(2);
});
$("history-edit-run").addEventListener("click", async () => {
  if (!selectedHistoryImage) return;
  const instruction = $("history-edit-prompt").value.trim();
  if (!instruction) { $("history-edit-status").textContent = "Describe what you want to change first."; return; }
  const button = $("history-edit-run"); button.disabled = true;
  $("history-edit-status").textContent = "Starting CyberRealistic edit…";
  try {
    const r = await api("/api/history/edit", {
      name: selectedHistoryImage.name,
      instruction,
      strength: Number($("history-edit-strength").value)
    });
    if (r.error) { $("history-edit-status").textContent = r.error; return; }
    $("history-edit-status").textContent = "Editing started. Watch the progress at the top; the new image will appear in History.";
    setRunning(true); startPolling();
  } catch (e) {
    $("history-edit-status").textContent = "Could not start edit: " + e;
  } finally { button.disabled = false; }
});

function applyTheme(name) {
  const t = CFG.themes[name] || "abyssal";
  document.documentElement.setAttribute("data-theme", t);
  document.documentElement.setAttribute("data-bs-theme", t === "light" ? "light" : "dark");
  try { localStorage.setItem("igl-theme", t); } catch (e) {}
}

(async function init() {
  CFG = await api("/api/config");
  const ts = $("theme"); Object.keys(CFG.themes).forEach((n) => { const o = el("option", null, n); o.value = n; ts.appendChild(o); });
  let cur = document.documentElement.getAttribute("data-theme");
  Object.keys(CFG.themes).forEach((n) => { if (CFG.themes[n] === cur) ts.value = n; });
  ts.addEventListener("change", () => applyTheme(ts.value));
  CFG.quants.forEach((q) => { const o = el("option", null, q); o.value = q; $("quant").appendChild(o); });
  CFG.samplers.forEach((q) => { const o = el("option", null, q); o.value = q; $("sampler").appendChild(o); });
  CFG.families.forEach((f) => { const o = el("option", null, f.label); o.value = f.value; $("family").appendChild(o); });
  $("negative").value = CFG.default_negative;
  $("token-note").textContent = CFG.has_token ? "A saved token is in use; paste one here to override it." : "Optional. Public ungated models download without accepting a licence; token is used only for authenticated Hub access.";
  fillModels(CFG.models, CFG.default_model); $("model").value = CFG.default_model;
  await onModel();
  selectTab("story");
  const s = await api("/api/status");
  if (s.running) { setRunning(true); startPolling(); } else if ((s.images || []).length) showImages(s.images);
})();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# The app (runs inside the private venv)
# --------------------------------------------------------------------------- #
class UserError(Exception):
    """An error whose message is meant to be shown to the user as-is."""


def run_app(args):
    os.environ["HF_HOME"] = str(ROOT / "hf_cache")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    hf_token = resolve_hf_token()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import gc
    import html as htmlmod
    import random
    from collections import deque

    # Mirror Python stdout/stderr to a bounded in-memory buffer for the web UI,
    # while continuing to write every byte to the original terminal streams.
    log_lock = threading.Lock()
    log_chunks = deque(maxlen=12000)

    class TeeLogStream:
        def __init__(self, original):
            self.original = original

        def write(self, value):
            if value:
                with log_lock:
                    log_chunks.append(str(value))
            return self.original.write(value)

        def flush(self):
            return self.original.flush()

        def isatty(self):
            return self.original.isatty()

        def fileno(self):
            return self.original.fileno()

        @property
        def encoding(self):
            return getattr(self.original, "encoding", "utf-8")

    sys.stdout = TeeLogStream(sys.stdout)
    sys.stderr = TeeLogStream(sys.stderr)
    print("Terminal log capture enabled (web app + original console).")

    import torch
    from diffusers import (
        AutoencoderKL,
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
    )
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    from tqdm.auto import tqdm as base_tqdm

    for d in (MODELS_DIR, OUTPUT_DIR, STATIC_DIR, STORY_REF_DIR):
        d.mkdir(parents=True, exist_ok=True)
    have_bootstrap = ensure_bootstrap(args.offline)
    if not torch.cuda.is_available():
        print("WARNING: CUDA GPU not available. Generation will fail. Try --reinstall.")

    DEFAULT_NEG = "blurry, lowres, bad anatomy, deformed, extra fingers, watermark, text"
    GIB = 1024 ** 3

    # ----- model catalog ---------------------------------------------------- #
    # Families = what the GGUF transformer is, plus the base repo that supplies text encoder, VAE
    # and configs (its transformer weights are never downloaded).
    FAMILIES = {}  # No non-OpenRAIL GGUF catalog entries are offered in this build.
    GGUF_CATALOG = {}
    # Ordered by expected wall-clock speed on a GTX 1660 6GB, not by model quality.
    # Estimates assume one image, FP16, batch=1 and the listed default resolution.
    # SDXL uses model CPU offload; it will be slower than an all-GPU SD 1.5 run.
    CLASSIC = {
        "01 | SDXL Lightning 4-step | OpenRAIL++": dict(
            kind="sdxl-lightning", repo="stabilityai/stable-diffusion-xl-base-1.0", steps=4, cfg=0.0, size=768,
            lightning_steps=4,
            info="FASTEST SDXL option. 4-step distilled UNet from ByteDance/SDXL-Lightning with its required Euler trailing scheduler and CFG 0. Default 768 to reduce 6GB VRAM pressure; CPU offload can make it slower than the step count suggests."),
        "02 | RealVisXL v5.0 Lightning | SDXL | OpenRAIL++": dict(
            kind="sdxl", repo="SG161222/RealVisXL_V5.0_Lightning", steps=5, cfg=1.5, size=768,
            recommended_sampler="DPM++ SDE Karras",
            info="Photorealistic distilled SDXL. The model card recommends 5 steps, CFG 1–2 and DPM++ SDE Karras. My recommended speed/quality balance for realistic portraits and photographic scenes; CPU offload still applies on 6GB."),
        "03 | DreamShaper XL Turbo | SDXL | OpenRAIL++": dict(
            kind="sdxl", repo="Lykon/dreamshaper-xl-turbo", steps=6, cfg=2.0, size=768,
            info="Fast SDXL fine-tune (OpenRAIL++). The model card recommends 6 steps and CFG 2. A good speed/quality compromise for artistic, illustrative and semi-realistic images. Default 768 for 6GB VRAM; 1024 may require automatic downscaling."),
        "04 | DreamShaper 8 | SD 1.5 | OpenRAIL-M": dict(
            kind="sd", repo="Lykon/dreamshaper-8", steps=24, cfg=6.5, size=640,
            info="Fast all-GPU SD 1.5 fine-tune (CreativeML OpenRAIL-M). Strong general-purpose stylised and semi-realistic images. Native training size is 512; default 640 is a compromise. Choose 512 for speed or 768 for more pixels (may look softer)."),
        "05 | SDXL Lightning 8-step | OpenRAIL++": dict(
            kind="sdxl-lightning", repo="stabilityai/stable-diffusion-xl-base-1.0", steps=8, cfg=0.0, size=768,
            lightning_steps=8,
            info="Higher-step SDXL Lightning variant. Usually takes longer than 4-step but can improve detail and stability. Uses the matching 8-step distilled UNet and Euler trailing scheduler; CFG 0."),
        "06 | Realistic Vision v6.0 | SDXL | OpenRAIL-M": dict(
            kind="sdxl", repo="stablediffusionapi/realistic-vision-v6.0", steps=28, cfg=5.5, size=768,
            info="Photorealistic SDXL fine-tune (CreativeML OpenRAIL-M). More natural-looking people and photographic scenes than fast distilled models, but much slower on 6GB because SDXL components are CPU-offloaded. Default 768; 1024 may downscale."),
        "07 | RealVisXL v5.0 | SDXL | OpenRAIL++": dict(
            kind="sdxl", repo="SG161222/RealVisXL_V5.0", steps=30, cfg=5.0, size=768,
            info="Photorealism-focused SDXL checkpoint (OpenRAIL++). A strong choice for realistic portraits, people and camera-like scenes. Large download; SDXL CPU offload on 6GB makes it slow. Start at 768."),
    }
    G_CAT, G_FILE, G_CLS, G_LOC = "[GGUF] ", "[GGUF file] ", "[Classic] ", "[local] "
    G_SDC = "[sd.cpp] "
    # GGUF files below are verified stable-diffusion.cpp conversions from Hugging Face.
    # Q4_K_S is the default for a 6 GB GPU; Q8_0 is available for higher fidelity
    # and is RAM-offloaded by the native engine when it does not fit in VRAM.
    SDC_MODELS = {
        "01 | RealVisXL v5 Lightning | Fast photorealism": dict(quality_rating=8.2, speed_rating=9.5, rating_note="Estimated for GTX 1660 6GB at 768×768, Q4_K_S, one image; not a measured benchmark. Higher speed score means faster.",
            repo="off-grid-ai/realvisxl-v5.0-lightning-GGUF",
            files={"Q4_K_S": "realvisxl-v5.0-lightning-Q4_K_S.gguf", "Q8_0": "realvisxl-v5.0-lightning-Q8_0.gguf"},
            steps=6, cfg=1.5, size=768, sampler="euler",
            info="Fast photorealistic SDXL Lightning. Recommended starting model: 6 steps, CFG 1.5, Euler. Q4_K_S is the 6 GB-friendly default; Q8_0 uses more RAM for higher fidelity."),
        "02 | DreamShaper XL v2 Turbo | Fast creative": dict(quality_rating=8.6, speed_rating=9.3, rating_note="Estimated for GTX 1660 6GB at 768×768, Q4_K_S, one image; not a measured benchmark. Higher speed score means faster.",
            repo="off-grid-ai/dreamshaper-xl-v2-turbo-GGUF",
            files={"Q4_K_S": "dreamshaper-xl-v2-turbo-Q4_K_S.gguf", "Q8_0": "dreamshaper-xl-v2-turbo-Q8_0.gguf"},
            steps=6, cfg=1.5, size=768, sampler="euler",
            info="Fast SDXL Turbo model for creative, painterly, illustrative and semi-realistic images. A good first choice for watercolor, gouache, sketch and dreamy style recipes. Model-card settings: 6 steps, CFG 1.5, Euler."),
        "01 | RealVisXL v5 | Photorealism": dict(quality_rating=9.1, speed_rating=6.0, rating_note="Estimated for GTX 1660 6GB at 768×768, Q4_K_S, one image; not a measured benchmark. Higher speed score means faster.",
            repo="off-grid-ai/realvisxl-v5.0-GGUF",
            files={"Q4_K_S": "realvisxl-v5.0-Q4_K_S.gguf", "Q8_0": "realvisxl-v5.0-Q8_0.gguf"},
            steps=24, cfg=5.0, size=768, sampler="dpm++2m",
            info="Higher-quality photorealistic SDXL. Slower than Lightning; model card recommends 24 steps, CFG 5 and DPM++ 2M."),
        "02 | Juggernaut XL v9 | Cinematic realism": dict(quality_rating=9.2, speed_rating=6.0, rating_note="Estimated for GTX 1660 6GB at 768×768, Q4_K_S, one image; not a measured benchmark. Higher speed score means faster.",
            repo="off-grid-ai/juggernaut-xl-v9-GGUF",
            files={"Q4_K_S": "juggernaut-xl-v9-Q4_K_S.gguf", "Q8_0": "juggernaut-xl-v9-Q8_0.gguf"},
            steps=24, cfg=5.0, size=768, sampler="dpm++2m",
            info="Versatile SDXL checkpoint for cinematic, photographic and general-purpose images. Slower high-detail profile: 24 steps, CFG 5, DPM++ 2M."),
        "03 | CyberRealistic v9 | Z-Image Turbo | Photorealism": dict(
            quality_rating=9.3, speed_rating=4.8,
            rating_note="Estimated, not benchmarked, for GTX 1660 6GB + 16GB system RAM. Uses Q4_K diffusion GGUF, Q4_K_M Qwen encoder, 8 steps and CPU offload. Approximately 1MP; slower than the previous Q3_K/4-step profile.",
            repo="gimly13/z-image-cyberrealistic-v9-gguf",
            files={"Q3_K": "zit_CyberRealistic_v9-Q3_K.gguf",
                   "Q4_K_S": "zit_CyberRealistic_v9-Q4_K_S.gguf",
                   "Q4_K": "zit_CyberRealistic_v9-Q4_K.gguf"},
            default_quant="Q4_K", steps=8, cfg=1.0, size=1024, sampler="euler",
            components={"clip_l": "clip/Qwen3-4B-Q4_K_M.gguf", "vae": "vae/ae.safetensors"},
            architecture="z-image",
            info="Photorealistic Z-Image Turbo fine-tune. Source-tested v9 starting profile: Q4_K diffusion, Qwen3-4B Q4_K_M encoder, 8 steps, CFG 1.0, Euler, approximately 1MP. Uses CPU offload on 6GB GPUs; expect slower generation. CyberRealistic-specific prompt normalization removes scene labels and explicitly requests one continuous image frame, while preserving the scene's intended subject count. Negative prompts are not passed for this Turbo architecture."),
        "05 | Illustrious XL v2 | Anime / illustration": dict(quality_rating=9.2, speed_rating=5.8, rating_note="Estimated for GTX 1660 6GB at 768×768, Q4_K_S, one image; not a measured benchmark. Higher speed score means faster.",
            repo="off-grid-ai/illustrious-xl-v2.0-GGUF",
            files={"Q4_K_S": "illustrious-xl-v2.0-Q4_K_S.gguf", "Q8_0": "illustrious-xl-v2.0-Q8_0.gguf"},
            steps=24, cfg=5.0, size=768, sampler="dpm++2m",
            info="Stylised illustration-focused SDXL model. Useful for graphic, decorative and highly designed art; pair with the watercolor, ink or storybook style recipes. Recommended model-card profile: 24 steps, CFG 5, DPM++ 2M."),
        "06 | Animagine XL 4.0 | Anime / character art": dict(quality_rating=9.1, speed_rating=5.8, rating_note="Estimated for GTX 1660 6GB at 768×768, Q4_K_S, one image; not a measured benchmark. Higher speed score means faster.",
            repo="off-grid-ai/animagine-xl-4.0-GGUF",
            files={"Q4_K_S": "animagine-xl-4.0-Q4_K_S.gguf", "Q8_0": "animagine-xl-4.0-Q8_0.gguf"},
            steps=24, cfg=5.0, size=768, sampler="dpm++2m",
            info="Anime and character illustration model. Recommended model-card profile: 24 steps, CFG 5, DPM++ 2M."),
        "07 | Pony Diffusion V6 XL | Stylised characters": dict(quality_rating=8.9, speed_rating=5.8, rating_note="Estimated for GTX 1660 6GB at 768×768, Q4_K_S, one image; not a measured benchmark. Higher speed score means faster.",
            repo="off-grid-ai/pony-diffusion-v6-xl-GGUF",
            files={"Q4_K_S": "pony-diffusion-v6-xl-Q4_K_S.gguf", "Q8_0": "pony-diffusion-v6-xl-Q8_0.gguf"},
            steps=24, cfg=5.0, size=768, sampler="dpm++2m",
            info="Specialised stylised-character and anime model. Recommended model-card profile: 24 steps, CFG 5, DPM++ 2M."),
    }
    # Prompt-level art recipes. These do not pretend to be separate checkpoints;
    # they steer any selected SDXL model, with DreamShaper XL Turbo recommended.
    ART_STYLES = {
        "none": "",
        "watercolor": "traditional watercolor painting, translucent watercolor pigments, delicate layered washes, soft wet-on-wet edges, subtle cold-pressed paper texture, luminous color blooms",
        "loose-watercolor": "loose expressive watercolor painting, spontaneous brushwork, flowing pigment, visible paper white, soft color bleeds, atmospheric washes, impressionistic composition",
        "ink-wash": "sumi-e inspired ink and watercolor painting, expressive brush lines, restrained palette, fluid ink contours, delicate transparent color washes, handmade washi paper texture",
        "pencil": "beautiful graphite pencil sketch, hand-drawn construction lines, varied pencil pressure, fine hatching and cross-hatching, subtle paper grain, monochrome study",
        "charcoal": "expressive charcoal drawing, rich charcoal blacks, soft smudged shading, visible charcoal strokes, tonal study on textured drawing paper",
        "ink": "fine-line pen and ink illustration, confident contour lines, delicate cross-hatching, detailed hand-drawn linework, archival ink on warm paper",
        "gouache": "gouache painting, opaque matte pigments, bold simplified shapes, layered brush strokes, rich yet harmonious colors, textured painted surface",
        "pastel": "soft pastel painting, powdery pastel pigments, gentle blended edges, muted harmonious palette, visible pastel strokes on textured paper",
        "dreamy": "dreamlike ethereal fine art, hazy luminous atmosphere, soft glowing light, delicate color gradients, poetic visual mood, subtle surreal details, painterly finish",
        "surreal": "surrealist dreamscape painting, imaginative impossible architecture, symbolic visual metaphors, floating forms, atmospheric perspective, mysterious poetic mood",
        "storybook": "whimsical hand-painted storybook illustration, charming shapes, expressive characters, warm gentle palette, delicate texture, magical narrative atmosphere",
        "botanical": "vintage botanical watercolor plate, scientifically observed plant forms, fine ink details, transparent watercolor, naturalist illustration, aged ivory paper",
        "oil": "expressive oil painting, visible impasto brush strokes, rich pigment, painterly color mixing, artful canvas texture, confident composition",
        "concept": "high-end fantasy concept art, imaginative worldbuilding, cinematic composition, atmospheric depth, painterly details, evocative lighting"
    }
    # Keep the supported photorealism and illustration choices in the UI.
    SDC_MODELS = {
        name: spec for name, spec in SDC_MODELS.items()
        if ("RealVisXL v5 |" in name) or ("Juggernaut XL v9 |" in name) or ("CyberRealistic v9 |" in name)
    }
    SDC_DEFAULT_LABEL = next(name for name in SDC_MODELS if name.startswith("01 |"))
    # best quality first; Auto walks this list
    QUANTS = ["Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K"]
    QUANT_FALLBACK = ["Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"]
    MEM_LEVELS = ["gpu", "balanced", "partial"]

    SAMPLERS = {
        "DPM++ 2M Karras": lambda c: DPMSolverMultistepScheduler.from_config(
            c, algorithm_type="dpmsolver++", solver_order=2, use_karras_sigmas=True, final_sigmas_type="sigma_min"),
        "DPM++ SDE Karras": lambda c: DPMSolverMultistepScheduler.from_config(
            c, algorithm_type="sde-dpmsolver++", solver_order=2, use_karras_sigmas=True),
        "Euler a": lambda c: EulerAncestralDiscreteScheduler.from_config(c),
        "DDIM": lambda c: DDIMScheduler.from_config(c),
    }

    state = {"pipe": None, "key": None, "plan": None, "esc": {}, "scale": {}, "force_prec": {}, "file_cache": {}}
    job = {"running": False, "progress": 0.0, "message": "Ready.", "images": [], "error": None, "stop": False}
    job_lock = threading.Lock()

    def setmsg(progress=None, message=None):
        with job_lock:
            if progress is not None:
                job["progress"] = max(0.0, min(1.0, float(progress)))
            if message is not None:
                job["message"] = message

    # ----- model discovery -------------------------------------------------- #
    def local_gguf_files():
        return sorted(p for p in MODELS_DIR.glob("*.gguf") if p.is_file())

    def model_list():
        # This build intentionally exposes GGUF image models only.
        return [dict(
                    group="GGUF models | native stable-diffusion.cpp",
                    label=f"{n}  |  Q {spec['quality_rating']:.1f}  ·  Speed {spec['speed_rating']:.1f}",
                    value=G_SDC + n)
                for n, spec in SDC_MODELS.items()]

    def guess_family(path: Path):
        n = path.name.lower()
        if any(t in n for t in ("z_image", "z-image", "zimage")):
            return "zimage"
        if "klein" in n or re.search(r"flux[-_.]?2", n):
            return "flux2-9b" if "9b" in n else "flux2-4b"
        if "schnell" in n:
            return "flux1-schnell"
        if "flux" in n:
            return "flux1-dev"
        try:  # fall back to the architecture string stored in the GGUF header
            from gguf import GGUFReader
            f = GGUFReader(str(path)).fields.get("general.architecture")
            arch = bytes(f.parts[f.data[0]]).decode("utf-8", "ignore").lower()
            if arch in ("lumina2", "z_image", "zimage"):
                return "zimage"
            if arch == "flux":
                return "flux1-dev"
            if arch in ("sd1", "sdxl", "unet"):
                raise UserError("That GGUF is a UNet model (SD 1.5 / SDXL). diffusers can't load those; "
                                "use the safetensors version or stable-diffusion.cpp.")
        except UserError:
            raise
        except Exception:
            pass
        return None

    def get_spec(choice, force_sdxl=False, family="auto"):
        if choice.startswith(G_SDC):
            name = choice[len(G_SDC):]
            if name not in SDC_MODELS:
                raise UserError("Unknown GGUF model. Press Refresh models.")
            spec = dict(SDC_MODELS[name])
            spec.update(kind="sdcpp", label=name, gguf=True, local_gguf=False, gated=False)
            return spec
        if choice.startswith(G_CAT):
            name = choice[len(G_CAT):]
            if name not in GGUF_CATALOG:
                raise UserError("Unknown model. Press Refresh models.")
            c = GGUF_CATALOG[name]
            fam = FAMILIES[c["family"]]
            return dict(kind="gguf", family=c["family"], gguf_repo=c["repo"], steps=fam["steps"], cfg=fam["cfg"],
                        size=fam["size"], gated=fam["gated"], info=fam["info"], gguf=True, local_gguf=False)
        if choice.startswith(G_FILE):
            path = MODELS_DIR / Path(choice[len(G_FILE):]).name
            if not path.exists():
                raise UserError("That GGUF file is gone. Press Refresh models.")
            fid = family if family in FAMILIES else guess_family(path)
            if not fid:
                raise UserError("Couldn't tell which model family this GGUF is. Pick one under Advanced > "
                                "GGUF family.")
            fam = FAMILIES[fid]
            return dict(kind="gguf", family=fid, gguf_file=str(path), steps=fam["steps"], cfg=fam["cfg"],
                        size=fam["size"], gated=fam["gated"], gguf=True, local_gguf=True,
                        info=f"Local GGUF file, treated as **{fam['label']}**. " + fam["info"])
        if choice.startswith(G_CLS):
            key = choice[len(G_CLS):]
            if key not in CLASSIC:
                raise UserError("Unknown model. Press Refresh models.")
            spec = dict(CLASSIC[key], gguf=False, local_gguf=False, gated=False)
            if force_sdxl and spec["kind"] == "sd":
                spec["kind"] = "sdxl"
            return spec
        if choice.startswith(G_LOC):
            name = Path(choice[len(G_LOC):]).name
            xl = force_sdxl or any(t in name.lower() for t in ("xl", "pony", "illustrious"))
            return dict(kind="sdxl" if xl else "sd", file=str(MODELS_DIR / name), steps=28, cfg=7.0,
                        size=768 if xl else 512, gated=False, gguf=False, local_gguf=False,
                        info="Local safetensors checkpoint. Tick 'Treat as SDXL' under Advanced if it is SDXL-based.")
        raise UserError("Unknown model. Press Refresh models.")

    def default_precision(spec):
        # GTX 1660 (Turing, SM 7.5) supports FP16 but not native BF16.
        return "fp16"

    # ----- downloads -------------------------------------------------------- #
    def make_tqdm(label_prefix):
        class PTqdm(base_tqdm):
            def update(self, n=1):
                out = super().update(n)
                try:
                    if self.total:
                        label = (self.desc or "files").strip()[:40]
                        setmsg(min(self.n / self.total, 1.0) * 0.9, f"{label_prefix}: {label}")
                except Exception:
                    pass
                return out
        return PTqdm

    def ensure_repo(repo, token=None, fp16_variant=False, skip_transformer=False):
        tag = "__base" if skip_transformer else ""
        target = MODELS_DIR / "hf" / (repo.replace("/", "__") + tag)
        done = target / ".localgen_complete"
        if done.exists():
            return target, done.read_text().strip() == "fp16"
        if args.offline:
            raise UserError("Offline mode: this model isn't downloaded yet. Run once without --offline.")
        setmsg(0, f"Contacting Hugging Face for {repo} (first time only)...")
        try:
            files = HfApi(token=token or None).list_repo_files(repo)
        except Exception as e:
            raise UserError(f"Couldn't list {repo} on Hugging Face (gated repo needs a token, or no internet): {e}")
        has_st = any(f.endswith(".safetensors") for f in files)
        ignore = ["*.md", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.mp4", "*.onnx", "*.onnx_data", "*.msgpack",
                  "*.h5", "*.ot", "*.pb", "*.tflite", "*openvino*", "safety_checker/*", "*.gguf"]
        if has_st:
            ignore += ["*.bin", "*.ckpt", "*.pt", "*.pth"]
        if "model_index.json" in files:  # diffusers layout: skip duplicate root single-file weights
            ignore += [f for f in files if "/" not in f and f.endswith(".safetensors")]
        if skip_transformer:  # GGUF supplies the transformer; keep only its config.json
            ignore += ["transformer/*.safetensors", "transformer/*.safetensors.index.json", "transformer/*.bin"]
        kwargs = dict(repo_id=repo, local_dir=str(target), cache_dir=str(ROOT / "hf_cache"),
                      ignore_patterns=ignore, max_workers=4, resume_download=True)
        has_fp16 = fp16_variant and any(f.endswith(".fp16.safetensors") for f in files)
        if has_fp16:
            kwargs["allow_patterns"] = ["*.json", "*.txt", "*.fp16.safetensors"]
        if token:
            kwargs["token"] = token
        try:
            try:
                snapshot_download(**kwargs, tqdm_class=make_tqdm("Downloading"))
            except TypeError:
                kwargs.pop("resume_download", None)
                try:
                    snapshot_download(**kwargs, tqdm_class=make_tqdm("Downloading"))
                except TypeError:
                    snapshot_download(**kwargs)
        except Exception as e:
            if any(t in str(e) for t in ("401", "403", "gated", "restricted")):
                raise UserError("Hugging Face denied this download. Check whether the repository is gated or requires a licence click-through; a read token alone cannot bypass those requirements.")
            raise UserError(f"Download failed: {e}")
        done.write_text("fp16" if has_fp16 else "full")
        return target, has_fp16

    def gguf_dir(repo):
        return MODELS_DIR / "gguf" / repo.replace("/", "__")

    def repo_gguf_files(repo, token):
        """{filename: size_in_bytes} for the .gguf files of a repo (falls back to what is on disk)."""
        if repo in state["file_cache"]:
            return state["file_cache"][repo]
        files = {}
        if not args.offline:
            try:
                info = HfApi(token=token or None).model_info(repo, files_metadata=True)
                files = {s.rfilename: (s.size or 0) for s in info.siblings if s.rfilename.lower().endswith(".gguf")}
            except Exception as e:
                print(f"Couldn't list {repo}: {e}")
        if not files and gguf_dir(repo).exists():
            files = {str(p.relative_to(gguf_dir(repo))).replace("\\", "/"): p.stat().st_size
                     for p in gguf_dir(repo).rglob("*.gguf")}
        if files:
            state["file_cache"][repo] = files
        return files

    def match_quant(files, quant):
        pat = re.compile(rf"(?<![a-z0-9]){re.escape(quant.lower())}(?![a-z0-9])")
        hits = [f for f in files if pat.search(Path(f).name.lower())]
        return min(hits, key=len) if hits else None

    def pick_gguf(spec, quant, budget_gb, token):
        """Choose the file to use from the catalog repo: explicit quant, or the best that fits."""
        files = repo_gguf_files(spec["gguf_repo"], token)
        if not files:
            raise UserError(f"No GGUF files found in {spec['gguf_repo']} (no internet, or the repo changed).")
        if quant and quant != "Auto":
            f = match_quant(files, quant)
            if not f:
                have = ", ".join(sorted({q for q in QUANTS if match_quant(files, q)})) or "none recognised"
                raise UserError(f"{spec['gguf_repo']} has no {quant} file. Available: {have}.")
            return f, files[f]
        for q in QUANTS:  # best quality that fits fully on the GPU
            f = match_quant(files, q)
            if f and files[f] / GIB * 1.1 <= budget_gb:
                return f, files[f]
        for q in QUANT_FALLBACK:  # nothing fits: take a sensible middle quant and offload part of it
            f = match_quant(files, q)
            if f:
                return f, files[f]
        f = min(files, key=lambda k: files[k])
        return f, files[f]

    def ensure_gguf(repo, filename, token):
        target = gguf_dir(repo)
        path = target / filename
        if path.exists():
            return path
        if args.offline:
            raise UserError("Offline mode: this GGUF isn't downloaded yet. Run once without --offline.")
        setmsg(0, f"Downloading {filename} (first time only)...")
        kw = dict(repo_id=repo, filename=filename, local_dir=str(target))
        if token:
            kw["token"] = token
        try:
            try:
                p = hf_hub_download(**kw, tqdm_class=make_tqdm("Downloading GGUF"))
            except TypeError:
                p = hf_hub_download(**kw)
        except Exception as e:
            if any(t in str(e) for t in ("401", "403", "gated", "restricted")):
                raise UserError("Hugging Face denied this download. Check repository access and any licence requirements on its model page.")
            raise UserError(f"GGUF download failed: {e}")
        return Path(p)

    # ----- native Windows stable-diffusion.cpp backend ----------------------- #
    # The Python UI stays in the existing venv. The inference engine is a native
    # Windows CUDA executable; Docker/WSL2/Git/CMake are deliberately not used.
    SDC_RELEASE_API = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest"
    SDC_BIN_DIR = SDCPP_DIR / "bin"
    SDC_EXE = SDC_BIN_DIR / "sd-cli.exe"
    sdc_proc_lock = threading.Lock()
    sdc_proc = {"process": None}

    def download_file(url, destination, label):
        import urllib.request
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix(destination.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "ImageGenLocal/1.0"})
        print(f"{label}: {url}", flush=True)
        try:
            with urllib.request.urlopen(request, timeout=60) as response, open(temp, "wb") as out:
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"[Download] {label}: {done/total:.0%} ({done/1024**2:.0f}/{total/1024**2:.0f} MiB)", flush=True)
            temp.replace(destination)
        except Exception:
            temp.unlink(missing_ok=True)
            raise
        return destination

    def ensure_sd_cpp(require_zimage=False):
        if os.name != "nt":
            raise UserError("This backend installer is configured for native Windows. Run the app on Windows.")
        if SDC_EXE.exists() and (SDC_BIN_DIR / "stable-diffusion.dll").exists():
            # Reuse the installed engine only if it supports RAM auto-fit and IP-Adapter
            # flags used by this build. Otherwise fetch a current official Windows binary.
            try:
                probe = subprocess.run([str(SDC_EXE), "--help"], cwd=str(SDC_BIN_DIR),
                                       capture_output=True, text=True, timeout=30,
                                       encoding="utf-8", errors="replace",
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                help_text = (probe.stdout or "") + (probe.stderr or "")
                required = ["--offload-to-cpu", "--auto-fit", "--ip-adapter", "--clip_vision"]
                has_zimage_cli = "--llm" in help_text and "--vae" in help_text and "--diffusion-model" in help_text
                if all(flag in help_text for flag in required) and (not require_zimage or has_zimage_cli):
                    return str(SDC_EXE)
                print("Installed stable-diffusion.cpp lacks required RAM auto-fit / IP-Adapter options; updating it...", flush=True)
            except Exception as e:
                print(f"Could not validate installed stable-diffusion.cpp ({e}); updating it...", flush=True)
            SDC_EXE.unlink(missing_ok=True)
        if args.offline:
            raise UserError(f"Offline mode: native sd.cpp is not installed. Expected executable: {SDC_EXE}")
        import json
        import urllib.request
        import zipfile
        import tempfile

        SDCPP_DIR.mkdir(parents=True, exist_ok=True)
        SDC_BIN_DIR.mkdir(parents=True, exist_ok=True)
        setmsg(0.02, "Checking official stable-diffusion.cpp Windows CUDA release ...")
        req = urllib.request.Request(SDC_RELEASE_API, headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "ImageGenLocal/1.0",
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                release = json.loads(response.read().decode("utf-8"))
        except Exception as e:
            raise UserError(f"Could not query the official stable-diffusion.cpp releases: {e}")

        assets = release.get("assets", [])
        exe_asset = next((a for a in assets if
                          "bin-win-cuda12-x64.zip" in a.get("name", "").lower()
                          and a.get("browser_download_url")), None)
        runtime_asset = next((a for a in assets if
                              "cudart-sd-bin-win-cu12-x64.zip" in a.get("name", "").lower()
                              and a.get("browser_download_url")), None)
        if not exe_asset:
            names = "\n".join(a.get("name", "") for a in assets)
            raise UserError(
                "The latest official stable-diffusion.cpp release does not expose both the Windows CUDA 12 x64 executable and its matching CUDA runtime DLL package. "
                "No Docker or source build will be attempted. Available release assets:\n" + names
            )
        setmsg(0.05, "Downloading native Windows CUDA engine ...")
        with tempfile.TemporaryDirectory(prefix="sdcpp-install-") as temp_dir:
            temp_dir = Path(temp_dir)
            exe_zip = download_file(exe_asset["browser_download_url"], temp_dir / exe_asset["name"],
                                    "Downloading stable-diffusion.cpp Windows CUDA binary")
            with zipfile.ZipFile(exe_zip) as zf:
                zf.extractall(temp_dir / "engine")
            engine_root = temp_dir / "engine"
            cli_candidates = list(engine_root.rglob("sd-cli.exe"))
            if not cli_candidates:
                cli_candidates = list(engine_root.rglob("sd.exe"))
            if not cli_candidates:
                raise UserError("The downloaded Windows engine archive contains no sd-cli.exe or sd.exe. The archive may be incomplete or have changed layout.")
            cli = cli_candidates[0]
            # Keep all release DLLs beside the executable so Windows' DLL loader can resolve them.
            for item in cli.parent.iterdir():
                if item.is_file():
                    target = SDC_BIN_DIR / item.name
                    shutil.copy2(item, target)
            if cli.name.lower() != "sd-cli.exe":
                shutil.copy2(cli, SDC_BIN_DIR / "sd-cli.exe")
            else:
                shutil.copy2(cli, SDC_EXE)
            # Some release archives place DLLs in a sibling directory.
            for dll in engine_root.rglob("*.dll"):
                shutil.copy2(dll, SDC_BIN_DIR / dll.name)

            if runtime_asset:
                setmsg(0.10, "Downloading matching CUDA 12 runtime DLL package ...")
                runtime_zip = download_file(runtime_asset["browser_download_url"],
                                            temp_dir / runtime_asset["name"],
                                            "Downloading CUDA runtime DLLs")
                with zipfile.ZipFile(runtime_zip) as zf:
                    for member in zf.infolist():
                        if member.is_dir() or not member.filename.lower().endswith(".dll"):
                            continue
                        extracted = zf.extract(member, temp_dir / "runtime")
                        shutil.copy2(extracted, SDC_BIN_DIR / Path(member.filename).name)

        if not SDC_EXE.exists():
            raise UserError(f"Native sd.cpp installation did not produce {SDC_EXE}")
        # Verify the actual executable and Windows DLL loader before accepting install.
        probe = subprocess.run([str(SDC_EXE), "--help"], cwd=str(SDC_BIN_DIR),
                               capture_output=True, text=True, timeout=45,
                               encoding="utf-8", errors="replace",
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        probe_text = (probe.stdout or "") + (probe.stderr or "")
        has_zimage_cli = "--llm" in probe_text and "--vae" in probe_text and "--diffusion-model" in probe_text
        if (probe.returncode not in (0, 1) or "img_gen" not in probe_text.lower()
                or (require_zimage and not has_zimage_cli)):
            raise UserError(
                "The native Windows engine was downloaded but failed its startup/CLI validation. "
                "This is usually a missing/mismatched DLL or missing Z-Image CLI support. Full validation output:\n" + probe_text[-3000:]
            )
        print(f"Native stable-diffusion.cpp ready: {SDC_EXE}", flush=True)
        print(f"Release: {release.get('tag_name', 'latest')} | CUDA 12 Windows x64", flush=True)
        return str(SDC_EXE)

    def ensure_sdc_model(spec, quant="Auto", token=None):
        files = spec["files"]
        # Keep the 6 GB profile conservative. Models can override the Auto quant.
        quant = spec.get("default_quant", "Q4_K_S") if quant in (None, "", "Auto") else quant
        if quant not in files:
            raise UserError(f"{spec['label']} has no {quant} quant. Available: {', '.join(files)}")
        filename = files[quant]
        repo = spec["repo"]
        repo_dir = MODELS_DIR / "sd_cpp_models" / repo.replace("/", "__")
        target = repo_dir / filename
        if target.exists() and target.stat().st_size > 1_000_000_000:
            return target, quant
        if args.offline:
            raise UserError(f"Offline mode: {filename} is not downloaded. Download it from https://huggingface.co/{repo}/blob/main/{filename} into {repo_dir}.")
        setmsg(0.23, f"Downloading {spec['label']} {quant} GGUF (first run only) ...")
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(repo_id=repo, filename=filename,
                                         local_dir=str(repo_dir), token=token or None)
            result = Path(downloaded)
            if not result.exists() or result.stat().st_size < 1_000_000_000:
                raise RuntimeError("downloaded GGUF file is unexpectedly small")
            print(f"GGUF ready: {result} ({result.stat().st_size / 1024**3:.2f} GiB)")
            return result, quant
        except Exception as e:
            raise UserError(f"Could not download {filename} from Hugging Face ({repo}): {e}")

    def ensure_sdc_component(spec, relative_filename, token=None):
        """Download and validate a companion file from the same public GGUF bundle."""
        repo = spec["repo"]
        repo_dir = MODELS_DIR / "sd_cpp_models" / repo.replace("/", "__")
        target = repo_dir / relative_filename
        # The encoder is ~2.5 GB and the VAE ~335 MB; reject partial/HTML downloads.
        minimum_size = 1_000_000_000 if relative_filename.lower().endswith(".gguf") else 100_000_000
        if target.exists() and target.stat().st_size >= minimum_size:
            return target
        if args.offline:
            raise UserError(f"Offline mode: required companion {relative_filename} is missing. Download it from https://huggingface.co/{repo}/blob/main/{relative_filename} into {repo_dir}.")
        setmsg(0.30, f"Downloading required {Path(relative_filename).name} companion ...")
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(repo_id=repo, filename=relative_filename,
                                         local_dir=str(repo_dir), token=token or None)
            result = Path(downloaded)
            if not result.exists() or result.stat().st_size < minimum_size:
                raise RuntimeError(f"downloaded companion is unexpectedly small ({result.stat().st_size if result.exists() else 0} bytes)")
            print(f"Companion ready: {result} ({result.stat().st_size / 1024**2:.0f} MiB)", flush=True)
            return result
        except Exception as e:
            raise UserError(f"Could not download required companion {relative_filename} from Hugging Face ({repo}): {e}")

    def prepare_cyberrealistic_prompt(raw_prompt):
        """Prepare concise, affirmative natural-language prompts for CyberRealistic v9.

        Z-Image Turbo does not use conventional negative conditioning. Avoid putting
        unwanted concepts (collage, grid, duplicate people, etc.) into the positive
        prompt as negated text: the text encoder can still attend to those concepts.
        """
        prompt = str(raw_prompt or "").strip()
        # Remove screenplay labels and story-level production notes. In particular,
        # cross-scene continuity instructions belong to the story planner, not one frame.
        prompt = re.sub(r"(?im)^\s*(?:STORY\s+CONTINUITY|SCENE\s*#?\s*\d+)\s*:\s*", "", prompt)
        prompt = re.sub(r"(?im)^\s*(?:SHOT|FRAME)\s*#?\s*\d+\s*:\s*", "", prompt)
        prompt = re.sub(r"(?i)Maintain strict visual continuity across all scenes\.?", "", prompt)
        prompt = re.sub(r"(?i)Use the same character appearance, clothing, proportions, and recurring object designs\.?", "", prompt)
        prompt = re.sub(r"(?i)Do not add captions, lettering, subtitles, or watermarks\.?", "", prompt)
        prompt = re.sub(r"(?i)\b(?:the\s+)?same\b[^.]{0,100}\b(?:in|across)\s+(?:all|every)\s+scenes\.?", "", prompt)
        prompt = re.sub(r"\s+", " ", prompt).strip(" ,;\n")

        # Reinforce singular subject count with positive wording only. Do not inject
        # a list of forbidden objects/layouts into the prompt.
        singular_human = bool(re.search(
            r"\b(?:a|one|single|the)\s+(?:young\s+|old\s+|adult\s+)?(?:human|person|man|woman|boy|girl)\b",
            prompt, re.I))
        multiple_humans = bool(re.search(
            r"\b(?:two|three|four|five|six|several|many|multiple|crowd|group\s+of|people|men|women|children)\b",
            prompt, re.I))
        if singular_human and not multiple_humans:
            prompt += " The scene shows one person as the sole human subject."

        # Keep the prompt affirmative and photographic; avoid negated layout keywords.
        prompt += " A natural, realistic photograph with one clear camera viewpoint."
        return prompt

    def ensure_sdxl_ip_adapter(token=None):
        """Fetch the official SDXL Plus ViT-H adapter and matching CLIP-Vision-H encoder."""
        adapter_dir = MODELS_DIR / "ip_adapter_sdxl_vit_h"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        assets = (
            ("sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors", 700_000_000),
            ("models/image_encoder/model.safetensors", 2_000_000_000),
        )
        resolved = {}
        for remote_name, minimum in assets:
            target = adapter_dir / remote_name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size < minimum:
                if args.offline:
                    raise UserError(
                        "Offline mode: IP-Adapter weights are missing. Download the official SDXL Plus ViT-H "
                        f"assets into {adapter_dir} while preserving these paths: {remote_name}."
                    )
                setmsg(0.08, f"Downloading IP-Adapter component {Path(remote_name).name} (first use only; large download)…")
                try:
                    downloaded = hf_hub_download(
                        repo_id="h94/IP-Adapter", filename=remote_name,
                        local_dir=str(adapter_dir), token=token or None
                    )
                    target = Path(downloaded)
                    if not target.exists() or target.stat().st_size < minimum:
                        raise RuntimeError(f"downloaded file is too small ({target.stat().st_size if target.exists() else 0} bytes)")
                    print(f"IP-Adapter asset ready: {target} ({target.stat().st_size / 1024**3:.2f} GiB)", flush=True)
                except Exception as exc:
                    raise UserError(f"Could not download IP-Adapter component {remote_name}: {exc}")
            resolved[remote_name] = target
        return (resolved["sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors"],
                resolved["models/image_encoder/model.safetensors"])

    def run_sdcpp_image(p, spec, seed, output_path, index, count):
        exe = ensure_sd_cpp(require_zimage=bool(spec.get("components")))
        token = (p.get("token") or "").strip() or hf_token or None
        components = {}
        encoder_flag = None
        if spec.get("components"):
            # Validate backend compatibility before downloading several GB of model files.
            probe = subprocess.run([str(exe), "--help"], cwd=str(SDC_BIN_DIR), capture_output=True,
                                   text=True, timeout=30, encoding="utf-8", errors="replace",
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            help_text = (probe.stdout or "") + (probe.stderr or "")
            encoder_flag = "--llm" if "--llm" in help_text else None
            if not encoder_flag or "--vae" not in help_text:
                raise UserError("Your installed stable-diffusion.cpp build lacks the required Z-Image options (--diffusion-model, --llm, and --vae). Update the native backend to the latest official release, then retry.")
        model, quant = ensure_sdc_model(spec, p.get("quant", "Auto"), token)
        if spec.get("components"):
            for component_name, relative_filename in spec["components"].items():
                components[component_name] = ensure_sdc_component(spec, relative_filename, token)
        width, height = int(p["width"]), int(p["height"])
        width = max(512, min(1024, round(width / 64) * 64))
        height = max(512, min(1024, round(height / 64) * 64))
        steps = max(1, min(40, int(p.get("steps", spec["steps"]))))
        cfg = max(0.0, min(10.0, float(p.get("cfg", spec["cfg"]))))
        prompt = (p.get("prompt") or "").strip()
        if spec.get("architecture") == "z-image":
            prompt = prepare_cyberrealistic_prompt(prompt)
            print("CyberRealistic normalized prompt:", prompt, flush=True)
        negative = (p.get("negative") or "").strip()
        model_flag = "--diffusion-model" if spec.get("architecture") == "z-image" else "-m"
        command = [exe, "-M", "img_gen", model_flag, str(model), "-p", prompt,
                   "-o", str(output_path), "-W", str(width), "-H", str(height),
                   "--steps", str(steps), "--cfg-scale", str(cfg),
                   "--sampling-method", spec.get("sampler", "euler"), "--seed", str(seed),
                   # Keep compute on CUDA, but allow sd.cpp to place weights in system RAM
                   # and stream them as required. Auto-fit reserves VRAM for activations.
                   "--backend", "cuda0", "--offload-to-cpu", "--auto-fit", "on",
                   "--max-vram", str(max(2.0, min(4.5, torch.cuda.get_device_properties(0).total_memory / GIB - 1.2))),
                   "--mmap"]
        reference_path = p.get("reference_image_path")
        if reference_path:
            # CyberRealistic v9 is a Z-Image Turbo fine-tune, not SDXL. Use
            # the native img2img path (init image + denoising strength) rather
            # than passing incompatible SDXL IP-Adapter weights to it.
            strength = max(0.20, min(1.0, float(p.get("reference_strength", 0.55))))
            probe = subprocess.run([str(exe), "--help"], cwd=str(SDC_BIN_DIR), capture_output=True,
                                   text=True, timeout=30, encoding="utf-8", errors="replace",
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            help_text = (probe.stdout or "") + (probe.stderr or "")
            if spec.get("architecture") == "z-image":
                required_flags = ("--init-img", "--strength")
                missing = [flag for flag in required_flags if flag not in help_text]
                if missing:
                    raise UserError("Your installed stable-diffusion.cpp build does not expose Z-Image img2img options (" + ", ".join(missing) + "). Update the native backend and retry.")
                command += ["--init-img", str(reference_path), "--strength", str(strength)]
                print(f"CyberRealistic img2img enabled | transformation strength {strength:.2f} | starting photo {reference_path}", flush=True)
            else:
                required_flags = ("--clip_vision", "--ip-adapter", "--ip-adapter-image", "--ip-adapter-strength")
                missing = [flag for flag in required_flags if flag not in help_text]
                if missing:
                    raise UserError("Your installed stable-diffusion.cpp build does not expose IP-Adapter options (" + ", ".join(missing) + "). Update the native backend and retry.")
                adapter_path, vision_path = ensure_sdxl_ip_adapter(token)
                command += ["--clip_vision", str(vision_path), "--ip-adapter", str(adapter_path),
                            "--ip-adapter-image", str(reference_path), "--ip-adapter-strength", str(strength)]
                print(f"SDXL IP-Adapter enabled | strength {strength:.2f} | reference {reference_path}", flush=True)
        if components:
            command += [encoder_flag, str(components["clip_l"]), "--vae", str(components["vae"])]
        # Z-Image Turbo is distilled at CFG 1.0; negative prompts are not part of
        # the verified CyberRealistic v9 recipe and are intentionally omitted.
        if negative and not spec.get("architecture") == "z-image":
            command += ["-n", negative]
        print("Launching native Windows sd.cpp CUDA:", subprocess.list2cmdline(command), flush=True)
        setmsg((index / count), f"Image {index + 1}/{count} | starting native stable-diffusion.cpp ...")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        proc = subprocess.Popen(command, cwd=str(SDC_BIN_DIR), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                bufsize=1, creationflags=creationflags)
        with sdc_proc_lock:
            sdc_proc["process"] = proc
        started = time.time()
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    print("[sd.cpp] " + line, flush=True)
                    match = re.search(r"(?:step\s*)?(\d{1,3})\s*%", line, re.I)
                    if match:
                        pct = max(0, min(100, int(match.group(1)))) / 100
                        setmsg((index + pct) / count, f"Image {index + 1}/{count} | sd.cpp {pct:.0%} | {time.time()-started:.0f}s")
                    else:
                        step_match = re.search(r"(?:step|sample|sampling)[^0-9]*(\d+)\s*/\s*(\d+)", line, re.I)
                        if step_match and int(step_match.group(2)) > 0:
                            pct = max(0.0, min(1.0, int(step_match.group(1)) / int(step_match.group(2))))
                            setmsg((index + pct) / count, f"Image {index + 1}/{count} | sd.cpp step {step_match.group(1)}/{step_match.group(2)}")
                if job["stop"] and proc.poll() is None:
                    proc.terminate()
            code = proc.wait()
        finally:
            with sdc_proc_lock:
                sdc_proc["process"] = None
        if job["stop"]:
            return None
        if code != 0:
            raise UserError(f"Native stable-diffusion.cpp failed (exit {code}). See Terminal logs for the full error.")
        if not output_path.exists():
            matches = sorted(output_path.parent.glob(output_path.stem + "*.png"),
                             key=lambda x: x.stat().st_mtime, reverse=True)
            if matches:
                matches[0].replace(output_path)
        if not output_path.exists() or output_path.stat().st_size < 1000:
            raise UserError("sd.cpp finished without producing the expected PNG. See Terminal logs.")
        return Image.open(output_path).convert("RGB")

    def generate_sdcpp(p, spec):
        prompt = (p.get("prompt") or "").strip()
        if not prompt:
            raise UserError("Type a prompt first.")
        style_key = str(p.get("style", "none"))
        style_recipe = ART_STYLES.get(style_key, "")
        effective_prompt = f"{prompt}, {style_recipe}" if style_recipe else prompt
        generation_p = dict(p)
        generation_p["prompt"] = effective_prompt
        count = max(1, min(8, int(p.get("count", 1))))
        seed_in = int(p.get("seed", -1))
        base_seed = seed_in if seed_in >= 0 else random.randint(0, 2**31 - 1)
        steps = max(1, min(40, int(p.get("steps", spec["steps"]))))
        cfg = float(p.get("cfg", spec["cfg"]))
        started = time.time()
        selected_quant = p.get("quant", "Auto")
        selected_quant = spec.get("default_quant", "Q4_K_S") if selected_quant in (None, "", "Auto") else selected_quant
        print(f"Using native stable-diffusion.cpp CUDA | {spec['label']} | {p.get('width')}x{p.get('height')} | {steps} steps | CFG {cfg} | quant {selected_quant}", flush=True)
        for i in range(count):
            if job["stop"]:
                break
            seed = (base_seed + i) % (2**31 - 1)
            name = f"{time.strftime('%Y%m%d_%H%M%S')}_{seed}.png"
            output_path = OUTPUT_DIR / name
            img = run_sdcpp_image(generation_p, spec, seed, output_path, i, count)
            if img is None or job["stop"]:
                break
            meta = PngInfo()
            for key, value in (("prompt", prompt), ("style_preset", style_key), ("effective_prompt", effective_prompt),
                               ("negative_prompt", p.get("negative", "")), ("seed", seed), ("model", spec["label"]), ("backend", "stable-diffusion.cpp"),
                               ("steps", steps), ("cfg", cfg), ("resolution", f"{p['width']}x{p['height']}"),
                               ("quantization", selected_quant), ("compute_backend", "CUDA (native Windows)"), ("parameter_backend", "RAM offload / auto-fit"),
                               ("sampler", p.get("sampler", "")),
                               ("reference_image", bool(p.get("reference_image_path"))),
                               ("reference_strength", p.get("reference_strength", "")),
                               ("source_image", p.get("source_image_name", "")),
                               ("edit_instruction", p.get("edit_instruction", ""))):
                meta.add_text(key, str(value))
            # Rewrite with app metadata while retaining the decoded pixels.
            img.save(output_path, format="PNG", pnginfo=meta)
            with job_lock:
                job["images"].append(dict(url=f"/outputs/{name}", name=name, seed=seed))
        msg = f"{'Stopped. ' if job['stop'] else 'Done. '}{len(job['images'])} image(s) in {time.time()-started:.0f}s | {spec['label']} | native CUDA + RAM auto-offload"
        setmsg(1.0 if not job["stop"] else None, msg)

    # ----- pipeline management ---------------------------------------------- #
    def unload():
        state["pipe"] = None
        state["key"] = None
        state["plan"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def torch_dtype(prec):
        if prec == "bf16":
            raise UserError("BF16 is not supported natively by the GTX 1660. Select FP16 or Auto.")
        return torch.float32 if prec == "fp32" else torch.float16

    def vram_free_gb():
        try:
            return torch.cuda.mem_get_info()[0] / GIB
        except Exception:
            return 0.0

    def choose_level(requested, gguf_gb, te_gb, free_gb):
        if requested in MEM_LEVELS:
            return MEM_LEVELS.index(requested)
        if free_gb >= gguf_gb * 1.1 + te_gb + 1.5:
            return 0
        if free_gb >= gguf_gb * 1.1 + 1.3:
            return 1
        return 2

    def place(pipe, level, gguf_gb, free_gb, scale):
        """Put the pipeline on the GPU, offloading only what is needed. Returns a description."""
        cuda = torch.device("cuda")
        if level == 0:
            try:
                pipe.to("cuda")
                return "all on GPU"
            except Exception as e:
                print(f"pipe.to(cuda) failed ({e}); using balanced offload instead")
                level = 1
        if level == 1:
            pipe.enable_model_cpu_offload()
            return "balanced offload (transformer on GPU while sampling)"
        # partial: text encoder / VAE swap in and out; transformer blocks are paged so only what fits stays on GPU
        tr = pipe.transformer
        n_blocks = max((len(m) for m in tr.children() if isinstance(m, torch.nn.ModuleList)), default=0)
        try:
            if n_blocks < 2:
                raise RuntimeError("no block list to page")
            pipe._exclude_from_cpu_offload = ["transformer"]
            pipe.enable_model_cpu_offload()
            budget = max(free_gb - 1.6, 0.4)
            frac = min(1.0, budget / (gguf_gb * 1.1))
            per = max(1, int(n_blocks * frac * 0.6 * scale))
            tr.enable_group_offload(onload_device=cuda, offload_device=torch.device("cpu"),
                                    offload_type="block_level", num_blocks_per_group=per, use_stream=False)
            return f"partial offload ({per} of {n_blocks} transformer blocks on the GPU at a time)"
        except Exception as e:
            print(f"Partial offload failed ({e}); falling back to sequential offload")
            try:
                pipe.remove_all_hooks()
            except Exception:
                pass
            pipe._exclude_from_cpu_offload = []
            pipe.enable_sequential_cpu_offload()
            return "sequential offload (layer by layer)"

    def load_gguf(spec, choice, p, token, prec):
        if not torch.cuda.is_available():
            raise UserError("GGUF models run on the GPU, but PyTorch can't see one. Try --reinstall.")
        try:
            import gguf  # noqa: F401
            from diffusers import GGUFQuantizationConfig
        except ImportError as e:
            raise UserError(f"GGUF support is missing. Run with --reinstall to install it. ({e})")
        fam = FAMILIES[spec["family"]]
        try:
            import diffusers as dz
            if fam["kind"] == "zimage":
                pipe_cls, tr_cls = dz.ZImagePipeline, dz.ZImageTransformer2DModel
            elif fam["kind"] == "flux2":
                pipe_cls, tr_cls = dz.Flux2KleinPipeline, dz.Flux2Transformer2DModel
            else:
                pipe_cls, tr_cls = dz.FluxPipeline, dz.FluxTransformer2DModel
        except AttributeError as e:
            raise UserError(f"Your diffusers is too old for {fam['label']}. Run with --reinstall. ({e})")

        dtype = torch_dtype(prec)
        unload()  # measure free VRAM with nothing of ours loaded
        free_gb = vram_free_gb()
        req = p.get("memory", "auto")

        if "gguf_file" in spec:
            gguf_path = Path(spec["gguf_file"])
            gguf_gb = gguf_path.stat().st_size / GIB
            fname = gguf_path.name
        else:
            # budget for the best-fitting quant: transformer alone must fit (text encoder swaps out)
            budget = free_gb - 1.5 if req != "gpu" else free_gb - fam["te_gb"] - 1.2
            fname, size = pick_gguf(spec, p.get("quant"), budget, token)
            gguf_gb = size / GIB
            gguf_path = None

        level = choose_level(req, gguf_gb, fam["te_gb"], free_gb)
        level = max(level, state["esc"].get(choice, 0))
        scale = state["scale"].get(choice, 1.0)
        key = (choice, fname, prec, level, round(scale, 3))
        if state["pipe"] is not None and state["key"] == key:
            return state["pipe"], state["plan"]

        if gguf_path is None:
            gguf_path = ensure_gguf(spec["gguf_repo"], fname, token)
        base, _ = ensure_repo(fam["base"], token=token, skip_transformer=True)

        setmsg(0.93, f"Loading {fname} (GGUF) ...")
        qcfg = GGUFQuantizationConfig(compute_dtype=dtype)
        try:
            transformer = tr_cls.from_single_file(str(gguf_path), config=str(base), subfolder="transformer",
                                                  quantization_config=qcfg, torch_dtype=dtype)
        except Exception as first:
            try:  # let diffusers infer the config instead
                transformer = tr_cls.from_single_file(str(gguf_path), quantization_config=qcfg, torch_dtype=dtype)
            except Exception:
                raise UserError(f"Couldn't load the GGUF transformer: {first}")

        kw = dict(transformer=transformer, torch_dtype=dtype)
        te_bf16 = fam["te_gb"] * 3.3
        if not (free_gb >= gguf_gb * 1.1 + te_bf16 + 1.5):  # 4-bit text encoder unless VRAM is plentiful
            try:
                from diffusers import PipelineQuantizationConfig
                kw["quantization_config"] = PipelineQuantizationConfig(
                    quant_backend="bitsandbytes_4bit",
                    quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                                  "bnb_4bit_compute_dtype": dtype, "bnb_4bit_use_double_quant": True},
                    components_to_quantize=fam["quant_components"])
            except ImportError as e:
                raise UserError(f"This diffusers has no PipelineQuantizationConfig; run with --reinstall. ({e})")
        setmsg(0.96, "Loading text encoder and VAE ...")
        pipe = pipe_cls.from_pretrained(str(base), **kw)
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
        plan = place(pipe, level, gguf_gb, free_gb, scale)
        plan = f"{Path(fname).name} | {plan}"
        pipe._lg_level = level
        state["pipe"], state["key"], state["plan"] = pipe, key, plan
        return pipe, plan

    def load_sd(spec, prec, token=None):
        sdxl = spec["kind"] in ("sdxl", "sdxl-lightning")
        dtype = torch_dtype(prec)
        cls = StableDiffusionXLPipeline if sdxl else StableDiffusionPipeline
        extra = {}
        if not sdxl:
            extra.update(safety_checker=None, requires_safety_checker=False)

        if spec["kind"] == "sdxl-lightning":
            # The Lightning repo is a distilled UNet checkpoint, not a standalone pipeline.
            from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
            from safetensors.torch import load_file
            base_path, _ = ensure_repo(spec["repo"], token=token, fp16_variant=True)
            setmsg(0.88, "Loading SDXL VAE in FP32 to prevent FP16 decode overflow...")
            vae_kw = {"torch_dtype": torch.float32}
            if (base_path / "vae" / "diffusion_pytorch_model.fp16.safetensors").exists():
                vae_kw["variant"] = "fp16"
            vae = AutoencoderKL.from_pretrained(str(base_path), subfolder="vae", **vae_kw)
            setmsg(0.90, f"Preparing SDXL Lightning {int(spec.get('lightning_steps', 4))}-step UNet...")
            unet = UNet2DConditionModel.from_config(str(base_path / "unet"))
            checkpoint = hf_hub_download(
                repo_id="ByteDance/SDXL-Lightning",
                filename=f"sdxl_lightning_{int(spec.get('lightning_steps', 4))}step_unet.safetensors",
                token=token or None,
                cache_dir=str(ROOT / "hf_cache"),
            )
            weights = load_file(checkpoint, device="cpu")
            missing, unexpected = unet.load_state_dict(weights, strict=True)
            del weights
            if missing or unexpected:
                raise UserError(f"Lightning UNet weights did not match (missing={len(missing)}, unexpected={len(unexpected)}). Update diffusers and retry.")
            unet.to(dtype=dtype)
            setmsg(0.95, "Loading SDXL base components...")
            kw = dict(unet=unet, vae=vae, torch_dtype=dtype, use_safetensors=True)
            if (base_path / "unet" / "diffusion_pytorch_model.fp16.safetensors").exists():
                kw["variant"] = "fp16"
            pipe = StableDiffusionXLPipeline.from_pretrained(str(base_path), **kw)
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config, timestep_spacing="trailing")
        else:
            if "repo" in spec:
                path, has_fp16 = ensure_repo(spec["repo"], token=token, fp16_variant=True)
                setmsg(0.94, "Loading model into memory...")
                kw = dict(torch_dtype=dtype, use_safetensors=True, **extra)
                if has_fp16:
                    kw["variant"] = "fp16"
                if sdxl:
                    from diffusers import AutoencoderKL
                    setmsg(0.91, "Loading SDXL VAE in FP32 to prevent FP16 decode overflow...")
                    vae_kw = {"torch_dtype": torch.float32}
                    if (path / "vae" / "diffusion_pytorch_model.fp16.safetensors").exists():
                        vae_kw["variant"] = "fp16"
                    kw["vae"] = AutoencoderKL.from_pretrained(
                        str(path), subfolder="vae", **vae_kw
                    )
                pipe = cls.from_pretrained(str(path), **kw)
            else:
                setmsg(0.5, "Loading checkpoint file into memory...")
                pipe = cls.from_single_file(spec["file"], torch_dtype=dtype, **extra)

        pipe._lg_cfg = pipe.scheduler.config
        try:
            pipe.enable_attention_slicing("max")
        except Exception:
            pipe.enable_attention_slicing()
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass
        # Keep the SDXL VAE in FP32. Passing torch_dtype=FP16 to the pipeline
        # constructor can cast an explicitly supplied FP32 VAE back to FP16.
        # Re-assert FP32 AFTER pipeline construction, before installing offload hooks.
        # SDXL's UNet/text encoders remain FP16 to limit GTX 1660 VRAM use.
        if sdxl:
            pipe.vae.to(dtype=torch.float32)
            vae_dtypes = {p.dtype for p in pipe.vae.parameters()}
            if vae_dtypes != {torch.float32}:
                raise UserError(f"SDXL VAE did not remain FP32 (found dtypes: {vae_dtypes}).")

            # SDXL pipeline may pass FP16 latents to an FP32 VAE. Convert only
            # the decoder input; all VAE parameters remain consistently FP32.
            try:
                def _cast_vae_input_to_param_dtype(module, inputs):
                    if inputs and torch.is_tensor(inputs[0]):
                        target_dtype = module.weight.dtype
                        if inputs[0].dtype != target_dtype:
                            return (inputs[0].to(dtype=target_dtype), *inputs[1:])
                    return inputs
                pipe.vae.post_quant_conv.register_forward_pre_hook(
                    _cast_vae_input_to_param_dtype
                )
            except Exception as e:
                print(f"Could not install SDXL VAE dtype compatibility hook: {e}")
            print("SDXL VAE parameter dtype:", sorted(str(d) for d in vae_dtypes))

            # Temporary diagnostics: identify whether non-finite values originate
            # in the denoising latents or during VAE decoding. This wrapper does
            # not modify tensors or their dtypes.
            _original_vae_decode = pipe.vae.decode
            def _diagnostic_vae_decode(z, *args, **kwargs):
                try:
                    z_finite = bool(torch.isfinite(z).all().item())
                    z_min = float(torch.nan_to_num(z.detach()).min().item())
                    z_max = float(torch.nan_to_num(z.detach()).max().item())
                    print(f"SDXL diagnostic | VAE input: dtype={z.dtype}, finite={z_finite}, range=[{z_min:.6g}, {z_max:.6g}]")
                except Exception as diag_error:
                    print(f"SDXL diagnostic | VAE input check failed: {diag_error}")
                decoded = _original_vae_decode(z, *args, **kwargs)
                try:
                    sample = decoded[0] if isinstance(decoded, tuple) else decoded.sample
                    finite = bool(torch.isfinite(sample).all().item())
                    clean = torch.nan_to_num(sample.detach())
                    print(f"SDXL diagnostic | VAE output: dtype={sample.dtype}, finite={finite}, range=[{float(clean.min().item()):.6g}, {float(clean.max().item()):.6g}]")
                except Exception as diag_error:
                    print(f"SDXL diagnostic | VAE output check failed: {diag_error}")
                return decoded
            pipe.vae.decode = _diagnostic_vae_decode
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        return pipe

    def load_pipe(choice, p, token):
        spec = get_spec(choice, p.get("force_sdxl", False), p.get("family", "auto"))
        prec = p.get("precision", "auto")
        prec = prec if prec != "auto" else default_precision(spec)
        prec = state["force_prec"].get(choice, prec)
        if spec["kind"] == "gguf":
            pipe, plan = load_gguf(spec, choice, p, token, prec)
            pipe.set_progress_bar_config(disable=True)
            return pipe, spec, prec, plan
        key = (choice, spec["kind"], prec)
        if state["pipe"] is not None and state["key"] == key:
            return state["pipe"], spec, prec, state["plan"]
        unload()
        pipe = load_sd(spec, prec, token)
        pipe.set_progress_bar_config(disable=True)
        state["pipe"], state["key"], state["plan"] = pipe, key, "on GPU"
        return pipe, spec, prec, "on GPU"

    def looks_black(img):
        return max(img.convert("L").getextrema()) < 8

    # ----- generation ------------------------------------------------------- #
    def generate(p):
        prompt = (p.get("prompt") or "").strip()
        if not prompt:
            raise UserError("Type a prompt first.")
        choice = p.get("model", "")
        steps, count = int(p["steps"]), max(1, min(8, int(p["count"])))
        width, height = int(p["width"]), int(p["height"])
        cfg = float(p["cfg"])
        negative = p.get("negative") or ""
        sampler = p.get("sampler") if p.get("sampler") in SAMPLERS else "Euler a"
        token = (p.get("token") or "").strip() or hf_token or None
        t_all = time.time()
        notes = []

        try:
            selected_spec = get_spec(choice, p.get("force_sdxl", False), p.get("family", "auto"))
            if selected_spec["kind"] == "sdcpp":
                return generate_sdcpp(p, selected_spec)
            if p.get("reference_image_path"):
                raise UserError("Story reference conditioning is currently available for native SDXL GGUF models only. Select an [sd.cpp] SDXL model, or remove the reference image.")
            pipe, spec, prec, plan = load_pipe(choice, p, token)
            sampler = spec.get("recommended_sampler") or sampler
        except UserError:
            raise
        except torch.cuda.OutOfMemoryError:
            unload()
            raise UserError("Out of GPU memory while loading. Set Memory to 'Partial offload' or pick a smaller "
                            "quantization.")
        except Exception as e:
            unload()
            traceback.print_exc()
            raise UserError(f"Could not load the model: {e}")
        notes.append(plan)

        try:
            seed_in = int(p.get("seed", -1))
        except Exception:
            seed_in = -1
        base_seed = seed_in if seed_in >= 0 else random.randint(0, 2 ** 32 - 1)
        torch.cuda.reset_peak_memory_stats()

        for i in range(count):
            if job["stop"]:
                break
            s = (base_seed + i) % (2 ** 32)
            clock = [time.time()]

            def on_step(pp, step, timestep, kw, i=i, clock=clock):
                latents = kw.get("latents") if isinstance(kw, dict) else None
                if torch.is_tensor(latents):
                    try:
                        finite = bool(torch.isfinite(latents).all().item())
                        clean = torch.nan_to_num(latents.detach())
                        print(f"SDXL diagnostic | denoise step {step + 1}: finite={finite}, range=[{float(clean.min().item()):.6g}, {float(clean.max().item()):.6g}]")
                    except Exception as diag_error:
                        print(f"SDXL diagnostic | latent check failed: {diag_error}")
                done = step + 1
                el = max(time.time() - clock[0], 1e-6)
                rate = done / el
                eta = (steps - done) / rate if rate else 0
                setmsg((i + done / steps) / count,
                       f"Image {i + 1}/{count} | step {done}/{steps} | {rate:.2f} it/s | ~{eta:.0f}s left")
                if job["stop"]:
                    pp._interrupt = True
                return kw

            def render(pp, spec, s=s, on_step=on_step, clock=clock):
                clock[0] = time.time()
                gen = torch.Generator("cuda").manual_seed(s)
                call = dict(prompt=prompt, num_inference_steps=steps, guidance_scale=cfg, width=width,
                            height=height, generator=gen, callback_on_step_end=on_step)
                neg = negative.strip()
                if spec["kind"] == "sdxl-lightning":
                    # Lightning 4-step requires Euler with trailing timestep spacing and CFG=0.
                    call["guidance_scale"] = 0.0
                elif spec["kind"] in ("sd", "sdxl"):
                    try:
                        pp.scheduler = SAMPLERS[sampler](pp._lg_cfg)
                    except Exception as e:
                        print(f"Sampler '{sampler}' failed ({e}); falling back to Euler a.")
                        pp.scheduler = SAMPLERS["Euler a"](pp._lg_cfg)
                    call["negative_prompt"] = neg or None
                elif spec.get("family") == "zimage" and neg and cfg > 1:
                    call["negative_prompt"] = neg
                pp._interrupt = False
                setmsg(i / count, f"Image {i + 1}/{count} | encoding prompt, starting ...")
                try:
                    return pp(**call).images[0]
                except TypeError as e:
                    if "callback_on_step_end" in str(e):
                        call.pop("callback_on_step_end")
                        return pp(**call).images[0]
                    raise

            attempts = 0
            while True:
                try:
                    img = render(pipe, spec)
                    if not job["stop"] and looks_black(img):
                        # Do not try BF16 on Turing/GTX 1660; only classic SD pipelines get an FP32 retry.
                        retry = "fp32" if prec == "fp16" and spec["kind"] in ("sd", "sdxl") else None
                        if retry:
                            setmsg(i / count, f"Black image with {prec}. Reloading in {retry} ...")
                            state["force_prec"][choice] = retry
                            notes.append(f"switched to {retry} automatically")
                            pipe, spec, prec, plan = load_pipe(choice, p, token)
                            img = render(pipe, spec)
                    break
                except torch.cuda.OutOfMemoryError:
                    attempts += 1
                    if spec["kind"] == "gguf":
                        if attempts > 3:
                            unload()
                            raise UserError("Out of GPU memory. Lower resolution or select a smaller GGUF quantization.")
                        lvl = getattr(pipe, "_lg_level", 0)
                        if lvl < 2:
                            state["esc"][choice] = lvl + 1
                        else:
                            state["scale"][choice] = state["scale"].get(choice, 1.0) * 0.5
                        setmsg(i / count, "Out of GPU memory, offloading more to CPU and retrying ...")
                    else:
                        if attempts > 2 or min(width, height) <= 640:
                            unload()
                            raise UserError("Out of GPU memory even after automatic downscaling. Try 640px or choose DreamShaper 8.")
                        new_w = max(640, int((width * 0.75) // 64) * 64)
                        new_h = max(640, int((height * 0.75) // 64) * 64)
                        width, height = new_w, new_h
                        setmsg(i / count, f"VRAM limit reached; retrying at {width}x{height}...")
                    unload()
                    try:
                        pipe, spec, prec, plan = load_pipe(choice, p, token)
                    except UserError:
                        raise
                    except Exception as e:
                        unload()
                        raise UserError(f"Could not reload after memory adjustment: {e}")
                    notes.append(f"memory fallback at {width}x{height} -> {plan}")
                except UserError:
                    raise
                except Exception as e:
                    traceback.print_exc()
                    raise UserError(f"Generation failed: {e}")

            if job["stop"]:
                break

            meta = PngInfo()
            for k, v in (("prompt", prompt), ("negative_prompt", negative), ("seed", s), ("model", choice),
                         ("steps", steps), ("cfg", cfg), ("resolution", f"{width}x{height}"), ("precision", prec),
                         ("memory_plan", plan)):
                meta.add_text(k, str(v))
            name = f"{time.strftime('%Y%m%d_%H%M%S')}_{s}.png"
            img.save(OUTPUT_DIR / name, pnginfo=meta)
            with job_lock:
                job["images"].append(dict(url=f"/outputs/{name}", name=name, seed=s))

        peak = torch.cuda.max_memory_allocated() / GIB
        msg = (f"{'Stopped. ' if job['stop'] else 'Done. '}{len(job['images'])} image(s) in "
               f"{time.time() - t_all:.0f}s | peak VRAM {peak:.1f} GB | {'; '.join(dict.fromkeys(notes))}")
        setmsg(1.0 if not job["stop"] else None, msg)
        torch.cuda.empty_cache()

    def job_thread(p):
        try:
            generate(p)
        except UserError as e:
            with job_lock:
                job["error"] = str(e)
        except Exception as e:
            traceback.print_exc()
            with job_lock:
                job["error"] = f"Unexpected error: {e}"
        finally:
            with job_lock:
                job["running"] = False

    # ----- HTTP API --------------------------------------------------------- #
    THEMES = {"Abyssal Storm": "abyssal", "Forest Moss": "forest", "Cocoa Zen": "cocoa",
              "Complete Dark": "dark", "Light Parchment": "light"}
    default_theme = {"dark": "abyssal", "light": "light", "auto": "auto"}.get(getattr(args, "theme", "dark"), "abyssal")
    first = G_SDC + SDC_DEFAULT_LABEL

    try:
        _prop = torch.cuda.get_device_properties(0)
        _cap = torch.cuda.get_device_capability(0)
        gpu_label = f"{_prop.name} | {_prop.total_memory / GIB:.0f} GB | SM {_cap[0]}.{_cap[1]}"
        if _cap < (8, 0):
            print("Turing/GTX 1660 detected: native stable-diffusion.cpp GGUF backend; CUDA compute with RAM auto-fit enabled.")
    except Exception:
        gpu_label = "no GPU detected"

    bs_link = ('<link rel="stylesheet" href="/static/bootstrap.min.css">' if have_bootstrap else
               f'<link rel="stylesheet" href="{BOOTSTRAP_URL}">')
    page = (INDEX_HTML.replace("__BSLINK__", bs_link).replace("__GPU__", htmlmod.escape(gpu_label))
            .replace("__DEFAULT_THEME__", default_theme)).encode("utf-8")

    def history_items(limit=60):
        out = []
        files = sorted(OUTPUT_DIR.glob("*.png"), key=lambda f: f.stat().st_mtime, reverse=True)[:limit]
        for f in files:
            prompt, seed, model = "", "", ""
            try:
                with Image.open(f) as im:
                    prompt, seed = im.text.get("prompt", ""), im.text.get("seed", "")
                    model = im.text.get("model", "")
            except Exception:
                pass
            out.append(dict(url=f"/outputs/{f.name}", name=f.name, prompt=prompt, seed=seed, model=model))
        return out

    class Handler(BaseHTTPRequestHandler):
        server_version = "ImageGenLocal"

        def log_message(self, *a):
            pass

        def _host_ok(self):
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            return host in ("127.0.0.1", "localhost", "::1")

        def _send(self, code, body, ctype, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

        def _file(self, path: Path, ctype):
            if not path.is_file():
                return self._send(404, b"not found", "text/plain")
            data = path.read_bytes()
            self._send(200, data, ctype, {"Cache-Control": "max-age=3600"})

        def do_GET(self):
            if not self._host_ok():
                return self._send(403, b"forbidden", "text/plain")
            u = urlparse(self.path)
            route = u.path
            if route == "/":
                return self._send(200, page, "text/html; charset=utf-8")
            if route == "/static/bootstrap.min.css":
                return self._file(STATIC_DIR / "bootstrap.min.css", "text/css; charset=utf-8")
            if route.startswith("/outputs/"):
                return self._file(OUTPUT_DIR / Path(route[len("/outputs/"):]).name, "image/png")
            if route == "/api/config":
                return self._json(dict(
                    models=model_list(), default_model=first, themes=THEMES, quants=["Auto", "Q8_0", "Q4_K_S"],
                    samplers=list(SAMPLERS), default_negative=DEFAULT_NEG, has_token=bool(hf_token),
                    families=[dict(label="Auto (guess from file name)", value="auto")] +
                             [dict(label=f["label"], value=k) for k, f in FAMILIES.items()]))
            if route == "/api/models":
                return self._json(dict(models=model_list()))
            if route == "/api/modelinfo":
                m = (parse_qs(u.query).get("model") or [""])[0]
                try:
                    s = get_spec(m)
                except UserError as e:
                    return self._json(dict(error=str(e)))
                except Exception as e:
                    return self._json(dict(error=str(e)))
                return self._json(dict(info=re.sub(r"\*\*", "", s["info"]), steps=s["steps"], cfg=s["cfg"],
                                       size=s["size"], gguf=s["gguf"], local_gguf=s["local_gguf"], kind=s["kind"],
                                       quality_rating=s.get("quality_rating"), speed_rating=s.get("speed_rating"),
                                       rating_note=s.get("rating_note"),
                                       quants=["Auto"] + list(s.get("files", {}).keys())))
            if route == "/api/status":
                with job_lock:
                    return self._json(dict(job))
            if route == "/api/logs":
                with log_lock:
                    captured = "".join(log_chunks)
                return self._json(dict(logs=captured))
            if route == "/api/history":
                return self._json(dict(items=history_items()))
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if not self._host_ok():
                return self._send(403, b"forbidden", "text/plain")
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                return self._send(415, b"json only", "text/plain")
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(min(n, 20_000_000)) or b"{}")
            except Exception:
                return self._json(dict(error="Bad request."), 400)
            route = urlparse(self.path).path
            if route == "/api/history/edit":
                with job_lock:
                    if job["running"]:
                        return self._json(dict(error="A generation is already running. Stop it or wait for it to finish."), 409)
                filename = str(body.get("name") or "")
                if not filename or Path(filename).name != filename or not filename.lower().endswith(".png"):
                    return self._json(dict(error="Choose a valid image from History."), 400)
                source_path = OUTPUT_DIR / filename
                if not source_path.is_file():
                    return self._json(dict(error="That history image no longer exists. Refresh History and select it again."), 404)
                instruction = str(body.get("instruction") or "").strip()
                if not instruction:
                    return self._json(dict(error="Describe what you want to change."), 400)
                if len(instruction) > 1600:
                    return self._json(dict(error="Keep the edit instruction under 1,600 characters."), 400)
                try:
                    strength = max(0.12, min(0.65, float(body.get("strength", 0.28))))
                    with Image.open(source_path) as source_image:
                        source_image.verify()
                    with Image.open(source_path) as source_image:
                        source_width, source_height = source_image.size
                        source_meta = dict(source_image.text)
                    if source_width < 1 or source_height < 1 or source_width * source_height > 25_000_000:
                        return self._json(dict(error="The selected image dimensions are not supported."), 400)
                except Exception as exc:
                    return self._json(dict(error=f"Could not read the selected history image: {exc}"), 400)
                cyber_key = next((name for name in SDC_MODELS if name.startswith("03 | CyberRealistic v9 |")), None)
                if not cyber_key:
                    return self._json(dict(error="CyberRealistic v9 is not available in the current model catalog."), 500)
                # Fit the source to the model's 1024px generation limit without changing its aspect ratio materially.
                fit = min(1.0, 1024.0 / max(source_width, source_height))
                edit_width = max(512, min(1024, int(round((source_width * fit) / 64.0) * 64)))
                edit_height = max(512, min(1024, int(round((source_height * fit) / 64.0) * 64)))
                original_description = (source_meta.get("effective_prompt") or source_meta.get("prompt") or "").strip()
                original_description = original_description[:1200]
                edit_prompt = (
                    "Preserve the source image as the visual truth. Keep its existing visual style, rendering method, "
                    "colour palette, lighting, texture, camera viewpoint, perspective, composition, atmosphere, "
                    "subject identity and all unmentioned objects and details unchanged. "
                    + ("Original image description: " + original_description + ". " if original_description else "")
                    + "Apply only this requested change: " + instruction + ". "
                    "Do not restyle, redesign, reframe, or alter anything that was not requested."
                )
                edit_payload = {
                    "prompt": edit_prompt,
                    "negative": "",
                    "model": G_SDC + cyber_key,
                    "style": "none",
                    "family": "auto",
                    "quant": "Q4_K",
                    "force_sdxl": False,
                    "sampler": "Euler a",
                    "steps": 8,
                    "cfg": 1.0,
                    "width": edit_width,
                    "height": edit_height,
                    "count": 1,
                    "seed": -1,
                    "precision": "auto",
                    "token": hf_token or "",
                    "reference_image_path": str(source_path),
                    "reference_strength": strength,
                    "source_image_name": filename,
                    "edit_instruction": instruction,
                }
                with job_lock:
                    job.update(running=True, progress=0.0, message="Starting CyberRealistic history edit…",
                               images=[], error=None, stop=False)
                try:
                    threading.Thread(target=job_thread, args=(edit_payload,), daemon=True).start()
                except Exception as exc:
                    with job_lock:
                        job["running"] = False
                    return self._json(dict(error=f"Could not start the edit: {exc}"), 500)
                return self._json(dict(ok=True, model=cyber_key, strength=strength))
            if route == "/api/story/parse":
                try:
                    story = str(body.get("story") or "").strip()
                    if not story:
                        return self._json(dict(error="Enter a story first."), 400)
                    raw_scenes = parse_story_scenes(story)
                    scenes = simplify_story_scenes(raw_scenes)
                    entities = extract_story_entities(story)
                    return self._json(dict(scenes=scenes, entities=entities))
                except Exception as exc:
                    return self._json(dict(error=f"Could not parse story: {exc}"), 400)
            if route == "/api/story/generate":
                with job_lock:
                    if job["running"]:
                        return self._json(dict(error="A generation is already running."))
                scenes = body.get("scenes") or []
                if not isinstance(scenes, list) or not scenes:
                    return self._json(dict(error="No scenes were provided."))
                if len(scenes) > 40:
                    return self._json(dict(error="Please keep a story to 40 scenes or fewer."))
                entities = body.get("entities") or []
                if not isinstance(entities, list):
                    entities = []
                reference_image_path = None
                reference_data = body.get("reference_image") or ""
                if reference_data:
                    if not isinstance(reference_data, str) or not reference_data.startswith("data:image/jpeg;base64,"):
                        return self._json(dict(error="Invalid reference image. Upload a PNG, JPG or WebP image."), 400)
                    try:
                        encoded = reference_data.split(",", 1)[1]
                        if len(encoded) > 12_000_000:
                            return self._json(dict(error="Compressed reference image is too large."), 413)
                        raw_image = base64.b64decode(encoded, validate=True)
                        with Image.open(io.BytesIO(raw_image)) as check:
                            if check.format not in ("JPEG", "PNG", "WEBP"):
                                raise ValueError("unsupported image format")
                            if check.width * check.height > 20_000_000:
                                raise ValueError("image dimensions are too large")
                            check.verify()
                        with Image.open(io.BytesIO(raw_image)) as source_image:
                            clean_image = source_image.convert("RGB")
                            reference_image_path = STORY_REF_DIR / (time.strftime("%Y%m%d_%H%M%S_") + os.urandom(5).hex() + ".jpg")
                            clean_image.save(reference_image_path, "JPEG", quality=94, optimize=True)
                    except (ValueError, binascii.Error, OSError) as exc:
                        return self._json(dict(error=f"Could not process reference image: {exc}"), 400)
                common = (
                    "Maintain strict visual continuity across all scenes. "
                    "Use the same character appearance, clothing, proportions, and recurring object designs. "
                    "Do not add captions, lettering, subtitles, or watermarks."
                )
                entity_text = "\n".join(str(x) for x in entities if str(x).strip())
                prompts = []
                for index, scene_text in enumerate(scenes):
                    scene_text = str(scene_text or "").strip()
                    if not scene_text:
                        continue
                    prompt = "\n\n".join(x for x in (
                        "STORY CONTINUITY:\n" + entity_text if entity_text else "",
                        common,
                        f"SCENE {index + 1}:\n{scene_text}"
                    ) if x)
                    prompts.append(prompt)
                if not prompts:
                    return self._json(dict(error="All scene prompts are empty."))
                generation = {
                    "scenes": prompts,
                    "model": body.get("model", first),
                    "style": body.get("style", "none"),
                    "negative": body.get("negative", DEFAULT_NEG),
                    "width": int(body.get("width", 768)),
                    "height": int(body.get("height", 768)),
                    "steps": int(body.get("steps", 24)),
                    "cfg": float(body.get("cfg", 5.0)),
                    "count": max(1, min(8, int(body.get("count", 1)))),
                    "seed": int(body.get("seed", -1)),
                    "quant": body.get("quant", "Auto"),
                    "precision": body.get("precision", "auto"),
                    "token": body.get("token", ""),
                    "family": body.get("family", "auto"),
                    "force_sdxl": bool(body.get("force_sdxl", False)),
                    "sampler": body.get("sampler", "Euler a"),
                    "reference_image_path": str(reference_image_path) if reference_image_path else None,
                    "reference_strength": max(0.20, min(1.0, float(body.get("reference_strength", 0.55)))),
                }
                def story_batch():
                    try:
                        for index, prompt in enumerate(generation["scenes"]):
                            if job.get("stop"):
                                break
                            scene_payload = dict(generation)
                            scene_payload.pop("scenes", None)
                            scene_payload["prompt"] = prompt
                            with job_lock:
                                job.update(progress=0.0, message=f"Story: generating scene {index + 1} of {len(prompts)}",
                                           error=None)
                            generate(scene_payload)
                        with job_lock:
                            if not job.get("stop"):
                                job["message"] = f"Finished {len(prompts)} story scenes."
                    except UserError as exc:
                        with job_lock:
                            job["error"] = str(exc)
                    except Exception as exc:
                        traceback.print_exc()
                        with job_lock:
                            job["error"] = f"Story generation failed: {exc}"
                    finally:
                        with job_lock:
                            job["running"] = False
                with job_lock:
                    job.update(running=True, progress=0.0, message="Starting story generation…",
                               images=[], error=None, stop=False)
                threading.Thread(target=story_batch, daemon=True).start()
                return self._json(dict(ok=True, scene_count=len(prompts)))
            if route == "/api/generate":
                with job_lock:
                    if job["running"]:
                        return self._json(dict(error="A generation is already running."))
                    job.update(running=True, progress=0.0, message="Starting ...", images=[], error=None, stop=False)
                try:
                    threading.Thread(target=job_thread, args=(body,), daemon=True).start()
                except Exception as e:
                    with job_lock:
                        job["running"] = False
                    return self._json(dict(error=str(e)))
                return self._json(dict(ok=True))
            if route == "/api/stop":
                job["stop"] = True
                with sdc_proc_lock:
                    active = sdc_proc.get("process")
                    if active is not None and active.poll() is None:
                        try:
                            active.terminate()
                        except Exception:
                            pass
                return self._json(dict(message="Stopping generation ..."))
            if route == "/api/free":
                if job["running"]:
                    return self._json(dict(message="Can't free VRAM while generating."))
                unload()
                return self._json(dict(message="Model unloaded, VRAM freed."))
            if route == "/api/open":
                try:
                    if os.name == "nt":
                        os.startfile(str(OUTPUT_DIR))  # type: ignore[attr-defined]
                    elif sys.platform == "darwin":
                        subprocess.Popen(["open", str(OUTPUT_DIR)])
                    else:
                        subprocess.Popen(["xdg-open", str(OUTPUT_DIR)])
                    return self._json(dict(message=f"Opened {OUTPUT_DIR}"))
                except Exception as e:
                    return self._json(dict(message=f"Couldn't open folder: {e}"))
            return self._send(404, b"not found", "text/plain")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Image Gen Local running at {url}  (Ctrl+C to quit)")
    if not args.no_browser:
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBye.")
    finally:
        server.server_close()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Image Gen Local - local image generation UI")
    ap.add_argument("--offline", action="store_true", help="never contact the internet")
    ap.add_argument("--reinstall", action="store_true", help="rebuild the private environment")
    ap.add_argument("--setup-only", action="store_true", help="install only, don't launch")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--theme", choices=["dark", "light", "auto"], default="dark",
                    help="default look (auto follows your system); colour themes are in the UI")
    args = ap.parse_args()

    inside_venv = Path(sys.prefix).resolve() == VENV_DIR.resolve()
    if not inside_venv:
        if args.reinstall or not is_setup():
            setup(force=args.reinstall)
        if args.setup_only:
            return
        print("Launching Image Gen Local...")
        sys.exit(subprocess.call([str(venv_python()), str(Path(__file__).resolve()), *sys.argv[1:]]))
    run_app(args)


if __name__ == "__main__":
    main()
