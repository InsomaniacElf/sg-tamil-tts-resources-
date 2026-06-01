"""
Singapore Tamil TTS Resources – Daily Update Script
====================================================
Outputs:
  README.md     – Papers | Repos | Datasets | Articles  (newest first)
  tamil-tts.md  – Tamil TTS Papers | Low-Resource Papers | Tamil Models | Tamil Datasets
  repos.md      – TTS Repos | Low-Resource Repos | Tamil Repos

First run (papers.json empty):  backfill from 2019-01-01
Daily runs:                      last 7 days

LLM check (Gemini, title+abstract):
  is_tamil   → tamil-tts.md §Tamil TTS Papers
  is_low_res → tamil-tts.md §Low-Resource
Papers appear in BOTH README and tamil-tts.md as appropriate.
"""

import os, json, re, time, yaml, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests
import arxiv

# ── Paths ──────────────────────────────────────────────────────────────────
BASE          = Path(__file__).parent.parent.parent
CONFIG_FILE   = BASE / ".github/scripts/config.yaml"
DATA_PAPERS   = BASE / "data/papers.json"
DATA_REPOS    = BASE / "data/repos.json"
DATA_DATASETS = BASE / "data/datasets.json"
DATA_ARTICLES = BASE / "data/articles.json"
DATA_MODELS   = BASE / "data/models.json"
README_FILE   = BASE / "README.md"
TAMIL_FILE    = BASE / "tamil-tts.md"
REPOS_FILE    = BASE / "repos.md"

# ── Environment ────────────────────────────────────────────────────────────
GITHUB_TOKEN   = os.environ.get("GITHUB_TOKEN", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
HEADERS_GH     = {"Authorization": f"Bearer {GITHUB_TOKEN}",
                  "Accept": "application/vnd.github+json"} if GITHUB_TOKEN else {}
today_str      = datetime.now(timezone.utc).strftime("%Y-%m-%d")

_gemini_calls = 0   # global counter

# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

def load_config() -> dict:
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_json(path) -> list:
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 2:
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8")) or []
    except json.JSONDecodeError:
        return []

def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

def is_dup(existing: list, uid: str) -> bool:
    return any(e.get("id") == uid for e in existing)

def fmt_authors(authors) -> str:
    if isinstance(authors, list):
        try:
            names = [a.name for a in authors]
        except AttributeError:
            names = [str(a) for a in authors]
    else:
        names = [a.strip() for a in str(authors).split(",") if a.strip()]
    if len(names) > 3:
        return ", ".join(names[:3]) + " et al."
    return ", ".join(names)

def extract_gh_link(text: str) -> str | None:
    m = re.search(r"https?://github\.com/[\w\-./]+", text or "")
    return m.group(0) if m else None

def make_entry(uid, title, authors, published, pdf, code, etype, abstract="") -> dict:
    return {
        "id":         uid,
        "title":      title,
        "authors":    authors,
        "published":  published,
        "pdf":        pdf   or "null",
        "code":       code  or "null",
        "type":       etype,
        "abstract":   abstract[:800],
        "date_added": today_str,
        "is_tamil":   None,   # filled by LLM / keyword
        "is_low_res": None,
    }

# ══════════════════════════════════════════════════════════════════════════
#  GEMINI LLM TAGGER
# ══════════════════════════════════════════════════════════════════════════

def _gemini_call(prompt: str) -> str:
    global _gemini_calls
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 30, "temperature": 0.0},
    }
    for attempt in range(3):
        try:
            r = requests.post(url, json=body, timeout=20)
            _gemini_calls += 1
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip().upper()
            if r.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"    [Gemini] 429 rate limit – sleeping {wait}s")
                time.sleep(wait)
            else:
                print(f"    [Gemini] HTTP {r.status_code}: {r.text[:120]}")
                break
        except Exception as e:
            print(f"    [Gemini] Error: {e}")
            time.sleep(5)
    return "ERROR"

def llm_tag(title: str, abstract: str) -> tuple[bool, bool]:
    """Return (is_tamil, is_low_resource).  Gemini if key set, else keywords."""
    text = f"Title: {title}\nAbstract: {abstract[:1200]}"

    if GEMINI_API_KEY:
        prompt = (
            "You are classifying NLP/speech research papers for a Singapore Tamil TTS project.\n"
            "Read the title and abstract and answer TWO questions:\n\n"
            "Q1 (TAMIL): Is this paper about Tamil TTS, Tamil speech synthesis, "
            "Tamil speech corpus, multilingual TTS that explicitly includes Tamil "
            "(language code 'ta' or 'tam'), code-switching TTS with Tamil, "
            "or Dravidian-language TTS?  Answer YES or NO.\n\n"
            "Q2 (LOWRES): Is this paper about low-resource speech synthesis, "
            "data-efficient TTS, zero-shot or few-shot TTS for unseen/under-resourced "
            "languages, or minimal-data voice cloning?  Answer YES or NO.\n\n"
            "Reply in EXACTLY this format (two lines only):\n"
            "TAMIL: YES\n"
            "LOWRES: NO\n\n"
            f"{text}"
        )
        ans    = _gemini_call(prompt)
        tamil  = "TAMIL: YES"  in ans
        lowres = "LOWRES: YES" in ans
        time.sleep(2)   # free tier ~15 RPM
        return tamil, lowres

    # ── keyword fallback (no API key) ─────────────────────────────────────
    t = (title + " " + abstract).lower()
    tamil_kw = ["tamil", " ta ", '"ta"', "dravidian", "singapore tamil",
                "indictts", "indic tts", "tamil speech", "tamil tts", "tam "]
    low_kw   = ["low-resource", "low resource", "under-resourced",
                "zero-shot", "few-shot", "data-efficient", "minimal data"]
    tts_kw   = ["tts", "text-to-speech", "speech synthesis", "voice cloning",
                "speech generation"]
    is_t = any(k in t for k in tamil_kw) and any(k in t for k in tts_kw)
    is_l = any(k in t for k in low_kw)
    return is_t, is_l

def tag_all_untagged(entries: list[dict]) -> list[dict]:
    need = [e for e in entries if e.get("is_tamil") is None]
    if not need:
        return entries
    print(f"  Tagging {len(need)} un-tagged entries via LLM…")
    for i, e in enumerate(need, 1):
        t, l = llm_tag(e.get("title",""), e.get("abstract",""))
        e["is_tamil"]   = t
        e["is_low_res"] = l
        if i % 50 == 0:
            print(f"    …tagged {i}/{len(need)}")
    return entries

# ══════════════════════════════════════════════════════════════════════════
#  ARXIV
# ══════════════════════════════════════════════════════════════════════════

def fetch_arxiv(config: dict, days_back: int, max_results: int) -> list[dict]:
    cutoff  = datetime.now(timezone.utc) - timedelta(days=days_back)
    seen:   set[str] = set()
    out:    list[dict] = []
    client  = arxiv.Client(page_size=min(max_results, 100), delay_seconds=3)
    kw_dict = config["general_tts"]["keywords"]

    for topic, spec in kw_dict.items():
        for filt in spec["filters"]:
            search = arxiv.Search(
                query=f"all:{filt}",
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )
            try:
                for paper in client.results(search):
                    if paper.published.replace(tzinfo=timezone.utc) < cutoff:
                        continue
                    aid = paper.get_short_id().split("v")[0]
                    if aid in seen:
                        continue
                    seen.add(aid)
                    code = (extract_gh_link(paper.summary or "") or
                            extract_gh_link(paper.comment  or ""))
                    out.append(make_entry(
                        uid       = aid,
                        title     = paper.title.strip(),
                        authors   = fmt_authors(paper.authors),
                        published = paper.published.strftime("%Y-%m-%d"),
                        pdf       = f"https://arxiv.org/abs/{aid}",
                        code      = code,
                        etype     = "Paper",
                        abstract  = (paper.summary or "").replace("\n", " "),
                    ))
            except Exception as e:
                print(f"    [arXiv] '{filt}': {e}")
            time.sleep(2)

    print(f"  [arXiv] {len(out)} papers fetched")
    return out

# ══════════════════════════════════════════════════════════════════════════
#  HUGGINGFACE
# ══════════════════════════════════════════════════════════════════════════

def fetch_hf_models(config: dict) -> list[dict]:
    out: list[dict] = []
    seen: set[str]  = set()
    tts_tags = {"text-to-speech","tts","speech-synthesis","voice-synthesis","voice-cloning"}

    for term in config["huggingface"]["model_terms"]:
        url = ("https://huggingface.co/api/models"
               f"?search={term}&sort=lastModified&direction=-1&limit=20")
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            for m in resp.json():
                mid = m.get("id","")
                if not mid or mid in seen:
                    continue
                tags = [t.lower() for t in m.get("tags",[])]
                # must look like a TTS model
                if not (tts_tags & set(tags)):
                    if not any(w in mid.lower() for w in ["tts","speech","voice","synth"]):
                        continue
                seen.add(mid)

                langs = m.get("cardData",{}).get("language",[]) or []
                if isinstance(langs, str): langs = [langs]
                desc  = str(m.get("cardData",{}) or {}).lower()

                combined = (mid + " " + " ".join(langs) + " " + desc).lower()
                is_tamil   = any(w in combined for w in
                                 ["tamil","\"ta\"","'ta'",",ta,","[ta]","dravidian",
                                  "indictts","indic tts","singapore"])
                is_low_res = any(w in combined for w in
                                 ["low-resource","low resource","zero-shot","few-shot"])

                entry = make_entry(
                    uid       = f"hf_model_{mid}",
                    title     = mid,
                    authors   = m.get("author",""),
                    published = m.get("lastModified","")[:10],
                    pdf       = f"https://huggingface.co/{mid}",
                    code      = "null",
                    etype     = "Model",
                )
                entry["is_tamil"]   = is_tamil
                entry["is_low_res"] = is_low_res
                entry["_tags"]      = tags[:8]
                out.append(entry)
        except Exception as e:
            print(f"    [HF-models] '{term}': {e}")
        time.sleep(1)

    print(f"  [HF-models] {len(out)} models fetched")
    return out

def fetch_hf_datasets(config: dict) -> list[dict]:
    out: list[dict] = []
    seen: set[str]  = set()
    speech_tags = {"speech","audio","tts","text-to-speech","asr","voice","spoken"}

    for term in config["huggingface"]["dataset_terms"]:
        url = ("https://huggingface.co/api/datasets"
               f"?search={term}&sort=lastModified&direction=-1&limit=20")
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            for d in resp.json():
                did = d.get("id","")
                if not did or did in seen:
                    continue
                tags = [t.lower() for t in d.get("tags",[])]
                if not (speech_tags & set(tags)):
                    if not any(w in did.lower() for w in
                               ["speech","tts","audio","voice","spoken"]):
                        continue
                seen.add(did)

                langs = d.get("cardData",{}).get("language",[]) or []
                if isinstance(langs, str): langs = [langs]
                desc  = str(d.get("cardData",{}) or {}).lower()

                combined = (did + " " + " ".join(langs) + " " + desc).lower()
                is_tamil   = any(w in combined for w in
                                 ["tamil","\"ta\"","'ta'",",ta,","[ta]","dravidian",
                                  "indictts","indic speech"])
                is_low_res = any(w in combined for w in
                                 ["low-resource","low resource","zero-shot","few-shot"])

                entry = make_entry(
                    uid       = f"hf_dataset_{did}",
                    title     = did,
                    authors   = d.get("author",""),
                    published = d.get("lastModified","")[:10],
                    pdf       = f"https://huggingface.co/datasets/{did}",
                    code      = "null",
                    etype     = "Dataset",
                )
                entry["is_tamil"]   = is_tamil
                entry["is_low_res"] = is_low_res
                entry["_tags"]      = tags[:8]
                out.append(entry)
        except Exception as e:
            print(f"    [HF-datasets] '{term}': {e}")
        time.sleep(1)

    print(f"  [HF-datasets] {len(out)} datasets fetched")
    return out

# ══════════════════════════════════════════════════════════════════════════
#  GITHUB REPOS
# ══════════════════════════════════════════════════════════════════════════

def fetch_github_repos(config: dict) -> list[dict]:
    if not GITHUB_TOKEN:
        print("  [GitHub] No token → skipping")
        return []

    out: list[dict] = []
    seen: set[str]  = set()

    for q in config["github_queries"]:
        url = (f"https://api.github.com/search/repositories"
               f"?q={requests.utils.quote(q)}&sort=stars&order=desc&per_page=10")
        try:
            resp = requests.get(url, headers=HEADERS_GH, timeout=30)
            resp.raise_for_status()
            for repo in resp.json().get("items", []):
                full = repo.get("full_name","")
                if not full or full in seen:
                    continue
                name  = full.lower()
                desc  = (repo.get("description") or "").lower()
                combined = name + " " + desc
                # must look like TTS / speech
                if not any(w in combined for w in
                           ["tts","speech","voice","synthesis","cloning","tamil","indic"]):
                    continue
                seen.add(full)

                is_tamil   = any(w in combined for w in
                                 ["tamil","indic","dravidian","singapore"])
                is_low_res = any(w in combined for w in
                                 ["low-resource","low resource","zero-shot","few-shot"])

                entry = make_entry(
                    uid       = f"github_{full}",
                    title     = full,
                    authors   = repo.get("owner",{}).get("login",""),
                    published = (repo.get("pushed_at") or "")[:10],
                    pdf       = "null",
                    code      = repo.get("html_url",""),
                    etype     = "Repo",
                )
                entry["is_tamil"]   = is_tamil
                entry["is_low_res"] = is_low_res
                entry["_stars"]     = repo.get("stargazers_count", 0)
                entry["_desc"]      = desc[:200]
                out.append(entry)
        except Exception as e:
            print(f"    [GitHub] '{q}': {e}")
        time.sleep(7)   # search API: 10 req/min unauthenticated, 30 with token

    print(f"  [GitHub] {len(out)} repos fetched")
    return out

# ══════════════════════════════════════════════════════════════════════════
#  ARTICLES / SURVEYS
# ══════════════════════════════════════════════════════════════════════════

def fetch_articles(days_back: int) -> list[dict]:
    survey_queries = [
        "TTS survey review",
        "speech synthesis survey",
        "text-to-speech overview progress",
        "low-resource speech synthesis survey",
        "voice cloning survey",
        "multilingual TTS survey",
        "neural TTS review",
    ]
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    out:  list[dict] = []
    seen: set[str]   = set()
    client = arxiv.Client(delay_seconds=3)

    for q in survey_queries:
        search = arxiv.Search(
            query=f"all:{q}",
            max_results=30,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        try:
            for paper in client.results(search):
                if paper.published.replace(tzinfo=timezone.utc) < cutoff:
                    continue
                aid = paper.get_short_id().split("v")[0]
                if aid in seen:
                    continue
                # Only keep things that look like surveys/overviews
                tl = paper.title.lower()
                if not any(w in tl for w in
                           ["survey","review","overview","progress","advances","tutorial"]):
                    continue
                seen.add(aid)
                out.append(make_entry(
                    uid       = f"article_{aid}",
                    title     = paper.title.strip(),
                    authors   = fmt_authors(paper.authors),
                    published = paper.published.strftime("%Y-%m-%d"),
                    pdf       = f"https://arxiv.org/abs/{aid}",
                    code      = extract_gh_link(paper.summary or ""),
                    etype     = "Article",
                    abstract  = (paper.summary or "").replace("\n"," "),
                ))
        except Exception as e:
            print(f"    [Articles] '{q}': {e}")
        time.sleep(2)

    print(f"  [Articles] {len(out)} articles fetched")
    return out

# ══════════════════════════════════════════════════════════════════════════
#  MERGE
# ══════════════════════════════════════════════════════════════════════════

def merge(existing: list, new_list: list) -> tuple[list, int]:
    added = 0
    for e in new_list:
        if not is_dup(existing, e["id"]):
            existing.append(e)
            added += 1
    return existing, added

# ══════════════════════════════════════════════════════════════════════════
#  MARKDOWN HELPERS
# ══════════════════════════════════════════════════════════════════════════

def sort_newest(entries: list[dict]) -> list[dict]:
    return sorted(entries, key=lambda x: x.get("published","0000-00-00"), reverse=True)

def write_table(f, entries: list[dict]):
    if not entries:
        f.write("*No entries yet – will populate on next run.*\n\n")
        return
    f.write("| Publish Date | Title | Authors | PDF | Code |\n")
    f.write("|---|---|---|---|---|\n")
    for e in sort_newest(entries):
        date    = (e.get("published") or "")[:10]
        title   = (e.get("title") or "").replace("|","\\|")
        authors = (e.get("authors") or "").replace("|","\\|")
        pdf_raw = e.get("pdf","null") or "null"
        code_raw= e.get("code","null") or "null"
        etype   = e.get("type","")
        uid     = e.get("id","")

        # ── PDF cell ──
        if etype == "Repo":
            pdf_cell = "null"
        elif etype in ("Model","Dataset"):
            pdf_cell = f"[🤗]({pdf_raw})" if pdf_raw != "null" else "null"
        else:
            # arXiv paper — show short ID as link text
            aid = uid.replace("article_","")
            pdf_cell = (f"[{aid}]({pdf_raw})"
                        if pdf_raw != "null"
                        else "null")

        # ── Code cell ──
        if etype == "Repo":
            stars = e.get("_stars","")
            star_str = f" ⭐{stars}" if stars else ""
            code_cell = (f"[repo]({code_raw}){star_str}"
                         if code_raw != "null" else "null")
        else:
            code_cell = (f"[code]({code_raw})"
                         if code_raw != "null" else "null")

        f.write(f"| {date} | {title} | {authors} | {pdf_cell} | {code_cell} |\n")
    f.write("\n")

# ══════════════════════════════════════════════════════════════════════════
#  PAGE WRITERS
# ══════════════════════════════════════════════════════════════════════════

def write_readme(papers, repos, datasets, articles):
    with open(README_FILE, "w", encoding="utf-8") as f:
        f.write("# 🗣️ TTS Resources Daily\n\n")
        f.write(
            "A curated, automatically‑updated list of papers, repositories, "
            "datasets, and articles for **Text‑to‑Speech** research — with a "
            "focus on Singapore Tamil and low-resource TTS.\n\n"
        )
        f.write(
            f"**Updated:** {today_str} &nbsp;|&nbsp; "
            f"**Papers:** {len(papers)} &nbsp;|&nbsp; "
            f"**Repos:** {len(repos)} &nbsp;|&nbsp; "
            f"**Datasets:** {len(datasets)} &nbsp;|&nbsp; "
            f"**Articles:** {len(articles)}\n\n"
        )
        f.write("> 📌 Tamil-specific resources → [tamil-tts.md](tamil-tts.md)  \n")
        f.write("> 📌 Repos by focus → [repos.md](repos.md)\n\n")
        f.write("---\n\n")

        f.write("## 📄 Papers\n\n")
        write_table(f, papers)

        f.write("## 📦 Repositories\n\n")
        write_table(f, repos)

        f.write("## 📊 Datasets\n\n")
        write_table(f, datasets)

        f.write("## 📰 Articles & Surveys\n\n")
        write_table(f, articles)


def write_tamil_page(papers, models, datasets):
    # distil from the main lists using LLM flags
    tamil_papers = [p for p in papers   if p.get("is_tamil")]
    low_papers   = [p for p in papers   if p.get("is_low_res") and not p.get("is_tamil")]
    tamil_models = [m for m in models   if m.get("is_tamil")]
    tamil_ds     = [d for d in datasets if d.get("is_tamil")]

    with open(TAMIL_FILE, "w", encoding="utf-8") as f:
        f.write("# 🇸🇬 Singapore Tamil TTS & Low‑Resource Speech\n\n")
        f.write(
            "All entries are **distilled from the main list** using Gemini LLM "
            "(title + abstract) — not just keyword matching. "
            "Papers can appear in both [README.md](README.md) and here.\n\n"
        )
        f.write(
            f"**Updated:** {today_str} &nbsp;|&nbsp; "
            f"**Tamil papers:** {len(tamil_papers)} &nbsp;|&nbsp; "
            f"**Low-resource papers:** {len(low_papers)} &nbsp;|&nbsp; "
            f"**Tamil models:** {len(tamil_models)} &nbsp;|&nbsp; "
            f"**Tamil datasets:** {len(tamil_ds)}\n\n"
        )
        f.write("---\n\n")

        f.write("## 🎯 Tamil TTS Papers\n\n")
        f.write(
            "*Papers verified by LLM as being about Tamil TTS, Tamil speech synthesis, "
            "Tamil speech corpora, or multilingual TTS explicitly including Tamil (ta).*\n\n"
        )
        write_table(f, tamil_papers)

        f.write("## 🌍 Low-Resource Language TTS Papers\n\n")
        f.write(
            "*Papers on low-resource, zero-shot, or data-efficient speech synthesis — "
            "relevant methodology for building Singapore Tamil TTS. "
            "Excludes papers already in the Tamil section above.*\n\n"
        )
        write_table(f, low_papers)

        f.write("## 🤗 Tamil TTS Models\n\n")
        f.write("*HuggingFace models supporting Tamil TTS.*\n\n")
        write_table(f, tamil_models)

        f.write("## 📊 Tamil Speech Datasets\n\n")
        f.write("*HuggingFace datasets for Tamil speech / TTS.*\n\n")
        write_table(f, tamil_ds)


def write_repos_page(repos):
    # split by flag — a repo can match multiple sections
    tamil_r  = [r for r in repos if r.get("is_tamil")]
    lowres_r = [r for r in repos if r.get("is_low_res")]
    tts_r    = [r for r in repos
                if not r.get("is_tamil") and not r.get("is_low_res")]

    with open(REPOS_FILE, "w", encoding="utf-8") as f:
        f.write("# 📦 TTS Repositories\n\n")
        f.write(
            "Open-source repositories grouped by focus. "
            "Tamil and low-resource repos may also appear in their respective sections.\n\n"
        )
        f.write(
            f"**Updated:** {today_str} &nbsp;|&nbsp; "
            f"**TTS:** {len(tts_r)} &nbsp;|&nbsp; "
            f"**Low-resource:** {len(lowres_r)} &nbsp;|&nbsp; "
            f"**Tamil:** {len(tamil_r)}\n\n"
        )
        f.write("---\n\n")

        f.write("## 🎙️ TTS\n\n")
        f.write("*General TTS repos (not exclusively Tamil or low-resource).*\n\n")
        write_table(f, tts_r)

        f.write("## 🌍 Low Resource Language\n\n")
        f.write("*Repos focused on low-resource or under-resourced speech synthesis.*\n\n")
        write_table(f, lowres_r)

        f.write("## 🇸🇬 Tamil\n\n")
        f.write("*Repos related to Tamil TTS, Tamil speech, or Singapore Tamil.*\n\n")
        write_table(f, tamil_r)

# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    config = load_config()

    # Detect first run
    existing_papers = load_json(DATA_PAPERS)
    is_first_run    = len(existing_papers) == 0

    tts_cfg     = config["general_tts"]
    days_back   = tts_cfg["days_back_initial"]   if is_first_run else tts_cfg["days_back_daily"]
    max_results = tts_cfg["max_results_initial"]  if is_first_run else tts_cfg["max_results_daily"]

    print(f"\n{'='*62}")
    print(f"  Singapore Tamil TTS — Update ({today_str})")
    print(f"  First run: {is_first_run}  |  Look-back: {days_back}d  |  Max/query: {max_results}")
    print(f"  Gemini key present: {'YES' if GEMINI_API_KEY else 'NO (keyword fallback)'}")
    print(f"{'='*62}\n")

    # ── Fetch all sources ─────────────────────────────────────────────────
    print("📄 Fetching arXiv papers…")
    new_papers = fetch_arxiv(config, days_back, max_results)

    print("\n🤗 Fetching HuggingFace models…")
    new_models = fetch_hf_models(config)

    print("\n📊 Fetching HuggingFace datasets…")
    new_datasets = fetch_hf_datasets(config)

    print("\n🐙 Fetching GitHub repos…")
    new_repos = fetch_github_repos(config)

    print("\n📰 Fetching survey articles…")
    new_articles = fetch_articles(days_back)

    # ── Load existing DBs ─────────────────────────────────────────────────
    existing_repos     = load_json(DATA_REPOS)
    existing_datasets  = load_json(DATA_DATASETS)
    existing_models    = load_json(DATA_MODELS)
    existing_articles  = load_json(DATA_ARTICLES)

    # ── Merge new into existing ───────────────────────────────────────────
    existing_papers,  pa = merge(existing_papers,  new_papers)
    existing_repos,   ra = merge(existing_repos,   new_repos)
    existing_datasets,da = merge(existing_datasets,new_datasets)
    existing_models,  ma = merge(existing_models,  new_models)
    existing_articles,aa = merge(existing_articles,new_articles)

    print(f"\n  Added → papers:{pa}  repos:{ra}  datasets:{da}  models:{ma}  articles:{aa}")

    # ── LLM-tag ALL untagged papers (new + any previously missed) ─────────
    print("\n🔍 LLM tagging papers…")
    existing_papers = tag_all_untagged(existing_papers)

    # ── Save JSONs ────────────────────────────────────────────────────────
    save_json(DATA_PAPERS,   existing_papers)
    save_json(DATA_REPOS,    existing_repos)
    save_json(DATA_DATASETS, existing_datasets)
    save_json(DATA_MODELS,   existing_models)
    save_json(DATA_ARTICLES, existing_articles)
    print("  JSONs saved.")

    # ── Write markdown pages ──────────────────────────────────────────────
    print("\n📝 Writing pages…")
    write_readme(existing_papers, existing_repos, existing_datasets, existing_articles)
    print("  ✅ README.md")

    write_tamil_page(existing_papers, existing_models, existing_datasets)
    print("  ✅ tamil-tts.md")

    write_repos_page(existing_repos)
    print("  ✅ repos.md")

    print(f"\n{'='*62}")
    print(f"  Done!  papers:{len(existing_papers)}  repos:{len(existing_repos)}  "
          f"ds:{len(existing_datasets)}  models:{len(existing_models)}  articles:{len(existing_articles)}")
    print(f"  Gemini API calls this run: {_gemini_calls}")
    print(f"{'='*62}\n")
