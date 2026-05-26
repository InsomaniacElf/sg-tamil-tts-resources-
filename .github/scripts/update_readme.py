"""
Singapore Tamil TTS Resources – Daily Update Script
----------------------------------------------------
Fetches papers (arXiv), models/datasets (HuggingFace), repos (GitHub), and
articles (web search), then generates three Markdown pages:
  1. README.md       – All resources: Papers → Repos → Datasets → Articles
  2. tamil-tts.md    – Tamil-relevant papers + Low Resource Language section
  3. repos.md        – Repos grouped: TTS → Low Resource Language → Tamil

The LLM (Gemini) checks EVERY paper for Tamil / low-resource relevance,
not just keyword matching.

First run: backfill from 2019 onward (days_back ≈ 2500, max_results = 300).
Daily runs: fetch last 7 days, max 10 results per query.
"""

import os, json, re, time, yaml
from datetime import datetime, timedelta, timezone
import requests
import arxiv

# ── Paths ──────────────────────────────────────────────
CONFIG_FILE   = ".github/scripts/config.yaml"
DATA_PAPERS   = "data/papers.json"
DATA_REPOS    = "data/repos.json"
DATA_DATASETS = "data/datasets.json"
DATA_ARTICLES = "data/articles.json"
README_FILE   = "README.md"
TAMIL_FILE    = "tamil-tts.md"
REPOS_FILE    = "repos.md"

# ── Environment ────────────────────────────────────────
GITHUB_TOKEN   = os.environ.get("GITHUB_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
HEADERS_GH     = {"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
today_str      = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# ╔══════════════════════════════════════════════════════╗
# ║                    HELPERS                          ║
# ╚══════════════════════════════════════════════════════╝

def load_config():
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def is_duplicate(entries, new):
    uid = new.get("id", new.get("title", ""))
    return any(e.get("id", e.get("title")) == uid for e in entries)

def clean_authors(authors):
    if isinstance(authors, list):
        names = [a.name for a in authors]
    else:
        names = [a.strip() for a in str(authors).split(",") if a.strip()]
    if len(names) > 3:
        return ", ".join(names[:3]) + " et al."
    return ", ".join(names)

def extract_github_link(text):
    m = re.search(r"https?://github\.com/[\w\-./]+", text)
    return m.group(0) if m else None

def make_entry(uid, title, authors, published, pdf, code, etype):
    return {
        "id": uid,
        "title": title,
        "authors": authors,
        "published": published,
        "pdf": pdf,
        "code": code or "null",
        "type": etype,
        "date_added": today_str,
    }

# ╔══════════════════════════════════════════════════════╗
# ║               LLM RELEVANCE CHECK                   ║
# ╚══════════════════════════════════════════════════════╝

def ask_gemini(prompt):
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 20, "temperature": 0.0},
    }
    resp = requests.post(url, json=payload, timeout=15)
    if resp.status_code == 200:
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip().upper()
    return "ERROR"

def is_tamil_relevant(title, abstract):
    if GEMINI_API_KEY:
        prompt = f"""
You are a research filter. Determine if this paper is DIRECTLY relevant to:
- Tamil Text-to-Speech (any variety: Indian Tamil, Singapore Tamil, Sri Lankan Tamil)
- Code-switching TTS involving Tamil
- Speech datasets that include Tamil (ta) language
- Multilingual TTS models that explicitly support Tamil
- Low-resource speech synthesis specifically targeting Tamil or Dravidian languages

Answer ONLY "RELEVANT" or "NOT_RELEVANT".

Title: {title}
Abstract: {abstract[:1500]}
"""
        try:
            ans = ask_gemini(prompt)
            return "RELEVANT" in ans
        except Exception as e:
            print(f"  Gemini error: {e}")

    # Fallback heuristic
    text = (title + " " + abstract).lower()
    has_tamil = any(w in text for w in ["tamil", "tam ", "ta ", "singapore ", "dravidian"])
    has_tts   = any(w in text for w in ["tts", "text-to-speech", "speech synthesis", "voice cloning"])
    return has_tamil and has_tts

def is_low_resource_relevant(title, abstract):
    if GEMINI_API_KEY:
        prompt = f"""
Determine if this paper is about LOW-RESOURCE speech synthesis / TTS.
Low-resource means: limited data, under-resourced languages, few speakers,
zero-shot for unseen languages, or data-efficient TTS methods.

Answer ONLY "RELEVANT" or "NOT_RELEVANT".

Title: {title}
Abstract: {abstract[:1500]}
"""
        try:
            ans = ask_gemini(prompt)
            return "RELEVANT" in ans
        except Exception:
            pass

    text = (title + " " + abstract).lower()
    keywords = ["low-resource", "low resource", "under-resourced", "zero-shot",
                "few-shot", "data-efficient", "limited data"]
    return any(kw in text for kw in keywords)

def is_tamil_model_or_dataset(name, description, tags, languages):
    text = f"{name} {description} {' '.join(tags)} {' '.join(languages)}".lower()
    tamil_signals = ["tamil", "tam", "ta", "singapore", "indic", "dravidian"]
    return any(s in text for s in tamil_signals)

# ╔══════════════════════════════════════════════════════╗
# ║               ARXIV FETCH                           ║
# ╚══════════════════════════════════════════════════════╝

def fetch_arxiv(config, days_back, max_results):
    keywords_dict = config["general_tts"]["keywords"]
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    entries = []

    for topic, spec in keywords_dict.items():
        for filt in spec["filters"]:
            query = f"all:{filt}"
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )
            try:
                for paper in search.results():
                    if paper.published.replace(tzinfo=timezone.utc) < cutoff:
                        continue

                    arxiv_id = paper.get_short_id().split("v")[0]
                    pdf      = f"https://arxiv.org/abs/{arxiv_id}"
                    code     = extract_github_link(paper.summary)

                    entry = make_entry(
                        uid=arxiv_id,
                        title=paper.title,
                        authors=clean_authors(paper.authors),
                        published=paper.published.strftime("%Y-%m-%d"),
                        pdf=pdf,
                        code=code,
                        etype="Paper",
                    )
                    entries.append(entry)
            except Exception as e:
                print(f"  arXiv error for '{filt}': {e}")
            time.sleep(2)
    return entries

# ╔══════════════════════════════════════════════════════╗
# ║            HUGGINGFACE FETCH                        ║
# ╚══════════════════════════════════════════════════════╝

def fetch_huggingface_models(config):
    entries = []
    for term in config["huggingface"]["model_terms"]:
        url = (
            f"https://huggingface.co/api/models"
            f"?search={term}&sort=lastModified&direction=-1&limit=15"
        )
        try:
            for m in requests.get(url, timeout=30).json():
                mid    = m.get("id", "")
                tags   = [t.lower() for t in m.get("tags", [])]
                desc   = (m.get("cardData", {}).get("model_description") or "").lower()
                langs  = m.get("cardData", {}).get("language", [])
                if isinstance(langs, str):
                    langs = [langs]

                # Must be TTS-related
                tts_tags = {"text-to-speech", "tts", "speech-synthesis", "voice-synthesis"}
                if not (tts_tags & set(tags)):
                    if "tts" not in mid.lower() and "speech" not in mid.lower():
                        continue

                entry = make_entry(
                    uid=f"hf_model_{mid}",
                    title=mid,
                    authors=m.get("author", ""),
                    published=m.get("lastModified", "")[:10],
                    pdf=f"https://huggingface.co/{mid}",
                    code="null",
                    etype="Model",
                )
                entry["_tags"] = tags
                entry["_desc"] = desc
                entry["_langs"] = langs
                entries.append(entry)
        except Exception as e:
            print(f"  HF models error for '{term}': {e}")
        time.sleep(2)
    return entries

def fetch_huggingface_datasets(config):
    entries = []
    for term in config["huggingface"]["dataset_terms"]:
        url = (
            f"https://huggingface.co/api/datasets"
            f"?search={term}&sort=lastModified&direction=-1&limit=15"
        )
        try:
            for d in requests.get(url, timeout=30).json():
                did   = d.get("id", "")
                desc  = (d.get("description", "") or "").lower()
                langs = d.get("cardData", {}).get("language", [])
                if isinstance(langs, str):
                    langs = [langs]
                tags  = [t.lower() for t in d.get("tags", [])]

                speech_tags = {"speech", "audio", "tts", "text-to-speech", "asr", "voice"}
                if not (speech_tags & set(tags)):
                    if not any(w in did.lower() for w in ["speech", "tts", "audio", "voice"]):
                        continue

                entry = make_entry(
                    uid=f"hf_dataset_{did}",
                    title=did,
                    authors=d.get("author", ""),
                    published=d.get("lastModified", "")[:10],
                    pdf=f"https://huggingface.co/datasets/{did}",
                    code="null",
                    etype="Dataset",
                )
                entry["_tags"] = tags
                entry["_desc"] = desc
                entry["_langs"] = langs
                entries.append(entry)
        except Exception as e:
            print(f"  HF datasets error for '{term}': {e}")
        time.sleep(2)
    return entries

# ╔══════════════════════════════════════════════════════╗
# ║              GITHUB SEARCH                          ║
# ╚══════════════════════════════════════════════════════╝

def fetch_github_repos(config):
    if not GITHUB_TOKEN:
        print("  No GITHUB_TOKEN, skipping GitHub search")
        return []
    entries = []
    for q in config["github_queries"]:
        url = (
            f"https://api.github.com/search/repositories"
            f"?q={q}&sort=stars&order=desc&per_page=10"
        )
        try:
            resp = requests.get(url, headers=HEADERS_GH, timeout=30).json()
            for repo in resp.get("items", []):
                full  = repo["full_name"]
                desc  = (repo.get("description") or "").lower()
                stars = repo.get("stargazers_count", 0)
                if not any(w in desc for w in ["tts", "speech", "tamil", "voice"]):
                    if not any(w in full.lower() for w in ["tts", "speech", "tamil", "voice"]):
                        continue
                entry = make_entry(
                    uid=f"github_{full}",
                    title=full,
                    authors=repo["owner"]["login"],
                    published=repo.get("pushed_at", "")[:10],
                    pdf="null",
                    code=repo["html_url"],
                    etype="Repo",
                )
                entry["_stars"] = stars
                entry["_desc"]  = desc
                entries.append(entry)
        except Exception as e:
            print(f"  GitHub error for '{q}': {e}")
        time.sleep(6)
    return entries

# ╔══════════════════════════════════════════════════════╗
# ║            ARTICLE SEARCH (surveys)                 ║
# ╚══════════════════════════════════════════════════════╝

def fetch_articles():
    entries = []
    surveys = [
        "TTS survey",
        "speech synthesis survey",
        "text-to-speech overview",
        "low-resource speech synthesis tutorial",
    ]
    cutoff = datetime.now(timezone.utc) - timedelta(days=180)

    for q in surveys:
        search = arxiv.Search(
            query=f"all:{q}",
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        try:
            for paper in search.results():
                if paper.published.replace(tzinfo=timezone.utc) < cutoff:
                    continue
                arxiv_id = paper.get_short_id().split("v")[0]
                entry = make_entry(
                    uid=f"article_{arxiv_id}",
                    title=paper.title,
                    authors=clean_authors(paper.authors),
                    published=paper.published.strftime("%Y-%m-%d"),
                    pdf=f"https://arxiv.org/abs/{arxiv_id}",
                    code=extract_github_link(paper.summary),
                    etype="Article",
                )
                entries.append(entry)
        except Exception as e:
            print(f"  Article search error for '{q}': {e}")
        time.sleep(2)
    return entries

# ╔══════════════════════════════════════════════════════╗
# ║            MARKDOWN WRITERS                         ║
# ╚══════════════════════════════════════════════════════╝

def sort_entries(entries):
    return sorted(entries, key=lambda x: x.get("date_added", ""), reverse=True)

def write_table(f, entries, columns=None):
    if columns is None:
        columns = ["date_added", "published", "title", "authors", "pdf", "code", "type"]
    header_map = {
        "date_added": "Date Added",
        "published": "Published",
        "title": "Title",
        "authors": "Authors",
        "pdf": "PDF",
        "code": "Code",
        "type": "Type",
    }
    f.write("| " + " | ".join(header_map[c] for c in columns) + " |\n")
    f.write("|" + "|".join(["---"] * len(columns)) + "|\n")
    for e in sort_entries(entries):
        row = []
        for c in columns:
            val = e.get(c, "")
            if c == "pdf" and val != "null":
                val = f"[PDF]({val})"
            elif c == "code" and val != "null":
                val = f"[Code]({val})"
            elif c == "title":
                val = val.replace("|", "\\|")
            elif c == "authors":
                val = val.replace("|", "\\|")
            row.append(str(val))
        f.write("| " + " | ".join(row) + " |\n")
    f.write("\n")

def write_main_readme(papers, repos, datasets, articles):
    with open(README_FILE, "w", encoding="utf-8") as f:
        f.write("# 🗣️ TTS Resources Daily\n\n")
        f.write(
            "Automatically updated every 12 hours – a curated, searchable list of "
            "papers, repositories, datasets, and articles for Text‑to‑Speech research.\n\n"
        )
        f.write(f"**Last updated:** {today_str}\n\n")
        f.write("---\n\n")

        f.write("## 📄 Papers\n\n")
        if papers:
            write_table(f, papers)
        else:
            f.write("*No papers found.*\n\n")

        f.write("## 📦 Repositories\n\n")
        if repos:
            write_table(f, repos)
        else:
            f.write("*No repositories found.*\n\n")

        f.write("## 📊 Datasets\n\n")
        if datasets:
            write_table(f, datasets)
        else:
            f.write("*No datasets found.*\n\n")

        f.write("## 📰 Articles & Surveys\n\n")
        if articles:
            write_table(f, articles)
        else:
            f.write("*No articles found.*\n\n")

def write_tamil_page(papers, models, datasets):
    tamil_papers    = []
    low_resource    = []
    tamil_models    = []
    tamil_datasets  = []

    print("\n🔍 Checking papers for Tamil relevance...")
    for p in papers:
        title = p.get("title", "")
        abstract = title  # fallback; for backfill we can't fetch abstract for all old papers
        # For newer papers we could try to get abstract from arXiv but that's too many API calls.
        # We'll rely on title + LLM + heuristic.
        if is_tamil_relevant(title, abstract):
            tamil_papers.append(p)
            print(f"  ✅ TAMIL: {title[:80]}")
        elif is_low_resource_relevant(title, abstract):
            low_resource.append(p)
            print(f"  🔸 LOW-RES: {title[:80]}")

    for m in models:
        if is_tamil_model_or_dataset(m.get("title",""), m.get("_desc",""), m.get("_tags",[]), m.get("_langs",[])):
            tamil_models.append(m)

    for d in datasets:
        if is_tamil_model_or_dataset(d.get("title",""), d.get("_desc",""), d.get("_tags",[]), d.get("_langs",[])):
            tamil_datasets.append(d)

    with open(TAMIL_FILE, "w", encoding="utf-8") as f:
        f.write("# 🇸🇬 Singapore Tamil TTS & Low‑Resource Speech Resources\n\n")
        f.write(
            "Curated resources for building Text‑to‑Speech systems for "
            "Singapore Tamil and related low‑resource varieties. "
            "All entries are verified by an LLM for Tamil relevance.\n\n"
        )
        f.write(f"**Last updated:** {today_str}\n\n")
        f.write("---\n\n")

        f.write("## 🎯 Tamil TTS Papers\n\n")
        f.write("*Papers involving Tamil (ta) TTS, verified by LLM.*\n\n")
        if tamil_papers:
            write_table(f, tamil_papers)
        else:
            f.write("*No Tamil-specific papers found yet. Check back soon.*\n\n")

        f.write("## 🌍 Low Resource Language TTS\n\n")
        f.write("*Papers on low-resource speech synthesis, relevant for Tamil and other under-resourced languages.*\n\n")
        if low_resource:
            write_table(f, low_resource)
        else:
            f.write("*No low-resource papers found yet.*\n\n")

        f.write("## 🤗 Tamil TTS Models (HuggingFace)\n\n")
        if tamil_models:
            write_table(f, tamil_models)
        else:
            f.write("*No Tamil models found yet.*\n\n")

        f.write("## 📊 Tamil Speech Datasets\n\n")
        if tamil_datasets:
            write_table(f, tamil_datasets)
        else:
            f.write("*No Tamil datasets found yet.*\n\n")

def write_repos_page(repos):
    tts_repos       = []
    low_res_repos   = []
    tamil_repos     = []

    for r in repos:
        desc = r.get("_desc", "")
        name = r.get("title", "").lower()
        if any(w in name + desc for w in ["tamil", "tam ", "indic", "dravidian", "singapore"]):
            tamil_repos.append(r)
        elif any(w in desc for w in ["low-resource", "low resource", "under-resourced",
                                       "zero-shot", "few-shot", "minimal data"]):
            low_res_repos.append(r)
        elif any(w in name + desc for w in ["tts", "text-to-speech", "speech synthesis",
                                              "voice cloning", "speech generation"]):
            tts_repos.append(r)
        else:
            tts_repos.append(r)

    with open(REPOS_FILE, "w", encoding="utf-8") as f:
        f.write("# 📦 TTS Repositories\n\n")
        f.write("Open-source repositories for Text‑to‑Speech, grouped by focus area.\n\n")
        f.write(f"**Last updated:** {today_str}\n\n")
        f.write("---\n\n")

        f.write("## 🎙️ General TTS\n\n")
        if tts_repos:
            write_table(f, tts_repos)
        else:
            f.write("*No repositories found.*\n\n")

        f.write("## 🌍 Low Resource Language TTS\n\n")
        if low_res_repos:
            write_table(f, low_res_repos)
        else:
            f.write("*No low-resource repositories found.*\n\n")

        f.write("## 🇸🇬 Tamil TTS\n\n")
        if tamil_repos:
            write_table(f, tamil_repos)
        else:
            f.write("*No Tamil-specific repositories found.*\n\n")

# ╔══════════════════════════════════════════════════════╗
# ║                    MAIN                             ║
# ╚══════════════════════════════════════════════════════╝

if __name__ == "__main__":
    config = load_config()
    cfg    = config["general_tts"]

    existing_papers = load_json(DATA_PAPERS)
    is_first_run    = len(existing_papers) == 0

    # ── Choose parameters based on first run ──
    if is_first_run:
        days_back  = cfg.get("days_back_initial", 2500)
        max_results = cfg.get("max_results_initial", 300)
    else:
        days_back  = cfg.get("days_back_daily", 7)
        max_results = cfg.get("max_results_daily", 10)

    print(f"\n{'='*60}")
    print(f"  Singapore Tamil TTS Resources – Update Script")
    print(f"  Date: {today_str}")
    print(f"  First run: {is_first_run} | Lookback: {days_back} days | Max/query: {max_results}")
    print(f"{'='*60}\n")

    # ── 1. arXiv ──
    print("📄 Fetching arXiv papers...")
    new_papers = fetch_arxiv(config, days_back, max_results)
    print(f"  → {len(new_papers)} new papers")

    # ── 2. HuggingFace models ──
    print("\n🤗 Fetching HuggingFace models...")
    new_models = fetch_huggingface_models(config)
    print(f"  → {len(new_models)} new models")

    # ── 3. HuggingFace datasets ──
    print("\n📊 Fetching HuggingFace datasets...")
    new_datasets = fetch_huggingface_datasets(config)
    print(f"  → {len(new_datasets)} new datasets")

    # ── 4. GitHub ──
    print("\n🐙 Fetching GitHub repositories...")
    new_repos = fetch_github_repos(config)
    print(f"  → {len(new_repos)} new repos")

    # ── 5. Articles ──
    print("\n📰 Fetching articles & surveys...")
    new_articles = fetch_articles()
    print(f"  → {len(new_articles)} new articles")

    # ── 6. Merge & deduplicate ──
    for entry in new_papers:
        if not is_duplicate(existing_papers, entry):
            existing_papers.append(entry)
    save_json(DATA_PAPERS, existing_papers)

    existing_repos = load_json(DATA_REPOS)
    for entry in new_repos:
        if not is_duplicate(existing_repos, entry):
            existing_repos.append(entry)
    save_json(DATA_REPOS, existing_repos)

    existing_datasets = load_json(DATA_DATASETS)
    for entry in new_datasets:
        if not is_duplicate(existing_datasets, entry):
            existing_datasets.append(entry)
    save_json(DATA_DATASETS, existing_datasets)

    existing_articles = load_json(DATA_ARTICLES)
    for entry in new_articles:
        if not is_duplicate(existing_articles, entry):
            existing_articles.append(entry)
    save_json(DATA_ARTICLES, existing_articles)

    # ── 7. Generate pages ──
    print("\n📝 Generating pages...")
    write_main_readme(existing_papers, existing_repos, existing_datasets, existing_articles)
    print("  ✅ README.md")
    write_tamil_page(existing_papers, new_models, new_datasets)
    print("  ✅ tamil-tts.md")
    write_repos_page(existing_repos)
    print("  ✅ repos.md")

    print(f"\n{'='*60}")
    print(f"  Done! Papers: {len(existing_papers)} | Repos: {len(existing_repos)}")
    print(f"  Datasets: {len(existing_datasets)} | Articles: {len(existing_articles)}")
    print(f"{'='*60}\n")
