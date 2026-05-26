import os, json, re, time, yaml
from datetime import datetime, timedelta, timezone
import requests
import arxiv

CONFIG_FILE = ".github/scripts/config.yaml"
DATA_GENERAL = "data/entries_tts.json"
DATA_TAMIL  = "data/entries.json"
README_FILE = "README.md"
TAMIL_FILE  = "tamil-tts.md"

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
HEADERS_GH = {"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

def load_config():
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)

def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def is_duplicate(entries, new):
    uid = new.get("id", new.get("title",""))
    return any(e.get("id", e.get("title")) == uid for e in entries)

def clean_authors(authors):
    names = [a.name for a in authors] if isinstance(authors, list) else [a.strip() for a in str(authors).split(",")]
    if len(names) > 3:
        return ", ".join(names[:3]) + " et al."
    return ", ".join(names)

def extract_github_link(text):
    m = re.search(r"https?://github\.com/[\w\-./]+", text)
    return m.group(0) if m else None

# ----- LLM filter (used only for Tamil page) -----
def is_relevant_tamil(title, abstract):
    if GEMINI_API_KEY:
        prompt = f"""
You are a filter for a repository about Singapore Tamil Text‑to‑Speech and low‑resource speech synthesis.
Is this paper directly relevant? Focus on:
- Tamil TTS (any variety, especially Singapore Tamil)
- Low‑resource speech synthesis
- Code‑switching TTS
- Multilingual zero‑shot TTS for under‑resourced languages
- Accent/dialect adaptation in TTS
- Speech datasets for Tamil or Singaporean languages
Answer only "RELEVANT" or "NOT_RELEVANT".

Title: {title}
Abstract: {abstract}
"""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents":[{"parts":[{"text":prompt}]}],
                "generationConfig":{"maxOutputTokens":5,"temperature":0.0}
            }
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                ans = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip().upper()
                return "RELEVANT" in ans
        except Exception as e:
            print(f"Gemini error: {e}")
    # fallback heuristic
    text = (title + " " + abstract).lower()
    return ("tamil" in text or "singapore" in text or "low-resource" in text) and \
           any(w in text for w in ["tts", "speech synthesis", "text-to-speech"])

# ----- ArXiv fetch for a given keyword set -----
def fetch_arxiv_for_set(keywords_config, days_back, max_results_per_query, apply_filter=False):
    entries = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    for topic, spec in keywords_config.items():
        for filt in spec["filters"]:
            query = f"all:{filt}"
            search = arxiv.Search(query=query, max_results=max_results_per_query,
                                  sort_by=arxiv.SortCriterion.SubmittedDate)
            try:
                try:
                    results_iter = search.results()
                except AttributeError:
                    results_iter = arxiv.Client().results(search)
                for paper in results_iter:
                    if paper.published.replace(tzinfo=timezone.utc) < cutoff:
                        continue
                    if apply_filter and not is_relevant_tamil(paper.title, paper.summary):
                        continue
                    arxiv_id = paper.get_short_id().split("v")[0]
                    pdf = f"https://arxiv.org/abs/{arxiv_id}"
                    code = extract_github_link(paper.summary)
                    entry = {
                        "id": arxiv_id,
                        "title": paper.title,
                        "authors": clean_authors(paper.authors),
                        "published": paper.published.strftime("%Y-%m-%d"),
                        "pdf": pdf,
                        "code": code or "null",
                        "type": "Paper",
                        "date_added": today_str,
                    }
                    entries.append(entry)
            except Exception as e:
                print(f"arXiv error for '{filt}': {e}")
            time.sleep(2)
    return entries

# ----- HuggingFace (only for Tamil set) -----
def fetch_huggingface(config):
    entries = []
    for term in config["hf_terms"]:
        # models
        url = f"https://huggingface.co/api/models?search={term}&sort=lastModified&direction=-1&limit=5"
        try:
            for m in requests.get(url, timeout=30).json():
                mid = m.get("id")
                tags = [t.lower() for t in m.get("tags", [])]
                if not any(t in tags for t in ["text-to-speech", "tts"]):
                    continue
                if not any(t in tags for t in ["tamil", "ta", "low-resource", "multilingual"]):
                    continue
                entry = {
                    "id": f"hf_model_{mid}",
                    "title": mid,
                    "authors": m.get("author", ""),
                    "published": m.get("lastModified", "")[:10],
                    "pdf": f"https://huggingface.co/{mid}",
                    "code": "null",
                    "type": "Model",
                    "date_added": today_str,
                }
                entries.append(entry)
        except Exception:
            pass
        time.sleep(2)
        # datasets
        url = f"https://huggingface.co/api/datasets?search={term}&sort=lastModified&direction=-1&limit=5"
        try:
            for d in requests.get(url, timeout=30).json():
                did = d.get("id")
                langs = d.get("cardData", {}).get("language", [])
                if isinstance(langs, str): langs = [langs]
                if any("ta" in l.lower() or "tamil" in l.lower() for l in langs):
                    entry = {
                        "id": f"hf_dataset_{did}",
                        "title": did,
                        "authors": d.get("author", ""),
                        "published": d.get("lastModified", "")[:10],
                        "pdf": f"https://huggingface.co/datasets/{did}",
                        "code": "null",
                        "type": "Dataset",
                        "date_added": today_str,
                    }
                    entries.append(entry)
        except Exception:
            pass
        time.sleep(2)
    return entries

# ----- GitHub search (only for Tamil set) -----
def fetch_github(config):
    if not GITHUB_TOKEN:
        return []
    entries = []
    for q in config["github_queries"]:
        url = f"https://api.github.com/search/repositories?q={q}&sort=updated&order=desc&per_page=5"
        try:
            resp = requests.get(url, headers=HEADERS_GH, timeout=30).json()
            for repo in resp.get("items", []):
                full = repo["full_name"]
                desc = (repo.get("description") or "").lower()
                if not any(w in desc for w in ["tamil", "tts", "speech"]):
                    continue
                entry = {
                    "id": f"github_{full}",
                    "title": full,
                    "authors": repo["owner"]["login"],
                    "published": repo.get("pushed_at", "")[:10],
                    "pdf": "null",
                    "code": repo["html_url"],
                    "type": "Tool",
                    "date_added": today_str,
                }
                entries.append(entry)
        except Exception:
            pass
        time.sleep(6)
    return entries

# ----- Markdown table generator -----
def write_markdown(filepath, entries, title):
    entries_sorted = sorted(entries, key=lambda x: x["date_added"], reverse=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Last updated: {today_str}\n\n")
        f.write("| Date Added | Published | Title | Authors | PDF | Code | Type |\n")
        f.write("|------------|-----------|-------|---------|-----|------|------|\n")
        for e in entries_sorted:
            title_esc = e["title"].replace("|", "\\|")
            authors = e["authors"].replace("|", "\\|")
            pdf_link = f"[PDF]({e['pdf']})" if e["pdf"] != "null" else "null"
            code_link = f"[Code]({e['code']})" if e["code"] != "null" else "null"
            f.write(f"| {e['date_added']} | {e.get('published','')} | {title_esc} | {authors} | {pdf_link} | {code_link} | {e['type']} |\n")

# ----- Main -----
if __name__ == "__main__":
    config = load_config()

    # --- General TTS list ---
    general_keywords = config["general_tts"]["keywords"]
    general_days = config["general_tts"].get("days_back", 7)
    general_max = config["general_tts"].get("max_results_per_query", 5)
    gen_existing = load_json(DATA_GENERAL)
    new_gen = fetch_arxiv_for_set(general_keywords, general_days, general_max, apply_filter=False)
    for e in new_gen:
        if not is_duplicate(gen_existing, e):
            gen_existing.append(e)
    save_json(DATA_GENERAL, gen_existing)
    write_markdown(README_FILE, gen_existing, "TTS Paper Daily")

    # --- Tamil low‑resource list ---
    tamil_keywords = config["tamil_tts"]["keywords"]
    tamil_days = config["tamil_tts"].get("days_back", 14)
    tamil_max = config["tamil_tts"].get("max_results_per_query", 10)
    tam_existing = load_json(DATA_TAMIL)
    new_tam = fetch_arxiv_for_set(tamil_keywords, tamil_days, tamil_max, apply_filter=True)
    new_tam += fetch_huggingface(config["tamil_tts"])
    new_tam += fetch_github(config["tamil_tts"])
    for e in new_tam:
        if not is_duplicate(tam_existing, e):
            tam_existing.append(e)
    save_json(DATA_TAMIL, tam_existing)
    write_markdown(TAMIL_FILE, tam_existing,
                   "Singapore Tamil TTS & Low‑Resource Speech Resources")

    print("Both markdown files updated.")
