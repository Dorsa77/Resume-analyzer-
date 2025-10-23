# app.py
import streamlit as st
import re
import json
import pdfplumber
import pandas as pd
import dateparser
import phonenumbers
from email_validator import validate_email, EmailNotValidError
import spacy
from rapidfuzz import fuzz
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import StringIO
from datetime import datetime
import nltk
from nltk.corpus import wordnet
from functools import lru_cache
import tempfile, os
import unicodedata  # ← اضافه کن
import numpy as np
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer, util

def normalize_text(txt: str) -> str:
    # نرمال‌سازی یونیکد تا حروف لهجه‌دار درست باشند (João و …)
    return unicodedata.normalize("NFC", txt)

def _keep_letters_spaces(s: str) -> str:
    # فقط نویسه‌های حرفی همهٔ زبان‌ها + فاصله/آپاستروف/خط‌تیره را نگه می‌دارد
    return "".join(ch for ch in s if unicodedata.category(ch).startswith("L") or ch in " '-")


# ---------- Cache-heavy resources ----------
@st.cache_resource(show_spinner=False)
def get_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except Exception as e:
        st.warning(f"spaCy model load failed: {e}")
        return None

# ---- جایگزینِ تابع get_st_model و نمایش وضعیت ----
@st.cache_resource(show_spinner=True)
def get_st_model():
    """
    تلاش چندمرحله‌ای برای لود SBERT:
    1) مسیر محلی ./models/all-MiniLM-L6-v2 (اگر قبلاً دانلود کرده‌ای)
    2) نام اصلی "all-MiniLM-L6-v2"
    3) مدل جایگزین سبک‌تر "paraphrase-MiniLM-L6-v2"
    اگر همه شکست خورد، None برمی‌گرداند.
    """
    try_paths = [
        "./models/all-MiniLM-L6-v2",   # مسیر محلی (اختیاری)
        "all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
    ]
    last_err = None
    for name in try_paths:
        try:
            m = SentenceTransformer(name)
            # تست خیلی سبک: امبدینگ دو جمله
            _ = m.encode(["test a", "test b"], convert_to_tensor=True, normalize_embeddings=True)
            return m
        except Exception as e:
            last_err = e
    # اگر نرسیدیم
    st.sidebar.error(f"SBERT failed to load. Last error: {last_err}")
    return None

st_model = get_st_model()


@st.cache_resource(show_spinner=False)
def ensure_wordnet():
    try:
        _ = wordnet.synsets("test")
    except LookupError:
        try:
            nltk.download('wordnet')
        except Exception as e:
            st.warning(f"Could not download WordNet: {e}")

nlp = get_spacy()
ensure_wordnet()

# ---------- Helpers ----------
def semantic_similarity(text1, text2, threshold=0.55):
    if st_model is None:
        return False, 0.0
    emb1 = st_model.encode(text1, convert_to_tensor=True, normalize_embeddings=True)
    emb2 = st_model.encode(text2, convert_to_tensor=True, normalize_embeddings=True)
    sim = float(util.pytorch_cos_sim(emb1, emb2).item())
    return sim >= threshold, sim

EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
FALLBACK_PHONE_REGEX = re.compile(
    r"(?:\+?\d{1,3}[\s\-.]?)?"
    r"(?:\(?\d{3}\)?[\s\-.]?)"
    r"\d{3}[\s\-.]?\d{4}"
)

GPA_PATTERN = r"([0-9]+(?:\.[0-9]{1,2})?)\s*/\s*([0-9]+(?:\.[0-9]{1,2})?)"

DEGREES = [
    ("PhD",      ["phd", "ph.d", "doctor", "doctoral"], 5),
    ("Master",   ["master", "m.sc", "m.s", "ms"],       4),
    ("Bachelor", ["bachelor", "b.sc", "b.s", "ba"],     3),
    ("Associate",["associate", "a.sc", "a.s"],          2),
    ("High School", ["high school", "diploma"],         1),
]

FUZZY_THRESHOLD_SKILLS = 80

# ---------- WordNet expansion with caching ----------
@lru_cache(maxsize=2048)
def _expand_one_term(t: str):
    ex = {t.lower()}
    try:
        for syn in wordnet.synsets(t):
            for lemma in syn.lemmas():
                ex.add(lemma.name().replace('_', ' ').lower())
    except Exception:
        pass
    return ex

def expand_keywords(keywords: list) -> list:
    expanded = set()
    for kw in keywords:
        expanded |= _expand_one_term(kw)
    return list(expanded)

def expand_terms(terms):
    expanded = set()
    for t in terms:
        expanded |= _expand_one_term(t)
    return expanded

SECTION_KEYWORDS = {
    "education": [
        "education", "education & training", "academic background",
        "academic qualifications", "academic history", "training",
        "certifications", "courses"
    ],
    "experience": [
        "experience", "work experience", "professional experience",
        "employment", "employment history", "work history",
        "career history", "career summary"
    ],
    "skills": [
        "skills", "technical skills", "tech skills", "skills & technologies",
        "technologies", "core competencies", "competencies",
        "areas of expertise", "toolbox", "tooling"
    ]
}

_BULLET = r"(?:(?:[\-\*\u2022]\s*)?)"      # -, *, •
_ENDPUNCT = r"(?:\s*[:\-–—]?)*"            # :, -, –, —
_MD_PREFIX = r"(?:#{1,6}\s*)?"             # markdown '## '

def _norm_line(s: str) -> str:
    s = normalize_text(s)
    s = s.replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _looks_like_header_text(s: str) -> bool:
    if not s: return False
    if len(s) > 60: return False
    if "@" in s or "http" in s.lower(): return False
    if s.endswith("."): return False
    if len(re.findall(r"\d", s)) > 4: return False
    return True

HDR_PATTERNS = {}
for sec, keys in SECTION_KEYWORDS.items():
    variants = []
    for k in keys:
        k_esc = re.escape(k)
        # دقیقا توالی «\&» و «\-» را جایگزین می‌کنیم (دیگه re.sub لازم نیست)
        k_esc = k_esc.replace(r'\&', r'(?:\s*&\s*|\s+and\s+)')
        k_esc = k_esc.replace(r'\-', r'(?:\-|\s+)')
        variants.append(k_esc)
    pat = rf"^{_MD_PREFIX}{_BULLET}(?:{'|'.join(variants)}){_ENDPUNCT}$"
    HDR_PATTERNS[sec] = re.compile(pat, flags=re.IGNORECASE)



def _is_underline(line: str) -> bool:
    return bool(re.fullmatch(r"\s*[\-–—_=]{3,}\s*", line))

def _detect_header(line: str) -> str | None:
    L = _norm_line(line)
    if not _looks_like_header_text(L):
        return None
    for sec, pat in HDR_PATTERNS.items():
        if pat.search(L):
            return sec
    low = L.lower()
    if re.search(r"\bexperience with\b", low):
        return None
    for sec, keys in SECTION_KEYWORDS.items():
        for k in keys:
            if low.startswith(k.lower()) and len(L) <= 70:
                return sec
    return None

def split_into_sections(raw: str) -> dict:
    sections = {k: [] for k in SECTION_KEYWORDS}
    sections["other"] = []
    lines = raw.split("\n")
    current = "other"
    i = 0
    while i < len(lines):
        line = lines[i]
        normed = _norm_line(line)
        # underline style
        if _looks_like_header_text(normed) and (i + 1) < len(lines) and _is_underline(lines[i+1]):
            sec = _detect_header(normed) or _detect_header(normed.replace(":", ""))
            if sec:
                current = sec
                i += 2
                continue
        # single-line header
        sec = _detect_header(line)
        if sec:
            current = sec
            i += 1
            continue
        if normed:
            sections[current].append(line.rstrip())
        i += 1
    return {k: "\n".join(v) for k, v in sections.items()}


def extract_text_from_pdf(uploaded_file):
    # 1) pdfplumber
    try:
        uploaded_file.seek(0)
        with pdfplumber.open(uploaded_file) as pdf:
            txt = '\n'.join(p.extract_text() or '' for p in pdf.pages)
        if len(txt.strip()) > 100:
            return txt
    except Exception as e:
        st.warning(f"pdfplumber extraction failed: {str(e)}")
    # 2) pdfminer
    try:
        uploaded_file.seek(0)
        mgr = PDFResourceManager()
        ret = StringIO()
        dev = TextConverter(mgr, ret, laparams=LAParams())
        ip = PDFPageInterpreter(mgr, dev)
        for pg in PDFPage.get_pages(uploaded_file):
            ip.process_page(pg)
        dev.close()
        out = ret.getvalue()
        ret.close()
        if len(out.strip()) > 100:
            return out
    except Exception as e:
        st.warning(f"PDFMiner extraction failed: {str(e)}")
    # 3) OCR (needs poppler & tesseract on system)
    try:
        uploaded_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            from pdf2image import convert_from_path
            import pytesseract
            images = convert_from_path(tmp_path, dpi=300)
            text_ocr = '\n'.join(pytesseract.image_to_string(img, lang='eng') for img in images)
            return text_ocr
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        st.error("OCR failed (Poppler & Tesseract required). Details: " + str(e))
        return ''

def extract_email(raw: str) -> str:
    for m in EMAIL_REGEX.finditer(raw):
        cand = m.group(0).strip('.,;:')
        try:
            return validate_email(cand).email
        except EmailNotValidError:
            continue
        except Exception:
            continue
    return 'N/A'

def extract_phone(raw: str, default_region='US') -> str:
    text = raw.replace('\u00A0',' ')
    text = text.translate(str.maketrans({
        '\u2018': "'", '\u2019': "'", '\u201C':'"', '\u201D':'"'
    }))
    text = re.sub(r'[ \t]+',' ', text)
    for m in phonenumbers.PhoneNumberMatcher(text, default_region):
        return phonenumbers.format_number(m.number, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
    mo = FALLBACK_PHONE_REGEX.search(text)
    return mo.group(0).strip() if mo else 'N/A'

# Name affixes cleanup
def strip_name_affixes(name_str: str) -> str:
    name_str = normalize_text(name_str)  # ← این خط را اضافه کن
    PREFIXES = {"dr", "mr", "ms", "mrs", "prof"}
    SUFFIXES = {"phd", "msc", "bsc"}
    tokens = [t for t in re.split(r"\s+", name_str.strip()) if t]
    def norm(t: str) -> str:
        return re.sub(r"[^A-Za-z]", "", t).lower()
    while tokens and norm(tokens[0]) in PREFIXES:
        tokens.pop(0)
    while tokens and norm(tokens[-1].rstrip(",.;")) in SUFFIXES:
        tokens.pop()
    tokens = [t.strip(",.; ") for t in tokens if t.strip(",.; ")]
    return " ".join(tokens)

def extract_name(raw: str) -> tuple:
    text = normalize_text(raw)

    STOPWORDS = {
        "email","e-mail","mail","phone","tel","mobile","cell","contact",
        "location","address","linkedin","github","website","profile"
    }

    def cut_at_stopwords(parts: list) -> list:
        """parts را تا قبل از اولین کلمهٔ مزاحم کوتاه می‌کند."""
        cut = len(parts)
        for i, tok in enumerate(parts):
            if tok.lower() in STOPWORDS:
                cut = i
                break
        return parts[:cut]

    # 1) تلاش با NER اسپاسی
    if nlp:
        try:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    cleaned = strip_name_affixes(ent.text)
                    parts = [p for p in _keep_letters_spaces(cleaned).split() if p]
                    parts = cut_at_stopwords(parts)  # ← بریدن حتی در مسیر NER
                    if len(parts) >= 2:
                        first = parts[0].title()
                        last  = " ".join(parts[1:]).title()
                        return first, last
        except Exception:
            pass

    # 2) فال‌بک: خط‌به‌خط
    for ln in text.split("\n"):
        ln = ln.strip()
        if not ln:
            continue

        # اگر ":" یا "|" هست، فقط بخش قبل از آن‌ها را بگیر
        head = ln.split("|")[0]
        head = head.split(":")[0]

        cleaned = strip_name_affixes(head)
        parts = [p for p in _keep_letters_spaces(cleaned).split() if p]
        if len(parts) < 2:
            # اگر بعد از بریدنِ سر خط هنوز دو کلمه نشد، کل ln را امتحان کن
            cleaned = strip_name_affixes(ln)
            parts = [p for p in _keep_letters_spaces(cleaned).split() if p]
        parts = cut_at_stopwords(parts)

        if len(parts) >= 2:
            first = parts[0].title()
            last  = " ".join(parts[1:]).title()
            return first, last

    return "N/A", "N/A"

# --------- Degree keyword cache ---------
DEG_KEYS = {}
for label, syns, lvl in DEGREES:
    base = {label.lower(), *syns}
    ex = set()
    for term in base:
        ex |= _expand_one_term(term)
    DEG_KEYS[label] = ex
    
# ===== Auto synonym discovery (generic for ANY skills) =====
ALNUM_DASH = re.compile(r"[^a-z0-9\-\+\.# ]+")
STOP_TOKENS = {"and","or","of","the","a","an","in","on","for","with","to","by","at","from"}

def _norm(s: str) -> str:
    s = normalize_text(s).lower()
    s = s.replace("&"," and ").replace("/", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()

def _tokenize_words(s: str): 
    s = _norm(s)
    return [w for w in ALNUM_DASH.sub(" ", s).split() if w]

def _phrase_ok(ph: str) -> bool:
    if len(ph) < 3 or ph in STOP_TOKENS: return False
    if ph.isdigit() or ph.count(" ") > 3: return False
    return True


def _section_texts(txt: str) -> str:
    secs = split_into_sections(txt)
    return " \n ".join([
        secs.get('skills',''),
        secs.get('experience',''),
        secs.get('education',''),  # ← اضافه شد
        secs.get('other','')
    ])

def collect_corpus_candidates(resume_files, max_docs=200, max_vocab=5000):
    freq = {}
    docs = 0
    for f in resume_files:
        try:
            raw = extract_text_from_pdf(f)
            if not raw or len(raw) < 50: continue
            bag = _section_texts(normalize_text(raw).lower())
            words = _tokenize_words(bag)
            # 1-gram تا 3-gram
            L = len(words)
            seen = set()
            for n in (1,2,3):
                for i in range(L-n+1):
                    cand = " ".join(words[i:i+n])
                    if _phrase_ok(cand) and cand not in seen:
                        seen.add(cand)
                        freq[cand] = freq.get(cand, 0) + 1
            docs += 1
            if docs >= max_docs: break
        except Exception:
            continue
    return [p for p,_ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:max_vocab]]

def build_skill_synonyms_auto(
    req_skills: list[str],
    corpus_candidates: list[str],
    model,
    topk: int = 12,
    sim_threshold: float = 0.70,
    batch_size: int = 256
):
    """
    هر کاندید (phrase) به نزدیک‌ترین مهارتِ canon نسبت داده می‌شود.
    - aliasهای فنی متداول تزریق می‌شوند (C++, C#, Node.js, sklearn, ...).
    - اگر model موجود باشد، امبدینگ‌ها به صورت batch محاسبه می‌شوند.
    """

    def _batched_encode(items: list[str]):
        if not items:
            return None
        # SentenceTransformer با convert_to_tensor=True → Tensor
        tensors = []
        for i in range(0, len(items), batch_size):
            chunk = items[i:i+batch_size]
            t = model.encode(
                chunk,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            tensors.append(t)
        import torch  # مطمئن می‌شویم موجود است حتی اگر بالای فایل اضافه نشده باشد
        return torch.cat(tensors, dim=0) if tensors else None

    canonical_list = [s for s in (req_skills or []) if s and s.strip()]
    if not canonical_list:
        return {}, [], {}

    # --- 1) ساخت لیست کاندیدها از کورپوس + نرمال‌سازی فرم‌های رایج
    cand_list = [c for c in (corpus_candidates or []) if _phrase_ok(c)]
    for s in canonical_list:
        s_low = _norm(s)
        cand_list += [s_low, s_low.replace("-", " "), s_low.replace(" ", "-")]
    # یکتا
    cand_list = list(dict.fromkeys([_norm(c) for c in cand_list if c]))

    # --- 2) تزریق aliasهای فنی متداول (مورد ۸)
    EXTRA_TECH_ALIASES = {
        "c++": ["c++", "c plus plus"],
        "c#": ["c#", "c sharp"],
        "node.js": ["node.js", "node js", "nodejs"],
        "scikit-learn": ["scikit-learn", "sklearn"],
        "pytorch": ["pytorch", "torch"],
        "tensorflow": ["tensorflow", "tf"],
        "xgboost": ["xgboost", "xg boost"],
        "postgresql": ["postgresql", "postgres", "postgre sql"],
        "mysql": ["mysql", "my sql"],
        "mongodb": ["mongodb", "mongo db"],
        "matplotlib": ["matplotlib", "mpl"],
        "spss": ["spss"],
        "stata": ["stata"],
    }
    for canon in canonical_list:
        for a in EXTRA_TECH_ALIASES.get(_norm(canon), []):
            cand_list.append(_norm(a))

    # فرم‌های فاصله/خط‌تیره را هم تزریق/یکسان کنیم
    for s in list(cand_list):
        cand_list.append(s.replace("-", " "))
        cand_list.append(s.replace(" ", "-"))
    cand_list = list(dict.fromkeys([_norm(c) for c in cand_list if c]))

    # اگر مدل نداریم: فقط نگاشتِ همنوشته‌ها
    if model is None:
        skill_map = {canon: {_norm(canon)} for canon in canonical_list}
        alias2canon = {_norm(canon): canon for canon in canonical_list}
        # نگاشت‌های خاص فرم‌های رایج (بدون مدل هم مفیدن)
        for canon in canonical_list:
            key = _norm(canon)
            alias2canon.setdefault(key.replace("-", " "), canon)
            alias2canon.setdefault(key.replace(" ", "-"), canon)
            if key == "c++":
                alias2canon["c plus plus"] = canon
            if key == "c#":
                alias2canon["c sharp"] = canon
            if key == "node.js":
                alias2canon["node js"] = canon
                alias2canon["nodejs"]  = canon
            if key == "scikit-learn":
                alias2canon["sklearn"] = canon
        # تبدیل به skill_map
        for a, c in alias2canon.items():
            skill_map.setdefault(c, set()).add(a)
        return skill_map, canonical_list, alias2canon

    # --- 3) امبدینگ‌ها (Batch)
    skill_embs = _batched_encode(canonical_list)
    cand_embs  = _batched_encode(cand_list)
    sims = util.cos_sim(skill_embs, cand_embs).cpu().numpy()

    # --- 4) نسبت‌دادن هر کاندید به بهترین canon با آستانه
    alias2canon: dict[str, str] = {}
    for j, cand in enumerate(cand_list):
        col = sims[:, j]                 # شباهت cand به همه‌ی canonها
        i_best = int(np.argmax(col))
        best_sim = float(col[i_best])
        if best_sim >= sim_threshold:
            alias2canon[cand] = canonical_list[i_best]

    # canonهای پایه حتماً نگاشت داشته باشند
    for canon in canonical_list:
        alias2canon.setdefault(_norm(canon), canon)

    # --- 5) نگاشت‌های خاصِ فرم‌های پراستفاده (مورد ۸)
    for canon in canonical_list:
        key = _norm(canon)
        alias2canon.setdefault(key.replace("-", " "), canon)
        alias2canon.setdefault(key.replace(" ", "-"), canon)
        if key == "c++":
            alias2canon["c plus plus"] = canon
        if key == "c#":
            alias2canon["c sharp"] = canon
        if key == "node.js":
            alias2canon["node js"] = canon
            alias2canon["nodejs"]  = canon
        if key == "scikit-learn":
            alias2canon["sklearn"] = canon

    # --- 6) ساخت skill_map از alias2canon
    skill_map: dict[str, set] = {canon: set() for canon in canonical_list}
    for alias, canon in alias2canon.items():
        skill_map[canon].add(alias)

    # --- 7) نگه‌داشتن top-k aliasها بر اساس شباهت
    if topk and topk > 0:
        for idx_canon, canon in enumerate(canonical_list):
            # اندیس‌های کاندیدهایی که به این canon نسبت داده شده‌اند
            idxs = [i for i, cand in enumerate(cand_list) if cand in skill_map[canon]]
            # مرتب‌سازی بر پایه شباهت این canon به candها
            order = sorted(idxs, key=lambda k: sims[idx_canon, k], reverse=True)[:topk]
            kept = {cand_list[k] for k in order}
            # برش skill_map و هم‌زمان پاکسازی alias2canon اضافه
            for a in list(skill_map[canon]):
                if a not in kept:
                    skill_map[canon].remove(a)
                    if alias2canon.get(a) == canon:
                        del alias2canon[a]

    return skill_map, canonical_list, alias2canon

def extract_skills(
    raw: str,
    req_skills: list,
    use_semantic: bool = True,
    skill_map: dict | None = None,
    alias2canon: dict | None = None,
    canonical_list: list | None = None
) -> list:
    if not req_skills:
        return ['N/A']

    canonical_list = [s for s in (canonical_list or req_skills) if s and s.strip()]
    if not (skill_map and alias2canon and canonical_list):
        alias2canon = {_norm(s): s for s in canonical_list}
        skill_map   = {s: {_norm(s)} for s in canonical_list}

    text = normalize_text(raw)
    secs = split_into_sections(text)

    # نسخه‌های نرمال‌شده برای exact/regex
    skills_txt_norm = _norm(secs.get('skills', '') or '')
    exp_txt_norm    = _norm(secs.get('experience', '') or '')
    edu_txt_norm    = _norm(secs.get('education', '') or '')

    # نسخه‌های خام برای semantic (به خط‌ها نیاز داریم)
    skills_txt_raw = secs.get('skills', '') or ''
    exp_txt_raw    = secs.get('experience', '') or ''
    edu_txt_raw    = secs.get('education', '') or ''

    found = set()

    def _scan(block: str, allow_semantic: bool):
        # اگر بلاک خام بود، برای exact یک نسخه نرمال هم می‌سازیم
        is_norm = ("\n" not in block)
        norm_block = block if is_norm else _norm(block)

        # --- 1) exact/alias روی متن نرمال‌شده
        for alias, canon in alias2canon.items():
            if " " in alias:
                if alias in norm_block:
                    found.add(canon)
            else:
                if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", norm_block):
                    found.add(canon)

        # --- 2) semantic فقط اگر فعال باشد و بلاک خام باشد
        if allow_semantic and not is_norm and st_model is not None:
            lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
            for canon in canonical_list:
                if canon in found:
                    continue
                ok_line = False
                aliases = skill_map.get(canon, set())
                for ln in lines:
                    # اگر یکی از aliasها در خط بود آستانه پایین‌تر
                    if any(a in _norm(ln) for a in aliases):
                        ok, _ = semantic_similarity(canon, ln, threshold=0.67)
                    else:
                        ok, _ = semantic_similarity(canon, ln, threshold=0.72)
                    if ok:
                        ok_line = True
                        break
                if ok_line:
                    found.add(canon)

    # --- ترتیب اسکن ---
    # exact/alias روی نسخه‌های نرمال
    _scan(skills_txt_norm, allow_semantic=False)
    _scan(exp_txt_norm,    allow_semantic=False)
    _scan(edu_txt_norm,    allow_semantic=False)

    # semantic روی نسخه‌های خام (اگر use_semantic فعال است)
    if use_semantic and st_model is not None:
        _scan(skills_txt_raw, allow_semantic=True)

    matched = [c for c in canonical_list if c in found]
    return matched if matched else ['N/A']

def parse_date_str(date_str: str):
    try:
        return dateparser.parse(
            date_str,
            settings={
                "PREFER_DATES_FROM": "past",  # تاریخ‌های شغلی معمولاً گذشته‌اند
                "DATE_ORDER": "DMY"           # رفع ابهام 03/04/2021 → 3 آوریل 2021
            }
        )
    except Exception:
        return None

def parse_gpa_block(text: str):
    txt = normalize_text(text).lower()
    txt = txt.replace("⁄", "/").replace("／", "/")

    gpa_candidates = []

    # 1) نسبت‌ها مثل 3.7/4.0 یا 82/100
    for m in re.finditer(r'([0-9]+(?:\.[0-9]{1,2})?)\s*/\s*([0-9]+(?:\.[0-9]{1,2})?)', txt):
        try:
            raw = float(m.group(1)); scale = float(m.group(2))
            if 0.0 < raw <= scale <= 100:
                gpa_candidates.append(("ratio", round((raw/scale)*4.0, 2)))
        except:
            pass

    # 2) درصدِ برچسب‌دار مثل GPA: 85%
    for m in re.finditer(r'(?:gpa|cgpa|grade|score)\s*[:\-]?\s*([0-9]{1,3}(?:\.[0-9]{1,2})?)\s*%', txt):
        try:
            val = float(m.group(1))
            if 0 <= val <= 100:
                gpa_candidates.append(("percent_labeled", round((val/100.0)*4.0, 2)))
        except:
            pass

    # 3) مقدار تکیِ برچسب‌دار مثل GPA: 3.7
    for m in re.finditer(r'(?:gpa|cgpa|grade|score)\s*[:\-]?\s*([0-9]{1,2}(?:\.[0-9]{1,2})?)\b', txt):
        try:
            val = float(m.group(1))
            if 0 <= val <= 4.0:
                gpa_candidates.append(("single_labeled", round(val, 2)))
        except:
            pass

    if not gpa_candidates:
        return 'N/A'

    # ✅ انتخاب بهترین: اولویت «نسبت» > «درصدِ برچسب‌دار» > «مقدارِ برچسب‌دار»،
    # و داخل هر دسته «بیشترین مقدار»
    priority = {"ratio": 3, "percent_labeled": 2, "single_labeled": 1}
    gpa_candidates.sort(key=lambda t: (priority[t[0]], t[1]), reverse=True)
    return gpa_candidates[0][1]



def merge_intervals(intervals: list) -> list:
    if not intervals:
        return []
    sorted_ints = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_ints[0]]
    for start, end in sorted_ints[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged

def extract_experience_years(raw: str) -> float:
    secs = split_into_sections(raw)
    exp_text = secs.get('experience', '')
    patterns = [
    r'([A-Za-z]{3,9}\s*\d{4})\s*(?:-|–|—|to)\s*(Present|Now|Current|[A-Za-z]{3,9}\s*\d{4})',
    r'(\d{1,2}/\d{4})\s*(?:-|–|—|to)\s*(Present|Now|Current|\d{1,2}/\d{4})',
    r'(\d{4})\s*(?:-|–|—|to)\s*(Present|Now|Current|\d{4})'
    ]

    now = datetime.now()
    intervals = []
    for pat in patterns:
        for s, e in re.findall(pat, exp_text, flags=re.IGNORECASE):
            ds = parse_date_str(s)
            de = now if re.search(r'present|now|current', e, flags=re.IGNORECASE) else parse_date_str(e)
            if ds and de and de >= ds:
                intervals.append((ds, de))
    merged = merge_intervals(intervals)
    total_months = sum((end.year - start.year) * 12 + (end.month - start.month) for start, end in merged)
    return round(total_months / 12, 2)

def extract_education(raw: str) -> dict:
    """
    فقط GPA مربوط به بالاترین مدرک را برمی‌گرداند.
    خروجی: {'degree': <label>, 'level': <1..5>, 'gpa': <4.0-scale or 'N/A'>}
    """
    secs = split_into_sections(raw)
    edu_text = secs.get('education', '') or raw

    lines = [ln for ln in edu_text.split('\n') if ln.strip()]
    blocks = []  # [(start, end, degree_label, level)]

    # بلاک‌های هر مدرک را پیدا کن (پنجره‌ای از چند خط اطرافش)
    for idx, line in enumerate(lines):
        low_line = line.lower()
        for label, syns, lvl in DEGREES:
            keys = DEG_KEYS[label]
            if any(re.search(rf"\b{re.escape(k)}\b", low_line) for k in keys):
                start = max(0, idx - 2)
                end   = min(len(lines), idx + 4)
                blocks.append((start, end, label, lvl))
                break

    if not blocks:
        # هیچ مدرکی پیدا نشد
        return {'degree': 'N/A', 'level': 0, 'gpa': 'N/A'}

    # بالاترین مدرک
    blocks.sort(key=lambda b: b[3], reverse=True)
    top_start, top_end, top_label, top_level = blocks[0]

    # فقط در بلاک همان مدرک به‌دنبال GPA بگرد
    window_text = '\n'.join(lines[top_start:top_end])
    gpa4 = parse_gpa_block(window_text)  # 'N/A' یا عدد روی مقیاس 4.0

    return {'degree': top_label, 'level': top_level, 'gpa': gpa4}


# --- Strict major matching: Education-only, phrase-level ---
def _norm_phrase(s: str) -> str:
    s = normalize_text(s).lower()
    s = s.replace("&", " and ")
    s = s.replace("/", " ")
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _contains_phrase(text_norm: str, phrase_norm: str) -> bool:
    """اگر phrase چندکلمه‌ای بود: جستجوی ساده؛
       اگر تک‌واژه بود: با مرزبندی واژه (word boundary)."""
    if " " in phrase_norm:
        return phrase_norm in text_norm
    return re.search(rf"(?<!\w){re.escape(phrase_norm)}(?!\w)", text_norm) is not None

def extract_major_pct(raw: str, keywords: list) -> float:
    """
    فقط وقتی یکی از عبارات رشتهٔ آگهی در بخش Education رزومه دیده شود، 100%.
    در غیر این صورت 0%.
    """
    if not keywords:
        return 0.0

    # فقط بخش Education رزومه
    secs = split_into_sections(raw)
    edu_text = secs.get('education', '') or ''
    if not edu_text.strip():
        return 0.0

    txt = _norm_phrase(edu_text)

    for kw in keywords:
        if not kw:
            continue
        kw_norm = _norm_phrase(kw)
        if not kw_norm:
            continue
        if _contains_phrase(txt, kw_norm):
            return 100.0

    return 0.0

def score_candidate(raw: str, req: dict, weights: dict, use_semantic: bool=True) -> dict:
    found = extract_skills(
        raw,
        req.get('orig_skills', []),
        use_semantic=use_semantic,
        skill_map=req.get('_skill_synonyms'),
        alias2canon=req.get('_skill_alias2canon'),
        canonical_list=req.get('_skill_canon_list')
    )

    canon_orig = req.get('orig_skills', [])
    if found and found != ['N/A']:
        skill_pct = round(len(set(found)) / max(1, len(canon_orig)) * 100, 2)
    else:
        skill_pct = 0.0

    # ⬇ اینا باید همیشه حساب بشن (چه found باشه چه نه)
    yrs = extract_experience_years(raw)
    exp_req = req.get('min_experience', 0)
    exp_pct = round(min(yrs, exp_req) / exp_req * 100, 2) if exp_req > 0 else 0.0

    edu = extract_education(raw)
    deg_pct = round(edu['level'] / 5 * 100, 2)

    gpa_req = req.get('min_gpa')
    gpa_pct = 0.0
    if gpa_req and edu['gpa'] != 'N/A':
       gpa_pct = round(min(float(edu['gpa']), gpa_req) / gpa_req * 100, 2)

    edu_pct = round((deg_pct + gpa_pct) / 2, 2)
    
    
    major_pct = extract_major_pct(raw, req.get('required_major_keywords', []))

    overall_pct = round(
        skill_pct * weights['skills'] +
        exp_pct   * weights['experience'] +
        edu_pct   * weights['education'] +
        major_pct * weights['major'],
        2
    )

    return {
        'matched_skills': ', '.join(found) if found and found != ['N/A'] else 'N/A',
        'skill_pct': skill_pct,
        'experience_years': yrs,
        'experience_pct': exp_pct,
        'degree': edu['degree'],
        'gpa': edu['gpa'],
        'education_pct': edu_pct,
        'major_pct': major_pct,
        'overall_pct': overall_pct
    }

def process_resume(resume_file, job_req, weights, use_semantic: bool=True):
    raw_text = extract_text_from_pdf(resume_file)
    if not raw_text or len(raw_text.strip()) < 50:
        st.warning(f"Could not extract sufficient text from {resume_file.name}")
        return None
    email = extract_email(raw_text)
    phone = extract_phone(raw_text)
    first_name, last_name = extract_name(raw_text)
    scores = score_candidate(raw_text, job_req, weights, use_semantic=use_semantic)
    return {
        'filename': resume_file.name,
        'position': job_req['position_title'],
        'first_name': first_name,
        'last_name': last_name,
        'email': email,
        'phone': phone,
        **scores,
    }

# ---------- UI ----------

def main():
    st.set_page_config(page_title="Resume Matcher", layout="wide")

    st.markdown("""
        <style>
        .main-title { font-size: 1.13rem !important; font-weight: 700; margin-bottom: 3px; letter-spacing: -0.5px; text-align: left; }
        .sub-title { font-size: 0.98rem !important; font-weight: 400; margin-bottom: 0px; color: #444; text-align: left; }
        html, body, .stApp { font-size: 14px !important; }
        .stButton>button, .stDownloadButton>button, .stSelectbox>div, .stSlider { font-size: 13px !important; }
        .stDataFrame table, .stDataFrame thead, .stDataFrame tbody, .stDataFrame tr, .stDataFrame td, .stDataFrame th { font-size: 12px !important; }
        </style>
    """, unsafe_allow_html=True)

    # Header
    col1, col2, col3 = st.columns([2, 3, 5])
    with col1:
        try:
            st.image("images.png", width=52)
        except Exception:
            pass
        st.markdown("<div class='main-title'>Resume Matcher</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-title'>Multi-Resume Job Matching Analysis</div>", unsafe_allow_html=True)

    st.sidebar.header("📃 Upload Job Requirements")
    job_file = st.sidebar.file_uploader("📤 Upload Job Posting JSON", type="json")

    st.sidebar.markdown("### 📌 Sample Job JSON")
    sample_json = {
        "position_title": "Data Scientist",
        "skills": ["Python", "Machine Learning", "Data Analysis", "Statistics", "SQL"],
        "min_experience": 3,
        "min_gpa": 3.0,
        "required_major_keywords": ["Computer Science", "Statistics", "Data Science", "computer engineering"]
    }
    st.sidebar.download_button(
        label="📥 Download Sample",
        data=json.dumps(sample_json, indent=2),
        file_name="sample_job_posting.json",
        mime="application/json"
    )

    st.sidebar.header("⚖️ Matching Weights")
    skills_weight     = st.sidebar.slider("Skills Weight", 0.0, 1.0, 0.5, 0.01)
    experience_weight = st.sidebar.slider("Experience Weight", 0.0, 1.0, 0.25, 0.01)
    education_weight  = st.sidebar.slider("Education Weight", 0.0, 1.0, 0.15, 0.01)
    major_weight      = st.sidebar.slider("Major Weight", 0.0, 1.0, 0.10, 0.01)

    st.sidebar.header("⚙️ Options")
    use_semantic = st.sidebar.checkbox("Enable semantic skill matching (SBERT)", value=True)
  
    # 🔎 وضعیت لود بودن SBERT (داخل main)
    st.sidebar.markdown("### 🔎 Runtime status")
    if st_model is not None:
        st.sidebar.write("SBERT loaded: **True**")
        try:
            a = st_model.encode("machine learning", convert_to_tensor=True, normalize_embeddings=True)
            b = st_model.encode("deep learning", convert_to_tensor=True, normalize_embeddings=True)
            sim = float(util.pytorch_cos_sim(a, b).item())
            st.sidebar.write(f"SBERT sanity similarity (ML vs DL): **{sim:.2f}**")
        except Exception as e:
            st.sidebar.warning(f"SBERT test failed: {e}")
    else:
        st.sidebar.write("SBERT loaded: **False**")
        st.sidebar.info("Tip: pre-download the model into ./models/all-MiniLM-L6-v2 or ensure internet/torch is available.")

    
    # Normalize weights
    sum_weights = skills_weight + experience_weight + education_weight + major_weight
    if abs(sum_weights - 1.0) > 1e-6:
        skills_weight     /= sum_weights
        experience_weight /= sum_weights
        education_weight  /= sum_weights
        major_weight      /= sum_weights
        st.sidebar.info("Normalized weights to sum = 1.0")
    weights = {
        'skills': skills_weight,
        'experience': experience_weight,
        'education': education_weight,
        'major': major_weight
    }

    st.header("📊 Upload Resumes")
    resume_files = st.file_uploader(
        "📤 Upload multiple resumes (PDF)",
        type="pdf",
        accept_multiple_files=True
    )

    analyze_clicked = st.button("Analyze All Resumes")

    if analyze_clicked:
        if not job_file:
            st.warning("Please upload a Job Posting JSON first in the sidebar!")
            return
        if not resume_files:
            st.warning("Please upload at least one resume (PDF).")
            return

        try:
            job_req = json.load(job_file)
            required_fields = ['position_title', 'skills', 'min_experience']
            for field in required_fields:
                if field not in job_req:
                    st.error(f"Invalid job posting JSON: Missing '{field}' field")
                    return

            # مهارت‌های اصلی برای درصد
            job_req['orig_skills'] = job_req.get('skills', [])[:]
            # ⚠️ برای major expand نکن: false positive می‌دهد
            # job_req['required_major_keywords'] = expand_keywords(job_req.get('required_major_keywords', []))
            # --- Controls for auto-synonyms (debug/tuning) ---
            sim_threshold = st.sidebar.slider("Synonym sim threshold", 0.5, 0.9, 0.75, 0.01)
            topk = st.sidebar.slider("Synonym top-k per skill", 5, 30, 15, 1)

            # --- ساخت هم‌معنی‌های داینامیک از روی متن رزومه‌ها ---
            st.info("Building dynamic skill synonyms from resumes…")
            candidates = collect_corpus_candidates(resume_files, max_docs=120, max_vocab=5000)
            skill_map, canon_list, alias2canon = build_skill_synonyms_auto(
            job_req['orig_skills'], candidates, st_model, topk=topk, sim_threshold=sim_threshold
            )

            job_req['_skill_synonyms']    = skill_map
            job_req['_skill_canon_list']  = canon_list
            job_req['_skill_alias2canon'] = alias2canon

            # ---- Process resumes (ساخت results و df) ----
            results = []
            progress_bar = st.progress(0)
            status_text  = st.empty()

            for i, resume_file in enumerate(resume_files):
                # اطمینان از خواندن مجدد از ابتدای فایل
                try:
                    resume_file.seek(0)
                except Exception:
                    pass
                status_text.text(f"Processing {i+1}/{len(resume_files)}: {resume_file.name}")
                result = process_resume(resume_file, job_req, weights, use_semantic=use_semantic)
                progress_bar.progress((i + 1) / max(1, len(resume_files)))
                if result:
                    results.append(result)

            if not results:
                st.error("No valid resumes could be processed.")
                return

            df = pd.DataFrame(results).fillna('N/A')
            df = df.sort_values('overall_pct', ascending=False)

            # Robust candidate label for charts
            def _label(row):
                name = (row['first_name'] if row['first_name']!='N/A' else '') + ' ' + (row['last_name'] if row['last_name']!='N/A' else '')
                name = name.strip()
                if not name:
                    name = row.get('email') or row.get('filename') or 'Candidate'
                return name
            df['candidate_id'] = df.apply(_label, axis=1)

            st.success("Resumes analyzed!")

            st.subheader("📊 Summary Results")
            st.dataframe(df[[
                'filename', 'position', 'first_name', 'last_name',
                'skill_pct', 'experience_pct', 'education_pct', 'major_pct', 'overall_pct'
            ]].rename(columns={
                'filename': 'Resume',
                'position': 'Position',
                'first_name': 'First Name',
                'last_name': 'Last Name',
                'skill_pct': 'Skills %',
                'experience_pct': 'Experience %',
                'education_pct': 'Education %',
                'major_pct': 'Major %',
                'overall_pct': 'Match Score'
            }), height=400)

            st.subheader("📊 Comparison Chart")
            fig = go.Figure(data=[
                go.Bar(name='Skills Match',     x=df['candidate_id'], y=df['skill_pct']),
                go.Bar(name='Experience Match', x=df['candidate_id'], y=df['experience_pct']),
                go.Bar(name='Overall Match',    x=df['candidate_id'], y=df['overall_pct']),
            ])
            fig.update_layout(
                barmode='group',
                xaxis_title="Candidates",
                yaxis_title="Score (%)",
                legend_title="Metrics",
                height=450,
                template="simple_white"
            )
            st.plotly_chart(fig, use_container_width=True)

            num_top = min(3, len(df))
            st.subheader(f"🏆 Top {num_top} Candidates")
            for i in range(num_top):
                candidate = df.iloc[i]
                with st.expander(f"{i+1}. {candidate['first_name']} {candidate['last_name']} - Score: {candidate['overall_pct']}%"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Candidate Information**")
                        st.write(f"**Name:** {candidate['first_name']} {candidate['last_name']}")
                        st.write(f"**Email:** {candidate['email']}")
                        st.write(f"**Phone:** {candidate['phone']}")
                        st.write(f"**Degree:** {candidate['degree']}")
                        st.write(f"**GPA:** {candidate['gpa']}")
                        st.write(f"**Position:** {candidate['position']}")
                    with col2:
                        st.markdown("**Match Scores**")
                        st.metric("Overall Match", f"{candidate['overall_pct']}%")
                        st.metric("Skills Match", f"{candidate['skill_pct']}%")
                        st.metric("Experience Match", f"{candidate['experience_pct']}%")
                        st.metric("Education Match", f"{candidate['education_pct']}%")
                        st.metric("Major Relevance", f"{candidate['major_pct']}%")
                    st.markdown("**Matched Skills**")
                    st.write(candidate['matched_skills'])

            st.subheader("💾 Download Results")
            csv = df.drop(columns=['candidate_id']).to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name='resume_matching_results.csv',
                mime='text/csv'
            )

        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please upload a valid job posting JSON file.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

