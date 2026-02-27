# -*- coding: utf-8 -*-

import hashlib
import json
import re
import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
try:
    from langchain_core.documents import Document
except ModuleNotFoundError:
    @dataclass
    class Document:
        page_content: str
        metadata: dict


_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

_NPC_API_HEADERS = {
    "accept": "application/json, text/javascript, */*; q=0.01",
    "x-requested-with": "XMLHttpRequest",
    "user-agent": _REQUEST_HEADERS["User-Agent"],
    "referer": "https://flk.npc.gov.cn/fl.html",
}

_CANDIDATE_CONTENT_SELECTORS = (
    "article",
    ".article",
    ".article-content",
    ".content",
    ".detail-content",
    ".law-content",
    ".zw",
    "#zoom",
    "#UCAP-CONTENT",
)

_LEGAL_LINK_KEYWORDS = (
    "law",
    "laws",
    "flfg",
    "fgk",
    "法规",
    "法律",
    "条例",
    "司法解释",
)

_LEGAL_TITLE_KEYWORDS = (
    "法",
    "条例",
    "规定",
    "司法解释",
    "解释",
    "办法",
    "决定",
)

_TITLE_FIELDS = (
    "title",
    "name",
    "law_name",
    "doc_title",
    "标题",
    "名称",
)

_CONTENT_FIELDS = (
    "content",
    "text",
    "body",
    "law_content",
    "全文",
    "正文",
)

_TEXT_SUFFIXES = (".txt", ".md", ".markdown")
_JSON_SUFFIXES = (".json", ".jsonl")

_IGNORED_MD_FILE_NAMES = {"_index.md", "README.md", "README_CN.md", "法律法规模版.md"}
_IGNORED_REPO_DIRS = {".git", ".github", "scripts", "__cache__"}

# 来源于 https://github.com/lhlhlhlhl/legal_rag/blob/main/seed.ts
LEGAL_RAG_SEED_URLS = [
    "http://www.gov.cn/zhengce/",
    "http://www.gov.cn/zhengce/xxgk/",
    "http://www.npc.gov.cn/npc/c30834/",
    "http://www.npc.gov.cn/npc/c30834/202006/t20200602_306801.html",
    "http://www.chinacourt.org/law.shtml",
    "http://www.chinacourt.org/article/list/",
    "http://www.legalinfo.gov.cn/",
    "http://www.legalinfo.gov.cn/pub/sfzhw/",
    "http://www.moj.gov.cn/",
    "http://www.moj.gov.cn/pub/sfbgw/",
    "http://www.12348.gov.cn/",
    "http://www.12348.gov.cn/pub/m12348/",
]


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_markdown_text(text: str) -> str:
    text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, flags=re.DOTALL)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = _normalize_text(text)
    return text


def _md5(*parts: str) -> str:
    merged = "||".join([p or "" for p in parts])
    return hashlib.md5(merged.encode("utf-8")).hexdigest()


def _build_metadata(unique_id: str, source: str, page: int, title: str, doc_type: str, extra: Optional[dict] = None) -> dict:
    metadata = {
        "unique_id": unique_id,
        "source": source,
        "page": page,
        "images_info": [],
        "title": title,
        "doc_type": doc_type,
    }
    if extra:
        metadata.update(extra)
    return metadata


def _extract_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag.get("href", "").strip()
        if not href:
            continue
        abs_url = urljoin(base_url, href)
        abs_url = urldefrag(abs_url).url
        if abs_url.startswith("http://") or abs_url.startswith("https://"):
            links.append(abs_url)
    return links


def _is_same_domain(url: str, allowed_domains: set[str]) -> bool:
    netloc = urlparse(url).netloc.lower()
    return any(netloc == domain or netloc.endswith("." + domain) for domain in allowed_domains)


def _is_candidate_link(url: str, anchor_text: str = "") -> bool:
    haystack = (url + " " + anchor_text).lower()
    return any(keyword in haystack for keyword in _LEGAL_LINK_KEYWORDS)


def _extract_page_title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    if soup.title and soup.title.get_text(strip=True):
        return soup.title.get_text(strip=True)
    return ""


def _extract_page_content(soup: BeautifulSoup) -> str:
    for selector in _CANDIDATE_CONTENT_SELECTORS:
        node = soup.select_one(selector)
        if node:
            text = _normalize_text(node.get_text("\n", strip=True))
            if len(text) >= 120:
                return text
    return _normalize_text(soup.get_text("\n", strip=True))


def _looks_like_law_doc(url: str, title: str, content: str, min_content_chars: int) -> bool:
    if len(content) < min_content_chars:
        return False
    title_hit = any(keyword in title for keyword in _LEGAL_TITLE_KEYWORDS)
    url_hit = any(keyword in url.lower() for keyword in _LEGAL_LINK_KEYWORDS)
    return title_hit or url_hit


def crawl_official_legal_docs(
    start_urls: list[str],
    max_docs: int = 200,
    max_depth: int = 2,
    min_content_chars: int = 200,
    timeout: int = 20,
    sleep_seconds: float = 0.2,
) -> list[Document]:
    if not start_urls:
        return []

    allowed_domains = {urlparse(url).netloc.lower() for url in start_urls if urlparse(url).netloc}
    queue: deque[tuple[str, int]] = deque((url, 0) for url in start_urls)
    visited = set()
    documents = []
    session = requests.Session()
    session.headers.update(_REQUEST_HEADERS)

    while queue and len(documents) < max_docs:
        url, depth = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        try:
            response = session.get(url, timeout=timeout)
            if response.status_code != 200:
                continue
        except requests.RequestException:
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        title = _extract_page_title(soup)
        content = _extract_page_content(soup)

        if _looks_like_law_doc(url, title, content, min_content_chars):
            unique_id = _md5("official_crawl", url, title, content[:1000])
            metadata = _build_metadata(
                unique_id=unique_id,
                source=url,
                page=len(documents) + 1,
                title=title,
                doc_type="official_crawl",
                extra={"crawl_time": int(time.time())},
            )
            documents.append(Document(page_content=content, metadata=metadata))

        if depth < max_depth:
            for tag in soup.find_all("a", href=True):
                href = tag.get("href", "").strip()
                if not href:
                    continue
                abs_url = urldefrag(urljoin(url, href)).url
                if abs_url in visited:
                    continue
                if not _is_same_domain(abs_url, allowed_domains):
                    continue
                if _is_candidate_link(abs_url, tag.get_text(" ", strip=True)):
                    queue.append((abs_url, depth + 1))

        time.sleep(sleep_seconds)

    return documents


def collect_legal_rag_seed_docs(
    max_docs: int = 200,
    timeout: int = 20,
    min_content_chars: int = 200,
) -> list[Document]:
    session = requests.Session()
    session.headers.update(_REQUEST_HEADERS)
    docs = []

    for url in LEGAL_RAG_SEED_URLS:
        if len(docs) >= max_docs:
            break
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code != 200:
                continue
            soup = BeautifulSoup(response.text, "html.parser")
            title = _extract_page_title(soup)
            content = _extract_page_content(soup)
        except requests.RequestException:
            continue

        if len(content) < min_content_chars:
            continue
        unique_id = _md5("legal_rag_seed", url, title, content[:1000])
        metadata = _build_metadata(
            unique_id=unique_id,
            source=url,
            page=len(docs) + 1,
            title=title,
            doc_type="official_legal_rag_seed",
        )
        docs.append(Document(page_content=content, metadata=metadata))

    return docs


def _first_non_empty(record: dict, fields: tuple[str, ...]) -> str:
    for key in fields:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_json_records(node) -> list[dict]:
    records = []
    if isinstance(node, list):
        for item in node:
            records.extend(_extract_json_records(item))
    elif isinstance(node, dict):
        title = _first_non_empty(node, _TITLE_FIELDS)
        content = _first_non_empty(node, _CONTENT_FIELDS)
        if content:
            records.append({"title": title, "content": content})
        for value in node.values():
            records.extend(_extract_json_records(value))
    return records


def _read_text_with_fallback(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_markdown_title(path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
            if stripped:
                return stripped
    return path.stem


def _is_laws_markdown_file(path: Path, root: Path) -> bool:
    if path.suffix.lower() != ".md":
        return False
    if path.name in _IGNORED_MD_FILE_NAMES:
        return False

    rel_parts = path.relative_to(root).parts
    if not rel_parts:
        return False
    if rel_parts[0] in _IGNORED_REPO_DIRS:
        return False
    return True


def _docs_from_laws_markdown(path: Path, root: Path, doc_type: str) -> list[Document]:
    raw_text = _read_text_with_fallback(path)
    text = _clean_markdown_text(raw_text)
    if len(text) < 80:
        return []
    title = _extract_markdown_title(path, text)
    rel_parts = path.relative_to(root).parts
    category = rel_parts[0] if len(rel_parts) >= 1 else ""
    subcategory = rel_parts[1] if len(rel_parts) >= 2 else ""

    unique_id = _md5(doc_type, str(path), title, text[:1000])
    metadata = _build_metadata(
        unique_id=unique_id,
        source=str(path),
        page=1,
        title=title,
        doc_type=doc_type,
        extra={
            "category": category,
            "subcategory": subcategory,
            "source_repo": "https://github.com/LawRefBook/Laws",
        },
    )
    return [Document(page_content=text, metadata=metadata)]


def _docs_from_text_file(path: Path, doc_type: str) -> list[Document]:
    text = _normalize_text(_read_text_with_fallback(path))
    if not text:
        return []
    title = path.stem
    for line in text.splitlines():
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            if heading:
                title = heading
                break
    unique_id = _md5(doc_type, str(path), title, text[:1000])
    metadata = _build_metadata(
        unique_id=unique_id,
        source=str(path),
        page=1,
        title=title,
        doc_type=doc_type,
    )
    return [Document(page_content=text, metadata=metadata)]


def _docs_from_json_file(path: Path, doc_type: str) -> list[Document]:
    text = _read_text_with_fallback(path)
    docs = []

    if path.suffix.lower() == ".jsonl":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for record in _extract_json_records(obj):
                content = _normalize_text(record.get("content", ""))
                if len(content) < 80:
                    continue
                title = record.get("title") or f"{path.stem}_{index}"
                unique_id = _md5(doc_type, str(path), str(index), title, content[:1000])
                metadata = _build_metadata(
                    unique_id=unique_id,
                    source=str(path),
                    page=index + 1,
                    title=title,
                    doc_type=doc_type,
                )
                docs.append(Document(page_content=content, metadata=metadata))
        return docs

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return []

    records = _extract_json_records(obj)
    for index, record in enumerate(records):
        content = _normalize_text(record.get("content", ""))
        if len(content) < 80:
            continue
        title = record.get("title") or f"{path.stem}_{index}"
        unique_id = _md5(doc_type, str(path), str(index), title, content[:1000])
        metadata = _build_metadata(
            unique_id=unique_id,
            source=str(path),
            page=index + 1,
            title=title,
            doc_type=doc_type,
        )
        docs.append(Document(page_content=content, metadata=metadata))

    return docs


def _run_cmd(command: list[str], cwd: Optional[str] = None):
    subprocess.run(command, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def prepare_laws_repo(
    laws_dir: Optional[str],
    laws_repo_url: Optional[str],
    laws_repo_branch: str = "master",
    laws_repo_local_dir: str = ".cache/laws_repo",
) -> str:
    if laws_dir:
        return laws_dir

    if not laws_repo_url:
        raise ValueError("laws_dir 或 laws_repo_url 至少提供一个")

    local_path = Path(laws_repo_local_dir).resolve()
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if (local_path / ".git").exists():
        _run_cmd(["git", "-C", str(local_path), "fetch", "--depth", "1", "origin", laws_repo_branch])
        _run_cmd(["git", "-C", str(local_path), "checkout", laws_repo_branch])
        _run_cmd(["git", "-C", str(local_path), "pull", "--ff-only", "origin", laws_repo_branch])
    else:
        if local_path.exists():
            shutil.rmtree(local_path)
        _run_cmd(["git", "clone", "--depth", "1", "--branch", laws_repo_branch, laws_repo_url, str(local_path)])

    return str(local_path)


def load_open_source_laws(laws_dir: str, max_docs: int = 5000) -> list[Document]:
    root = Path(laws_dir)
    if not root.exists():
        raise FileNotFoundError(f"Laws 数据目录不存在: {laws_dir}")

    docs = []

    md_files = sorted(root.rglob("*.md"))
    for path in md_files:
        if not _is_laws_markdown_file(path, root):
            continue
        docs.extend(_docs_from_laws_markdown(path, root, doc_type="open_source_laws"))
        if len(docs) >= max_docs:
            return docs[:max_docs]

    if docs:
        return docs[:max_docs]

    files = []
    for suffix in _TEXT_SUFFIXES + _JSON_SUFFIXES:
        files.extend(root.rglob(f"*{suffix}"))
    files = sorted(set(files))

    for path in files:
        suffix = path.suffix.lower()
        if suffix in _TEXT_SUFFIXES:
            parsed = _docs_from_text_file(path, doc_type="open_source_laws")
        elif suffix in _JSON_SUFFIXES:
            parsed = _docs_from_json_file(path, doc_type="open_source_laws")
        else:
            parsed = []
        docs.extend(parsed)
        if len(docs) >= max_docs:
            break

    dedup_docs = []
    seen = set()
    for doc in docs[:max_docs]:
        unique_id = doc.metadata.get("unique_id")
        if unique_id and unique_id not in seen:
            seen.add(unique_id)
            dedup_docs.append(doc)
    return dedup_docs


def _npc_parse_html(content_html: str) -> str:
    soup = BeautifulSoup(content_html, "html.parser")
    node = soup.select_one(".law-content") or soup.select_one("body")
    if not node:
        return ""
    parts = []
    for p in node.find_all(["p", "div"]):
        text = p.get_text(" ", strip=True)
        if text:
            parts.append(text)
    if not parts:
        parts = [node.get_text("\n", strip=True)]
    return _normalize_text("\n".join(parts))


def fetch_npc_laws_via_api(
    max_docs: int = 200,
    max_pages: int = 20,
    page_size: int = 10,
    search_type: str = "1,9",
    timeout: int = 20,
    sleep_seconds: float = 0.2,
) -> list[Document]:
    docs = []
    session = requests.Session()
    session.headers.update(_NPC_API_HEADERS)
    now_ms = str(int(time.time() * 1000))

    for page in range(1, max_pages + 1):
        params = [
            ("searchType", f"title;accurate;{search_type}"),
            ("sortTr", "f_bbrq_s;desc"),
            ("gbrqStart", ""),
            ("gbrqEnd", ""),
            ("sxrqStart", ""),
            ("sxrqEnd", ""),
            ("sort", "true"),
            ("page", str(page)),
            ("size", str(page_size)),
            ("_", now_ms),
        ]
        try:
            list_resp = session.get("https://flk.npc.gov.cn/api/", params=params, timeout=timeout)
            if list_resp.status_code != 200:
                continue
            list_data = list_resp.json()
        except (requests.RequestException, ValueError):
            continue

        items = (((list_data or {}).get("result") or {}).get("data")) or []
        if not items:
            break

        for item in items:
            if len(docs) >= max_docs:
                return docs[:max_docs]
            law_id = item.get("id")
            if not law_id:
                continue
            try:
                detail_resp = session.post(
                    "https://flk.npc.gov.cn/api/detail",
                    data={"id": law_id},
                    timeout=timeout,
                )
                if detail_resp.status_code != 200:
                    continue
                detail_data = detail_resp.json()
            except (requests.RequestException, ValueError):
                continue

            result = (detail_data or {}).get("result") or {}
            title = result.get("title") or item.get("title") or ""
            publish = (item.get("publish") or result.get("publish") or "").split(" ")[0]
            level = result.get("level", "")
            body_items = result.get("body") if isinstance(result.get("body"), list) else []

            parsed_content = ""
            source_url = f"https://flk.npc.gov.cn/detail2.html?ZmY4MDgxODE3OTZhNjM2ZjAxNzk3MzQ0N2EwYjE0NDI%3D&id={law_id}"

            for body in body_items:
                file_type = str(body.get("type", "")).upper()
                if file_type != "HTML":
                    continue
                relative_url = body.get("url")
                if not relative_url:
                    continue
                try:
                    html_resp = session.get(f"https://wb.flk.npc.gov.cn{relative_url}", timeout=timeout)
                    if html_resp.status_code != 200:
                        continue
                    html_resp.encoding = "utf-8"
                except requests.RequestException:
                    continue
                parsed_content = _npc_parse_html(html_resp.text)
                source_url = f"https://wb.flk.npc.gov.cn{relative_url}"
                if parsed_content:
                    break

            if not parsed_content:
                fallback = result.get("content") or ""
                if isinstance(fallback, str):
                    parsed_content = _normalize_text(fallback)

            if len(parsed_content) < 120:
                continue

            unique_id = _md5("official_npc_api", law_id, title, publish, parsed_content[:1000])
            metadata = _build_metadata(
                unique_id=unique_id,
                source=source_url,
                page=len(docs) + 1,
                title=title,
                doc_type="official_npc_api",
                extra={
                    "law_id": law_id,
                    "level": level,
                    "publish": publish,
                    "source_repo": "https://github.com/LawRefBook/Laws",
                },
            )
            docs.append(Document(page_content=parsed_content, metadata=metadata))
            time.sleep(sleep_seconds)

    return docs[:max_docs]


def collect_legal_documents(
    source: str,
    official_mode: str = "npc_api",
    official_start_urls: Optional[list[str]] = None,
    npc_search_type: str = "1,9",
    npc_max_pages: int = 20,
    npc_page_size: int = 10,
    laws_dir: Optional[str] = None,
    laws_repo_url: Optional[str] = None,
    laws_repo_branch: str = "master",
    laws_repo_local_dir: str = ".cache/laws_repo",
    max_official_docs: int = 200,
    official_depth: int = 2,
    max_laws_docs: int = 5000,
) -> list[Document]:
    source = source.lower()
    official_mode = official_mode.lower()
    if source not in {"official", "laws", "both"}:
        raise ValueError("source 仅支持: official | laws | both")
    if official_mode not in {"npc_api", "crawler", "legal_rag_seed"}:
        raise ValueError("official_mode 仅支持: npc_api | crawler | legal_rag_seed")

    all_docs = []

    if source in {"official", "both"}:
        if official_mode == "npc_api":
            official_docs = fetch_npc_laws_via_api(
                max_docs=max_official_docs,
                max_pages=npc_max_pages,
                page_size=npc_page_size,
                search_type=npc_search_type,
            )
        elif official_mode == "legal_rag_seed":
            official_docs = collect_legal_rag_seed_docs(max_docs=max_official_docs)
        else:
            start_urls = official_start_urls or ["https://flk.npc.gov.cn/"]
            official_docs = crawl_official_legal_docs(
                start_urls=start_urls,
                max_docs=max_official_docs,
                max_depth=official_depth,
            )
        all_docs.extend(official_docs)

    if source in {"laws", "both"}:
        resolved_laws_dir = prepare_laws_repo(
            laws_dir=laws_dir,
            laws_repo_url=laws_repo_url,
            laws_repo_branch=laws_repo_branch,
            laws_repo_local_dir=laws_repo_local_dir,
        )
        laws_docs = load_open_source_laws(
            laws_dir=resolved_laws_dir,
            max_docs=max_laws_docs,
        )
        all_docs.extend(laws_docs)

    dedup_docs = []
    seen = set()
    for doc in all_docs:
        unique_id = doc.metadata.get("unique_id")
        if unique_id and unique_id not in seen:
            seen.add(unique_id)
            dedup_docs.append(doc)

    return dedup_docs
