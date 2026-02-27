# -*- coding: utf-8 -*-
# 法律文档专用切分器
# 支持三种文档类型：
#   1. 法律条文（民法典、刑法、行政法规、司法解释等）→ 章节为父，滑动窗口为子
#   2. 案例文档 → 整体保留，不切分
#   3. 无条文结构的纯文本 → 兜底整体保留

import re
import copy
import hashlib
from langchain_core.documents import Document

# ── 正则 ──────────────────────────────────────────────────────────────────────

# 条文号：第X条（汉字数字 + 阿拉伯数字均支持，必须带 MULTILINE 才能匹配非首行）
_ARTICLE_START_RE = re.compile(
    r'^(第[零一二三四五六七八九十百千万\d]+条)\s*', re.MULTILINE
)
# 章节标题：## 第X章 ...
_CHAPTER_RE = re.compile(r'^#{1,3}\s*(第.+?章[^\n]*?)$', re.MULTILINE)
# 条文内引用，如"第X条"
_REF_RE = re.compile(r'第[零一二三四五六七八九十百千万\d]+条')

# ── 切分参数 ──────────────────────────────────────────────────────────────────

_WINDOW = 4          # 滑动窗口：每个子chunk包含几条
_OVERLAP = 1         # 窗口重叠条数
_MIN_CHARS = 50      # 短于此字数的条文合并到相邻条文


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _md5(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def _make_prefix(law_name: str, category: str, chapter: str) -> str:
    """生成上下文前缀，如【民法典·侵权责任编·第三章 责任主体的特殊规定】"""
    parts = [p.strip() for p in [law_name, category, chapter] if p and p.strip()]
    if not parts:
        return ''
    return '【' + '·'.join(parts) + '】\n'


# ── 文档类型判断 ───────────────────────────────────────────────────────────────

def _is_case_doc(metadata: dict) -> bool:
    source = str(metadata.get('source', ''))
    category = str(metadata.get('category', ''))
    return '案例' in source or '案例' in category


def _has_articles(text: str) -> bool:
    return bool(_ARTICLE_START_RE.search(text, re.MULTILINE))


# ── 章节解析 ──────────────────────────────────────────────────────────────────

def _parse_chapters(text: str) -> list:
    """
    按章节标题切分文本。
    返回 [{'chapter': '第一章 一般规定', 'content': '...'}]
    无章节结构时返回单元素列表，chapter 为空字符串。
    """
    matches = list(_CHAPTER_RE.finditer(text))
    if not matches:
        return [{'chapter': '', 'content': text}]

    chapters = []
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            chapters.append({'chapter': title, 'content': content})
    return chapters


# ── 条文解析 ──────────────────────────────────────────────────────────────────

def _parse_articles(chapter_content: str) -> list:
    """
    从章节内容中提取所有条文。
    返回 [{'id': '第X条', 'text': '第X条 完整内容（含多款）'}]
    """
    lines = chapter_content.split('\n')
    articles = []
    current_id = None
    current_lines = []

    for line in lines:
        m = _ARTICLE_START_RE.match(line)
        if m:
            if current_id and current_lines:
                articles.append({
                    'id': current_id,
                    'text': '\n'.join(current_lines).strip()
                })
            current_id = m.group(1)
            current_lines = [line.strip()]
        elif current_id:
            stripped = line.strip()
            if stripped:
                current_lines.append(stripped)

    if current_id and current_lines:
        articles.append({
            'id': current_id,
            'text': '\n'.join(current_lines).strip()
        })

    return articles


# ── 全文条文索引（用于引用扩展）─────────────────────────────────────────────────

def _build_article_map(chapters: list) -> dict:
    """构建 {条文号: 条文全文} 的全文索引"""
    article_map = {}
    for ch in chapters:
        for art in _parse_articles(ch['content']):
            article_map[art['id']] = art['text']
    return article_map


# ── 引用扩展 ──────────────────────────────────────────────────────────────────

def _expand_references(chunk_text: str, article_map: dict) -> str:
    """
    检测 chunk 中的条文引用（如"适用本法第X条"），
    将被引用但不在 chunk 中的条文追加到末尾。
    """
    # 找出 chunk 正文中已有的条文号
    existing_ids = set(_ARTICLE_START_RE.findall(chunk_text))

    # 找出所有引用的条文号
    refs = _REF_RE.findall(chunk_text)

    expansions = []
    seen = set()
    for ref in refs:
        if ref in existing_ids or ref in seen:
            continue
        if ref in article_map:
            expansions.append(article_map[ref])
            seen.add(ref)

    if expansions:
        chunk_text += '\n\n【引用条文】\n' + '\n\n'.join(expansions)

    return chunk_text


# ── 滑动窗口切分（法律条文）──────────────────────────────────────────────────────

def _chapter_to_parent_children(
    articles: list,
    law_name: str,
    category: str,
    chapter: str,
    base_metadata: dict,
    article_map: dict,
) -> tuple:
    """
    对一章的条文做滑动窗口切分。
    返回 (parent_doc, [child_doc, ...])
      parent = 整章所有条文，存 MongoDB 供上下文召回
      child  = 每个滑动窗口，存向量库和 BM25 用于检索
    """
    prefix = _make_prefix(law_name, category, chapter)

    # ── 父 doc：整章 ──
    parent_text = prefix + '\n\n'.join(a['text'] for a in articles)
    parent_id = _md5(parent_text)
    parent_meta = copy.deepcopy(base_metadata)
    parent_meta.update({
        'unique_id': parent_id,
        'chunk_type': 'parent',
        'chapter': chapter,
        'law_name': law_name,
        'article_range': f"{articles[0]['id']}~{articles[-1]['id']}",
        'images_info': [],
    })
    parent_doc = Document(page_content=parent_text, metadata=parent_meta)

    # ── 子 doc：滑动窗口 ──
    child_docs = []
    i = 0
    while i < len(articles):
        group = articles[i: i + _WINDOW]

        # 合并过短的条文到前一条，避免碎片化
        merged_texts = []
        for art in group:
            if len(art['text']) < _MIN_CHARS and merged_texts:
                merged_texts[-1] += '\n' + art['text']
            else:
                merged_texts.append(art['text'])

        chunk_text = prefix + '\n\n'.join(merged_texts)

        # 扩展条文内引用
        chunk_text = _expand_references(chunk_text, article_map)

        child_id = _md5(chunk_text)
        child_meta = copy.deepcopy(base_metadata)
        child_meta.update({
            'unique_id': child_id,
            'parent_id': parent_id,
            'chunk_type': 'child',
            'chapter': chapter,
            'law_name': law_name,
            'article_ids': [a['id'] for a in group],
            'article_range': f"{group[0]['id']}~{group[-1]['id']}",
            'images_info': [],
        })
        child_docs.append(Document(page_content=chunk_text, metadata=child_meta))

        i += (_WINDOW - _OVERLAP)

    return parent_doc, child_docs


# ── 案例切分 ──────────────────────────────────────────────────────────────────

def _split_case_doc(doc: Document) -> list:
    """
    案例文件整体作为一个 chunk，不切分。
    案例结构完整（案情+分析+意义），拆开会丢失推理链。
    """
    meta = copy.deepcopy(doc.metadata)
    unique_id = _md5(doc.page_content)
    meta.update({
        'unique_id': unique_id,
        'chunk_type': 'case',
        'law_name': meta.get('title', ''),
        'images_info': [],
    })
    return [Document(page_content=doc.page_content, metadata=meta)]


# ── 主入口 ────────────────────────────────────────────────────────────────────

def legal_texts_split(raw_docs: list) -> tuple:
    """
    法律文档专用切分主函数。

    Args:
        raw_docs: load_open_source_laws() 返回的原始 Document 列表

    Returns:
        parent_docs: 章节级文档，需存入 MongoDB（供 merge_docs 父子召回）
        child_docs:  检索用文档，存入向量库和 BM25
    """
    all_parents = []
    all_children = []

    for doc in raw_docs:
        text = doc.page_content
        metadata = doc.metadata

        # ── 案例：整体保留 ──
        if _is_case_doc(metadata):
            all_children.extend(_split_case_doc(doc))
            continue

        # ── 无条文结构：兜底整体保留 ──
        if not _has_articles(text):
            meta = copy.deepcopy(metadata)
            meta.update({
                'unique_id': _md5(text),
                'chunk_type': 'plain',
                'images_info': [],
            })
            all_children.append(Document(page_content=text, metadata=meta))
            continue

        # ── 法律条文：章节父doc + 滑动窗口子doc ──
        law_name = metadata.get('title', '')
        category = metadata.get('category', '')

        chapters = _parse_chapters(text)
        article_map = _build_article_map(chapters)

        for ch in chapters:
            articles = _parse_articles(ch['content'])
            if not articles:
                continue

            parent_doc, child_docs = _chapter_to_parent_children(
                articles=articles,
                law_name=law_name,
                category=category,
                chapter=ch['chapter'],
                base_metadata=metadata,
                article_map=article_map,
            )
            all_parents.append(parent_doc)
            all_children.extend(child_docs)

    return all_parents, all_children
