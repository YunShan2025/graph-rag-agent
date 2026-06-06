"""
检索结果适配器

将不同搜索工具的原始输出统一转换为RetrievalResult数据模型，便于多Agent管线消费。
"""
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence
import json
import uuid

from graphrag_agent.agents.multi_agent.core.retrieval_result import (
    RetrievalMetadata,
    RetrievalResult,
)


def create_retrieval_metadata(
    *,
    source_id: str,
    source_type: str,
    confidence: float = 0.5,
    timestamp: Optional[datetime] = None,
    do_level: Optional[str] = None,
    community_id: Optional[str] = None,
    hop_count: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> RetrievalMetadata:
    """构建统一的RetrievalMetadata对象。"""
    return RetrievalMetadata(
        source_id=source_id,
        source_type=source_type,  # type: ignore[arg-type]
        confidence=confidence,
        timestamp=timestamp or datetime.now(),
        do_level=do_level,  # type: ignore[arg-type]
        community_id=community_id,
        hop_count=hop_count,
        extra=extra or {},
    )


def create_retrieval_result(
    *,
    evidence: Any,
    source: str,
    granularity: str,
    metadata: RetrievalMetadata,
    score: float = 0.5,
    result_id: Optional[str] = None,
) -> RetrievalResult:
    """构建RetrievalResult对象。"""
    return RetrievalResult(
        result_id=result_id or str(uuid.uuid4()),
        granularity=granularity,  # type: ignore[arg-type]
        evidence=evidence,
        metadata=metadata,
        source=source,  # type: ignore[arg-type]
        score=score,
    )


def results_to_payload(results: Sequence[RetrievalResult]) -> List[Dict[str, Any]]:
    """将RetrievalResult列表转换为可序列化的字典列表。"""
    return [result.to_dict() for result in results]


def _try_parse_aggregated_context(page_content: str) -> Optional[Dict[str, List[str]]]:
    """
    尝试解析 page_content 中的聚合格式。

    LocalSearch 的 retrieval_query 返回的 page_content 格式为：
    Entities:
    - entity1
    - entity2
    Reports:
    - report1
    Chunks:
    - chunk1
    Relationships:
    - rel1

    返回解析后的字典，如果不是聚合格式则返回 None。
    """
    if not page_content or not isinstance(page_content, str):
        return None

    # 检测是否包含聚合格式的特征
    sections = ["Entities:", "Reports:", "Chunks:", "Relationships:"]
    found_sections = [s for s in sections if s in page_content]
    if len(found_sections) < 2:
        return None

    # 解析各部分
    result = {}
    for i, section in enumerate(sections):
        start = page_content.find(section)
        if start < 0:
            result[section.rstrip(":")] = []
            continue

        # 找到下一个部分的开始
        end = len(page_content)
        for next_section in sections[i + 1:]:
            next_pos = page_content.find(next_section, start + len(section))
            if next_pos >= 0:
                end = next_pos
                break

        # 提取该部分的内容
        section_content = page_content[start + len(section):end].strip()

        # 解析列表项（以 "- " 开头的行）
        items = []
        for line in section_content.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                item = line[2:].strip()
                if item and item != "None":
                    items.append(item)

        result[section.rstrip(":")] = items

    return result


def _results_from_aggregated(
    parsed: Dict[str, Any],
    *,
    source: str,
    default_confidence: float = 0.6,
) -> List[RetrievalResult]:
    """
    从聚合字典创建多个 RetrievalResult。

    将 Chunks、Reports、Relationships、Entities 分别创建独立的 RetrievalResult。
    """
    results: List[RetrievalResult] = []

    # Chunks - 文档片段
    chunks = parsed.get("Chunks") or []
    for i, chunk_text in enumerate(chunks):
        if not chunk_text:
            continue
        metadata = create_retrieval_metadata(
            source_id=f"{source}_chunk_{i}",
            source_type="chunk",
            confidence=default_confidence,
        )
        results.append(
            create_retrieval_result(
                evidence=str(chunk_text),
                source=source,
                granularity="Chunk",
                metadata=metadata,
                score=default_confidence,
            )
        )

    # Reports - 社区摘要
    reports = parsed.get("Reports") or []
    for i, report_text in enumerate(reports):
        if not report_text:
            continue
        metadata = create_retrieval_metadata(
            source_id=f"{source}_report_{i}",
            source_type="community",
            confidence=default_confidence,
        )
        results.append(
            create_retrieval_result(
                evidence=str(report_text),
                source=source,
                granularity="Community",
                metadata=metadata,
                score=default_confidence,
            )
        )

    # Relationships - 关系描述
    relationships = parsed.get("Relationships") or []
    for i, rel_text in enumerate(relationships):
        if not rel_text:
            continue
        metadata = create_retrieval_metadata(
            source_id=f"{source}_rel_{i}",
            source_type="relationship",
            confidence=default_confidence,
        )
        results.append(
            create_retrieval_result(
                evidence=str(rel_text),
                source=source,
                granularity="AtomicKnowledge",
                metadata=metadata,
                score=default_confidence,
            )
        )

    # Entities - 实体描述
    entities = parsed.get("Entities") or []
    for i, entity_text in enumerate(entities):
        if not entity_text:
            continue
        metadata = create_retrieval_metadata(
            source_id=f"{source}_entity_{i}",
            source_type="entity",
            confidence=default_confidence,
        )
        results.append(
            create_retrieval_result(
                evidence=str(entity_text),
                source=source,
                granularity="AtomicKnowledge",
                metadata=metadata,
                score=default_confidence,
            )
        )

    return results


def _extract_page_content(doc: Any) -> str:
    """从文档对象提取 page_content。"""
    if isinstance(doc, dict):
        return doc.get("page_content") or doc.get("text") or doc.get("metadata", {}).get("text") or ""
    return getattr(doc, "page_content", None) or getattr(doc, "metadata", {}).get("text") or ""


def _extract_metadata_and_score(doc: Any, default_confidence: float):
    """从文档对象提取 metadata 和 score。"""
    if isinstance(doc, dict):
        metadata_dict = doc.get("metadata", {}) or {}
        score_value = doc.get("score", metadata_dict.get("score", default_confidence))
    else:
        metadata_dict = getattr(doc, "metadata", {}) or {}
        score_value = getattr(doc, "score", metadata_dict.get("score", default_confidence))
    return metadata_dict, float(score_value or default_confidence)


def results_from_documents(
    docs: Iterable[Any],
    *,
    source: str,
    default_confidence: float = 0.6,
    granularity: str = "Chunk",
) -> List[RetrievalResult]:
    """
    从LangChain Document或类似对象生成RetrievalResult列表。

    支持两种格式：
    1. 普通文档：每个文档生成一个 RetrievalResult
    2. 聚合字典格式（LocalSearch 使用）：解析为多个细粒度 RetrievalResult

    参数:
        docs: LangChain Document或兼容对象迭代器
        source: 检索来源（local_search / global_search 等）
        default_confidence: 默认置信度
        granularity: 结果粒度
    """
    results: List[RetrievalResult] = []
    for doc in docs:
        page_content = _extract_page_content(doc)

        # 检测是否为聚合字典格式
        parsed = _try_parse_aggregated_context(page_content)
        if parsed is not None:
            results.extend(
                _results_from_aggregated(parsed, source=source, default_confidence=default_confidence)
            )
        else:
            # 原有逻辑：单个文档
            metadata_dict, score = _extract_metadata_and_score(doc, default_confidence)
            source_id = str(
                metadata_dict.get("id")
                or metadata_dict.get("source_id")
                or metadata_dict.get("chunk_id")
                or uuid.uuid4()
            )
            community_id = metadata_dict.get("community_id") or metadata_dict.get("community")
            metadata = create_retrieval_metadata(
                source_id=source_id,
                source_type="chunk",
                confidence=metadata_dict.get("confidence", score),
                community_id=community_id,
                extra={
                    "source": metadata_dict.get("source"),
                    "document_id": metadata_dict.get("document_id"),
                    "raw_metadata": metadata_dict,
                },
            )
            results.append(
                create_retrieval_result(
                    evidence=page_content,
                    source=source,
                    granularity=granularity,
                    metadata=metadata,
                    score=score,
                )
            )
    return results


def results_from_entities(
    entities: Iterable[Dict[str, Any]],
    *,
    source: str,
    confidence: float = 0.55,
) -> List[RetrievalResult]:
    """从实体列表构造RetrievalResult。"""
    results: List[RetrievalResult] = []
    for entity in entities:
        entity_id = str(entity.get("id") or uuid.uuid4())
        description = entity.get("description") or entity.get("text") or ""
        metadata = create_retrieval_metadata(
            source_id=entity_id,
            source_type="entity",
            confidence=entity.get("confidence", confidence),
            extra={"raw_entity": entity},
        )
        results.append(
            create_retrieval_result(
                evidence=description,
                source=source,
                granularity="AtomicKnowledge",
                metadata=metadata,
                score=entity.get("weight", confidence),
            )
        )
    return results


def results_from_relationships(
    relationships: Iterable[Dict[str, Any]],
    *,
    source: str,
    confidence: float = 0.5,
) -> List[RetrievalResult]:
    """从关系数据生成RetrievalResult。"""
    results: List[RetrievalResult] = []
    for relation in relationships:
        relation_id = str(
            relation.get("id")
            or f"{relation.get('start')}->{relation.get('end')}:{relation.get('type')}"
            or uuid.uuid4()
        )
        description = relation.get("description") or ""
        metadata = create_retrieval_metadata(
            source_id=relation_id,
            source_type="relationship",
            confidence=relation.get("confidence", confidence),
            extra={"raw_relationship": relation},
        )
        evidence = description or f"{relation.get('start')} -{relation.get('type')}-> {relation.get('end')}"
        # 将 weight 归一化到 [0, 1] 范围，避免超出 score 的验证范围
        raw_weight = relation.get("weight", confidence)
        normalized_score = min(1.0, max(0.0, raw_weight / 10.0 if raw_weight > 1.0 else raw_weight))
        results.append(
            create_retrieval_result(
                evidence=evidence,
                source=source,
                granularity="AtomicKnowledge",
                metadata=metadata,
                score=normalized_score,
            )
        )
    return results


def merge_retrieval_results(*result_groups: Iterable[RetrievalResult]) -> List[RetrievalResult]:
    """合并多个RetrievalResult序列并去重（基于source_id与granularity）。"""
    merged: Dict[tuple[str, str], RetrievalResult] = {}
    for group in result_groups:
        for result in group:
            key = (result.metadata.source_id, result.granularity)
            existing = merged.get(key)
            if existing is None or result.score > existing.score:
                merged[key] = result
    return list(merged.values())
