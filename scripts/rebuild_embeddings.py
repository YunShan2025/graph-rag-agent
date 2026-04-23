"""重建向量索引脚本

用法:
  python scripts/rebuild_embeddings.py

说明:
  - 脚本会遍历磁盘缓存（如果使用磁盘后端），尝试从每个缓存项中提取原始查询或文本内容，
    使用当前配置的 embedding 提供者生成向量并重建 FAISS 索引。
  - 在使用 GPU 加速前，请先在系统中安装兼容的 `torch`（CUDA 版本）和 `sentence-transformers`。
"""

from pathlib import Path
import sys
import traceback

from graphrag_agent.cache_manager.manager import CacheManager
from graphrag_agent.cache_manager.models.cache_item import CacheItem
from graphrag_agent.config import settings
from graphrag_agent.config.neo4jdb import get_db_manager


def extract_text_from_cache_item(item: CacheItem):
    meta = item.metadata if hasattr(item, 'metadata') else {}
    # 优先使用 metadata 中的 original_query
    orig = meta.get('original_query') if isinstance(meta, dict) else None
    if orig:
        return orig

    content = item.content if hasattr(item, 'content') else None
    # 如果内容是字符串，直接用作文本
    if isinstance(content, str):
        return content

    # 如果内容是字典，尝试常见字段
    if isinstance(content, dict):
        for k in ('text', 'content', 'query'):
            if k in content and isinstance(content[k], str):
                return content[k]

    return None


def main():
    print("初始化 CacheManager（会加载当前缓存与索引配置）...")
    cm = CacheManager()

    if not cm.enable_vector_similarity or not getattr(cm, 'vector_matcher', None):
        print("当前未启用向量相似性（CACHE_ENABLE_VECTOR_SIMILARITY=False），请在配置中启用后重试。")
        sys.exit(1)

    vm = cm.vector_matcher

    # 如果存量索引存在，先清空并重建
    print("清空内存索引并开始重建...")
    vm.clear()

    # 尝试从存储后端读取所有缓存键
    storage = cm.storage
    keys = []

    try:
        # DiskCacheBackend 存在 metadata 属性
        if hasattr(storage, 'metadata'):
            keys = list(storage.metadata.keys())
        elif hasattr(storage, 'backend') and hasattr(storage.backend, 'metadata'):
            keys = list(storage.backend.metadata.keys())
        else:
            print("无法从缓存后端枚举键：后端不支持 metadata 枚举。将尝试从 vector matcher 的 key_to_query 重建。")
            keys = list(vm.key_to_query.keys())
    except Exception:
        print("读取缓存键时出错：")
        traceback.print_exc()
        sys.exit(1)

    added = 0
    skipped = 0
    for k in keys:
        try:
            raw = storage.get(k)
            if raw is None:
                skipped += 1
                continue

            item = CacheItem.from_any(raw)
            text = extract_text_from_cache_item(item)
            if not text:
                skipped += 1
                continue

            # 使用原始上下文（如果有）
            context = item.metadata if isinstance(item.metadata, dict) else {}

            vm.add_vector(k, text, context)
            added += 1
        except Exception:
            skipped += 1
            print(f"处理键 {k} 时出错：")
            traceback.print_exc()

    # 保存索引
    vm.save_index()

    print(f"重建完成：添加向量 {added}，跳过 {skipped}。索引文件位于: {getattr(vm, 'index_file', None)}")


def rebuild_from_neo4j(vm, limit_chunks: int = None, limit_entities: int = None):
    """从 Neo4j 读取已存在的节点文本并用当前 embedding provider 重建索引"""
    db = get_db_manager()
    added = 0

    try:
        # 先处理 Chunk 节点
        chunk_q = """
        MATCH (c:`__Chunk__`)
        WHERE c.text IS NOT NULL
        RETURN c.id AS id, c.text AS text
        """
        if limit_chunks:
            chunk_q += f"\nLIMIT {int(limit_chunks)}"

        df_chunks = db.execute_query(chunk_q)
        if df_chunks is not None and not df_chunks.empty:
            for _, row in df_chunks.iterrows():
                key = f"neo4j_chunk:{row['id']}"
                text = row['text'] if row['text'] is not None else ''
                try:
                    vm.add_vector(key, text, {'source': 'neo4j', 'node_type': '__Chunk__'})
                    added += 1
                except Exception as e:
                    print(f"为 chunk {row['id']} 添加向量失败: {e}")

        # 再处理 Entity 节点（可选）
        entity_q = """
        MATCH (e:`__Entity__`)
        WHERE e.description IS NOT NULL OR e.id IS NOT NULL
        RETURN e.id AS id, CASE WHEN e.description IS NOT NULL THEN e.description ELSE e.id END AS text
        """
        if limit_entities:
            entity_q += f"\nLIMIT {int(limit_entities)}"

        df_entities = db.execute_query(entity_q)
        if df_entities is not None and not df_entities.empty:
            for _, row in df_entities.iterrows():
                key = f"neo4j_entity:{row['id']}"
                text = row['text'] if row['text'] is not None else ''
                try:
                    vm.add_vector(key, text, {'source': 'neo4j', 'node_type': '__Entity__'})
                    added += 1
                except Exception as e:
                    print(f"为 entity {row['id']} 添加向量失败: {e}")

    except Exception as e:
        print(f"从Neo4j重建索引时出错: {e}")
        return 0

    # 保存索引
    vm.save_index()
    print(f"从 Neo4j 重建完成：添加向量 {added}。索引文件位于: {getattr(vm, 'index_file', None)}")
    return added


if __name__ == '__main__':
    main()
    # 如果添加为0，尝试从 Neo4j 重建
    try:
        cm = CacheManager()
        vm = cm.vector_matcher
        if vm.index.ntotal == 0:
            print("当前索引为空，尝试从 Neo4j 重建向量索引...")
            rebuild_from_neo4j(vm)
    except Exception:
        pass


if __name__ == '__main__':
    main()
