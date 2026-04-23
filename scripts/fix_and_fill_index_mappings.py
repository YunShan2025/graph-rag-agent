from graphrag_agent.cache_manager.manager import CacheManager
from graphrag_agent.config.neo4jdb import get_db_manager


def fill_mappings(limit_chunks=None, limit_entities=None):
    cm = CacheManager()
    vm = cm.vector_matcher
    if vm is None:
        print('向量匹配器未启用')
        return

    print('清空内存索引并开始重建映射...')
    vm.clear()

    db = get_db_manager()
    added = 0

    # 遍历 chunks
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
                print(f'为 chunk {row["id"]} 添加向量失败: {e}')

    # 遍历 entities
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
                print(f'为 entity {row["id"]} 添加向量失败: {e}')

    # 强制保存索引与映射
    vm.save_index()

    print(f'重建映射完成，添加向量: {added}, 索引总数: {vm.index.ntotal}, 映射大小: {len(vm.index_to_key)}')


if __name__ == '__main__':
    fill_mappings()
