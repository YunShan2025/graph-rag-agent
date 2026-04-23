from graphrag_agent.cache_manager.manager import CacheManager
from graphrag_agent.config.neo4jdb import get_db_manager

queries = [
    "毕业生 奖学金 管理 办法",
    "退役 士兵 教育 资助 管理",
    "学生 违纪 处分 规定",
]

cm = CacheManager()
vm = cm.vector_matcher
if vm is None:
    print('未启用向量相似性 (vector matcher 为 None)')
    exit(1)

print('FAISS 索引向量总数:', vm.index.ntotal)
print('index_to_key 映射大小:', len(vm.index_to_key))
if len(vm.index_to_key) > 0:
    sample = list(vm.index_to_key.items())[:10]
    print('index_to_key sample:', sample)

db = get_db_manager()

for q in queries:
    print('\n查询:', q)
    # 指定默认 thread_id 以便与索引中存储的 context 匹配
    # 先打印原始向量相似度（不使用阈值过滤），便于调试
    try:
        emb = vm.embedding_provider.encode(q)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        k = min(10, vm.index.ntotal)
        D, I = vm.index.search(emb, k)
        print('  原始相似度（阈值前）:')
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            key = vm.index_to_key.get(idx, '<unknown>')
            print(f'    idx={idx} key={key} score={score:.4f}')
    except Exception as e:
        print('  无法计算原始相似度:', e)

    res = vm.find_similar(q, context_info={'thread_id': 'default'}, top_k=5)
    if not res:
        print('  未找到相似向量（在阈值/上下文过滤后）')
        continue
    for key, score in res:
        print(f'  匹配键: {key}  相似度: {score:.4f}')
        try:
            if key.startswith('neo4j_chunk:'):
                nid = key.split(':', 1)[1]
                df = db.execute_query("MATCH (c:`__Chunk__` {id: $id}) RETURN c.text AS text", {'id': nid})
                if df is not None and not df.empty:
                    txt = df.iloc[0]['text']
                    print('    chunk 文本片段:', (txt or '')[:200].replace('\n',' '))
            elif key.startswith('neo4j_entity:'):
                nid = key.split(':', 1)[1]
                df = db.execute_query("MATCH (e:`__Entity__` {id: $id}) RETURN e.description AS desc, e.id AS id", {'id': nid})
                if df is not None and not df.empty:
                    desc = df.iloc[0].get('desc')
                    print('    entity 描述片段:', (desc or '')[:200].replace('\n',' '))
            else:
                # 尝试从缓存存储获取原始查询文本
                qtxt = vm.key_to_query.get(key)
                if qtxt:
                    print('    存储的原始查询:', qtxt[:200].replace('\n',' '))
        except Exception as e:
            print('    检索元数据错误:', e)

    # 额外打印原始向量相似度分数（不使用阈值过滤），便于调试
    try:
        emb = vm.embedding_provider.encode(q)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        k = min(10, vm.index.ntotal)
        D, I = vm.index.search(emb, k)
        print('  原始相似度（阈值前）:')
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            key = vm.index_to_key.get(idx, '<unknown>')
            print(f'    idx={idx} key={key} score={score:.4f}')
    except Exception as e:
        print('  无法计算原始相似度:', e)

print('\n完成')
