import pickle
p = 'cache/vector_index.pkl'
with open(p,'rb') as f:
    data = pickle.load(f)
print(type(data))
for k in ['key_to_index','index_to_key','key_to_context','key_to_query','next_index']:
    print(k, type(data.get(k)), (list(data.get(k).items())[:5] if isinstance(data.get(k), dict) else data.get(k)))
