"""Question groups (label-free): whitespace-normalised question text. H2V from run_identity.json (row_ids, questions_steps),
DeltaBench from inputs/deltabench/answers.json. Writes GROUPS_<bench>.json {uid: group_id}."""
import json, sys, hashlib
bench, src, out = sys.argv[1:4]
R = json.load(open(src, encoding='utf8'))
if bench == 'hard2verify':
    pairs = [(u, qs[0]) for u, qs in zip(R['row_ids'], R['questions_steps'])]
else:
    pairs = [(r['uid'], r['question']) for r in R]
norm = lambda t: ' '.join(t.split())
G = {u: 'q' + hashlib.sha1(norm(q).encode()).hexdigest()[:12] for u, q in pairs}
json.dump(G, open(out, 'w'), indent=0)
print(bench, 'answers', len(G), 'groups', len(set(G.values())))
