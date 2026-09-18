"""Verify the rule tables embedded in the better_attack/attack/*_distill.py scripts against
the authoritative JSON tables in distill/rules/ (extracted by the distill scripts).

Usage: python verify_embedded_rules.py
"""
import ast
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULES = os.path.join(ROOT, 'distill', 'rules')

# Script path -> JSON table + expected embedding transform
CASES = [
    # (script, json, kind)
    (r'better_attack\attack\Keccak\SHA3_512\4-round\stage1_blue_search_distill.py',
     'rules_sha3_512_stage1.json', 'sha3_s1'),
    (r'better_attack\attack\Keccak\SHA3_512\4-round\stage2_red_search_distill.py',
     'rules_sha3_512_stage2.json', 'sha3_s2'),
    (r'better_attack\attack\Keccak\SHA3_384\4-round\stage1_blue_search_distill.py',
     'rules_sha3_384_stage1.json', 'sha3_s1'),
    (r'better_attack\attack\Keccak\SHA3_384\4-round\stage2_red_search_distill.py',
     'rules_sha3_384_stage2.json', 'sha3_s2'),
    (r'better_attack\attack\Ascon\AsconXOF\Ascon_XOF_3_preimage_distill.py',
     'rules_ascon.json', 'ascon'),
]

def extract_const(path, name):
    """Extract the value of a module-level list/dict constant via AST."""
    with open(os.path.join(ROOT, path), encoding='utf-8') as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
    raise KeyError(f'{name} not found in {path}')


def rules_from_json(path):
    with open(os.path.join(RULES, path), encoding='utf-8') as f:
        return json.load(f)['rules']


def norm_embedded(rows):
    """Rows as embedded in scripts: (name, sign, w_norm, max_val) already
    normalized so that sum(|w|) == 1 (the scripts renormalize at import)."""
    s = sum(abs(r[2]) for r in rows)
    return [(r[0], r[1], r[2] / s, r[3]) for r in rows]


def check(name, embedded, json_rules, kind):
    """Compare embedded (already normalized) with the JSON table, applying the
    documented embedding transforms."""
    problems = []
    if kind == 'sha3_s1':
        # 512: exclude blue_z_active + pi1_blue_active_z; 384: exclude blue_z_active
        json_rules = [r for r in json_rules if r['name'] not in ('blue_z_active', 'pi1_blue_active_z')]
        json_rules = [r for r in json_rules if r['name'] not in ('blue_z_active',)]

    jmap = {r['name']: r for r in json_rules}
    emap = {r[0]: r for r in embedded}

    # The embedded tables renormalize the embedded subset to sum|w| = 1
    # (documented in the attack-script docstrings and the README embedding notes),
    # so we compare RELATIVE weights (ratios within each table).
    emb_sum = sum(abs(e[2]) for e in embedded)
    jsum = sum(abs(r['w_norm']) for r in json_rules)

    for r in json_rules:
        e = emap.get(r['name'])
        if e is None:
            problems.append(f"  MISSING in script: {r['name']}")
            continue
        if e[1] != r['sign']:
            problems.append(f"  SIGN mismatch {r['name']}: script={e[1]} json={r['sign']}")
        # relative weight: w_i / sum|w| must match
        e_rel = e[2] / emb_sum
        j_rel = r['w_norm'] / jsum
        if abs(e_rel - j_rel) > 2e-5:
            problems.append(f"  REL-WEIGHT mismatch {r['name']}: script={e_rel:.6f} json={j_rel:.6f}")
        if abs(e[3] - r['max_val']) > 1e-6:
            problems.append(f"  MAXVAL mismatch {r['name']}: script={e[3]} json={r['max_val']}")
    for r in embedded:
        if r[0] not in jmap:
            problems.append(f"  EXTRA in script: {r[0]}")

    if problems:
        print(f"FAIL: {name}")
        for p in problems:
            print(p)
        return False
    print(f"OK:   {name}  ({len(embedded)} rules match JSON {kind}, relative weights)")
    return True


all_ok = True
for script, jfile, kind in CASES:
    if kind == 'ascon':
        emb = extract_const(script, 'DISTILL_WEIGHTS')
        jrules = rules_from_json(jfile)
        problems = []
        jmap = {r['name']: r for r in jrules}
        for k, v in emb.items():
            j = jmap.get(k)
            if j is None:
                problems.append(f"  EXTRA key: {k}")
            elif abs(v - j['w_norm']) > 2e-6:
                problems.append(f"  WEIGHT mismatch {k}: script={v:.6f} json={j['w_norm']:.6f}")
        for r in jrules:
            if r['name'] not in emb:
                problems.append(f"  MISSING key: {r['name']}")
        if problems:
            print(f"FAIL: {script}")
            for p in problems:
                print(p)
            all_ok = False
        else:
            print(f"OK:   {script}  ({len(emb)} weights match JSON ascon)")
        continue

    emb = norm_embedded(extract_const(script, 'RULES_DISTILL'))
    if not check(script, emb, rules_from_json(jfile), kind):
        all_ok = False

print()
print('ALL OK' if all_ok else 'PROBLEMS FOUND')
