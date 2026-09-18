'Module docstring.'

import numpy as np

# ============================================================

# ============================================================

SLICE_NUMBER = 64
NUM_COLS = 5
STATE_SIZE = SLICE_NUMBER * NUM_COLS  # 320


STATE_GROUPS = ['init', 'ps0', 'pl0', 'ps1']
GROUP_SIZE = STATE_SIZE


# ============================================================

# ============================================================

def _flat_to_2d(flat_vals, start, size=STATE_SIZE):
    'module object.'
    arr = np.asarray(flat_vals[start:start + size], dtype=np.int32)
    return arr.reshape(SLICE_NUMBER, NUM_COLS)


def parse_all_states(r_flat, b_flat):
    'module object.'
    states = {}
    for i, name in enumerate(STATE_GROUPS):
        start = i * STATE_SIZE
        states[name] = {
            'r': _flat_to_2d(r_flat, start),
            'b': _flat_to_2d(b_flat, start),
        }
    return states


# ============================================================

# ============================================================

def _gini(arr):
    'module object.'
    arr = np.asarray(arr, dtype=np.float64)
    if arr.sum() == 0 or len(arr) <= 1:
        return 0.0
    sorted_arr = np.sort(arr)
    n = len(arr)
    cumsum = np.cumsum(sorted_arr)
    g = (2 * np.sum(np.arange(1, n + 1) * sorted_arr)) / (n * cumsum[-1]) - (n + 1) / n
    return float(max(0.0, min(1.0, g)))


def extract_single_group_features(state_2d, prefix, channel='b'):
    'module object.'
    f = {}
    s = np.asarray(state_2d, dtype=np.int32)
    assert s.shape == (SLICE_NUMBER, 5), f"expected ({SLICE_NUMBER},5), got {s.shape}"


    total = int(np.sum(s))
    f[f'{prefix}_total'] = total
    f[f'{prefix}_density'] = float(total) / float(STATE_SIZE)


    for x in range(5):
        col = s[:, x]
        f[f'{prefix}_x{x}_count'] = int(np.sum(col))
        f[f'{prefix}_x{x}_active'] = int(np.sum(col > 0))


    z_counts = s.sum(axis=1)  # (SLICE_NUMBER,)
    f[f'{prefix}_active_z'] = int(np.sum(z_counts > 0))
    f[f'{prefix}_z_gini'] = _gini(z_counts)


    z_adj = 0
    for x in range(5):
        col = s[:, x]
        z_adj += int(np.sum(col[:-1] & col[1:]))
    f[f'{prefix}_z_adj_pairs'] = z_adj


    f[f'{prefix}_z_max'] = int(np.max(z_counts))
    f[f'{prefix}_z_min'] = int(np.min(z_counts))
    f[f'{prefix}_z_std'] = float(np.std(z_counts.astype(np.float64)))


    x_active_per_z = (s > 0).sum(axis=1)
    f[f'{prefix}_x_active_per_z_mean'] = float(np.mean(x_active_per_z.astype(np.float64)))


    x_adj = 0
    for z in range(SLICE_NUMBER):
        row = s[z, :]
        x_adj += int(np.sum(row[:-1] & row[1:]))
    f[f'{prefix}_x_adj_pairs'] = x_adj


    for z in range(SLICE_NUMBER):
        f[f'{prefix}_z{z}'] = int(z_counts[z])

    return f


# ============================================================

# ============================================================

def extract_transition_features(states_dict, channel='b'):
    'module object.'
    f = {}
    ch = channel

    group_names = list(STATE_GROUPS)

    for i in range(len(group_names) - 1):
        g0 = group_names[i]
        g1 = group_names[i + 1]
        s0 = states_dict[g0][ch]
        s1 = states_dict[g1][ch]

        t0 = int(np.sum(s0))
        t1 = int(np.sum(s1))


        f[f'{g0}_to_{g1}_{ch}_delta'] = t1 - t0


        if t0 > 0:
            f[f'{g0}_to_{g1}_{ch}_growth_ratio'] = float(t1) / float(t0)
        else:
            f[f'{g0}_to_{g1}_{ch}_growth_ratio'] = 1.0 if t1 > 0 else 0.0


        overlap = int(np.sum((s0 > 0) & (s1 > 0)))
        f[f'{g0}_to_{g1}_{ch}_overlap'] = overlap
        if t0 > 0:
            f[f'{g0}_to_{g1}_{ch}_overlap_ratio'] = float(overlap) / float(t0)
        else:
            f[f'{g0}_to_{g1}_{ch}_overlap_ratio'] = 0.0


        new_positions = int(np.sum((s0 == 0) & (s1 > 0)))
        f[f'{g0}_to_{g1}_{ch}_new'] = new_positions


        vanished = int(np.sum((s0 > 0) & (s1 == 0)))
        f[f'{g0}_to_{g1}_{ch}_vanished'] = vanished

    return f


# ============================================================

# ============================================================

def extract_rb_interaction_features(states_dict):
    'module object.'
    f = {}
    for name in STATE_GROUPS:
        r = states_dict[name]['r']
        b = states_dict[name]['b']


        both = int(np.sum((r > 0) & (b > 0)))
        f[f'{name}_rb_both'] = both


        r_total = int(np.sum(r))
        b_total = int(np.sum(b))
        f[f'{name}_rb_ratio'] = float(r_total) / (float(b_total) + 1e-6)


        f[f'{name}_rb_density'] = float(r_total + b_total) / float(STATE_SIZE)


        r_only = int(np.sum((r > 0) & (b == 0)))
        f[f'{name}_r_only'] = r_only


        b_only = int(np.sum((r == 0) & (b > 0)))
        f[f'{name}_b_only'] = b_only

    return f


# ============================================================

# ============================================================

def extract_all_features(r_flat, b_flat):
    'module object.'
    states = parse_all_states(r_flat, b_flat)
    features = {}


    for name in STATE_GROUPS:
        features.update(
            extract_single_group_features(states[name]['b'], f'{name}_b', 'b'))
        features.update(
            extract_single_group_features(states[name]['r'], f'{name}_r', 'r'))


    features.update(extract_transition_features(states, 'b'))


    features.update(extract_transition_features(states, 'r'))


    features.update(extract_rb_interaction_features(states))

    return features


def get_feature_vector(features_dict, exclude_z_details=False):
    'module object.'
    vec = []
    for k in sorted(features_dict.keys()):
        v = features_dict[k]

        if exclude_z_details and ('_z' in k and k.split('_z')[-1].isdigit()):
            continue
        if isinstance(v, np.ndarray) and v.size == 1:
            vec.append(float(v))
        elif np.isscalar(v):
            vec.append(float(v))
        elif isinstance(v, np.ndarray):
            vec.extend(v.flatten().tolist())
        else:
            vec.append(float(v))
    return np.array(vec, dtype=np.float32)


# ============================================================

# ============================================================



def extract_milp_linear_features(r_flat, b_flat):
    'module object.'
    states = parse_all_states(r_flat, b_flat)
    f = {}

    def _add_counts(s, prefix):
        'module object.'
        f[f'{prefix}_total'] = int(np.sum(s))

        z_counts = s.sum(axis=1)
        z_nonzero = int(np.sum(z_counts > 0))
        f[f'{prefix}_active_z_count'] = z_nonzero
        f[f'{prefix}_z_max'] = int(np.max(z_counts))
        f[f'{prefix}_z_min'] = int(np.min(z_counts))
        f[f'{prefix}_z_spread'] = int(np.max(z_counts)) - int(np.min(z_counts))
        for z in range(SLICE_NUMBER):
            f[f'{prefix}_z{z}'] = int(z_counts[z])

    def _add_z_adj(s, prefix):
        'module object.'
        adj = 0
        for x in range(5):
            col = s[:, x]
            adj += int(np.sum(col[:-1] & col[1:]))
        f[f'{prefix}_z_adj_pairs'] = adj

    def _add_x_adj(s, prefix):
        'module object.'
        adj = 0
        for z in range(SLICE_NUMBER):
            row = s[z, :]
            adj += int(np.sum(row[:-1] & row[1:]))
        f[f'{prefix}_x_adj_pairs'] = adj

    def _add_overlap(sa, sb, prefix):
        'module object.'
        ov = int(np.sum((sa > 0) & (sb > 0)))
        f[f'{prefix}_overlap'] = ov

    def _add_rb_both(r, b, prefix):
        'module object.'
        both = int(np.sum((r > 0) & (b > 0)))
        f[f'{prefix}_rb_both'] = both


    for name in STATE_GROUPS:
        for ch, ch_name in [('b', 'b'), ('r', 'r')]:
            s = states[name][ch]
            prefix = f'{name}_{ch_name}'
            _add_counts(s, prefix)
            _add_z_adj(s, prefix)
            _add_x_adj(s, prefix)


    for name in STATE_GROUPS:
        _add_rb_both(states[name]['r'], states[name]['b'], name)


    state_order = ['init', 'ps0', 'pl0', 'ps1']
    for i in range(len(state_order) - 1):
        a_name, b_name = state_order[i], state_order[i + 1]
        for ch in ['b', 'r']:
            sa = states[a_name][ch]
            sb = states[b_name][ch]
            prefix = f'{a_name}_to_{b_name}_{ch}'
            f[f'{prefix}_delta'] = int(np.sum(sb)) - int(np.sum(sa))
            _add_overlap(sa, sb, prefix)


    init_r_total = f.get('init_r_total', 0)
    init_b_total = f.get('init_b_total', 0)
    f['init_rb_diff'] = init_r_total - 2 * init_b_total  # k=2

    return f


def extract_all_features_milp(r_flat, b_flat):
    'module object.'
    return extract_milp_linear_features(r_flat, b_flat)


# ============================================================

# ============================================================

if __name__ == '__main__':
    print("=== Ascon feature extractor v2 self-test ===")


    r_flat = np.random.randint(0, 2, size=4 * STATE_SIZE).astype(np.int32)
    b_flat = np.random.randint(0, 2, size=4 * STATE_SIZE).astype(np.int32)


    init_b = np.zeros((SLICE_NUMBER, NUM_COLS), dtype=np.int32)
    init_r = np.zeros((SLICE_NUMBER, NUM_COLS), dtype=np.int32)
    init_b[0, 0] = 1
    init_b[5, 0] = 1
    init_b[10, 0] = 1
    init_r[2, 0] = 1
    init_r[7, 0] = 1
    b_flat[0:STATE_SIZE] = init_b.flatten()
    r_flat[0:STATE_SIZE] = init_r.flatten()


    features = extract_all_features(r_flat, b_flat)
    print(f"total features: {len(features)}")


    n_single = sum(1 for k in features if any(k.startswith(f'{g}_') and not ('_to_' in k) for g in STATE_GROUPS))
    n_transition = sum(1 for k in features if '_to_' in k)
    n_rb = sum(1 for k in features if 'rb_' in k)

    print(f"  per-group statistical features: ~{n_single}")
    print(f"  inter-group transition features: {n_transition}")
    print(f"  r/b interaction features: {n_rb}")

    vec = get_feature_vector(features, exclude_z_details=True)
    print(f"\nscalar feature vector dim (excluding per-z details): {len(vec)}")

    vec_full = get_feature_vector(features, exclude_z_details=False)
    print(f"scalar feature vector dim (including per-z details): {len(vec_full)}")


    print("\n--- sample features ---")
    for k in sorted(features.keys())[:20]:
        v = features[k]
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}")
        else:
            print(f"  {k}: {v}")
