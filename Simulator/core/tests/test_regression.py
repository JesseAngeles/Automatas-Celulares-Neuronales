# core/tests/test_regression.py
#
# Suite de regresión del core CUDA (CA + NCA) contra referencias numpy:
#   - 48 referencias bit-exactas de CA: 6 configs × 4 fronteras × {step, run}
#   - 2 referencias de NCA: binaria (sigmoid) y multiclase (argmax)
#   - run(37) ≡ 37 × step() en grilla rectangular
#   - Modo estocástico: p=1.0 bit-exacto, misma semilla → misma salida,
#     semillas distintas → salidas distintas, fracción actualizada ≈ p
#
# Uso:  cd core && bind/bin/python tests/test_regression.py
#
# NOTA: c_neighbourhood/c_group_sizes son símbolos __constant__ compartidos
# por todas las instancias de una clase; no crear dos CAs con configs
# distintas al mismo tiempo (cada config se prueba secuencialmente).

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))
import CA as ca_module  # noqa: E402

CHECKS = {"pass": 0, "fail": 0}


def check(name: str, ok: bool):
    CHECKS["pass" if ok else "fail"] += 1
    if not ok:
        print(f"  FAIL  {name}")


# ── Referencia numpy del CA semitotalístico ───────────────────────────────────

def ref_multipliers(states, group_sizes):
    n_groups = len(group_sizes)
    mult = [0] * n_groups
    mult[n_groups - 1] = 1
    for g in range(n_groups - 2, -1, -1):
        mult[g] = mult[g + 1] * ((states - 1) * group_sizes[g + 1] + 1)
    return mult


def ref_step(grid, states, groups, rule, boundary):
    H, W = grid.shape
    group_sizes = [len(g) for g in groups]
    mult = ref_multipliers(states, group_sizes)
    out = np.empty_like(grid)
    for y in range(H):
        for x in range(W):
            rule_index = 0
            for g, group in enumerate(groups):
                s = 0
                for dx, dy in group:
                    nx, ny = x + dx, y + dy
                    if boundary == 1:      # PERIODIC
                        s += grid[ny % H, nx % W]
                    elif boundary == 2:    # REFLECTIVE: fuera → celda origen
                        if nx < 0 or nx >= W or ny < 0 or ny >= H:
                            s += grid[y, x]
                        else:
                            s += grid[ny, nx]
                    elif boundary == 3:    # REFLEXIVE: clamp
                        s += grid[min(max(ny, 0), H - 1), min(max(nx, 0), W - 1)]
                    else:                  # OPEN: fuera → 0
                        if 0 <= nx < W and 0 <= ny < H:
                            s += grid[ny, nx]
                rule_index += s * mult[g]
            out[y, x] = rule[rule_index]
    return out


def make_ca(grid, states, groups, rule):
    flat = [pair for g in groups for pair in g]
    group_sizes = [len(g) for g in groups]
    H, W = grid.shape
    return ca_module.CA(
        W, H, grid, states,
        np.array(flat, dtype=np.int32), len(flat),
        np.array(group_sizes, dtype=np.int32), len(group_sizes),
        np.array(rule, dtype=np.int32), len(rule),
    )


def rule_size(states, group_sizes):
    r = 1
    for gs in group_sizes:
        r *= (states - 1) * gs + 1
    return r


# ── Configs de CA (6) ─────────────────────────────────────────────────────────

MOORE_RING = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]

def ca_configs(rng):
    configs = []

    # 1. Life: 2 grupos (anillo Moore + centro), 2 estados
    configs.append(dict(
        name="life", states=2, shape=(24, 32),
        groups=[MOORE_RING, [[0, 0]]],
        rule=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ))

    # 2. 3 estados, un solo grupo 3×3 (incluye centro)
    groups = [[[dx, dy] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]]
    r = rule_size(3, [9])
    configs.append(dict(
        name="3states-1group", states=3, shape=(20, 20),
        groups=groups, rule=rng.integers(0, 3, size=r).tolist(),
    ))

    # 3. 2 estados, 3 grupos (anillo partido + centro)
    groups = [MOORE_RING[:4], MOORE_RING[4:], [[0, 0]]]
    r = rule_size(2, [4, 4, 1])
    configs.append(dict(
        name="2states-3groups", states=2, shape=(16, 40),
        groups=groups, rule=rng.integers(0, 2, size=r).tolist(),
    ))

    # 4. 4 estados, cruz de radio 2 en 2 grupos
    cross = [[0, -2], [0, -1], [0, 1], [0, 2], [-2, 0], [-1, 0], [1, 0], [2, 0]]
    groups = [cross, [[0, 0]]]
    r = rule_size(4, [8, 1])
    configs.append(dict(
        name="4states-cross-r2", states=4, shape=(22, 18),
        groups=groups, rule=rng.integers(0, 4, size=r).tolist(),
    ))

    # 5. Vecindad dispersa de radio 60: fuerza el kernel global (tile > 16 KB)
    far = [[[0, 0], [60, 0], [-60, 0], [0, 60], [0, -60]]]
    r = rule_size(2, [5])
    configs.append(dict(
        name="sparse-r60-untiled", states=2, shape=(70, 130),
        groups=far, rule=rng.integers(0, 2, size=r).tolist(),
    ))

    # 6. 5 estados, Moore radio 1 en 2 grupos, regla aleatoria
    groups = [MOORE_RING, [[0, 0]]]
    r = rule_size(5, [8, 1])
    configs.append(dict(
        name="5states-moore", states=5, shape=(19, 27),
        groups=groups, rule=rng.integers(0, 5, size=r).tolist(),
    ))

    return configs


def test_ca_bit_exact(rng, steps=5):
    print("── CA: 6 configs × 4 fronteras × {step, run} ──")
    for cfg in ca_configs(rng):
        H, W = cfg["shape"]
        init = rng.integers(0, cfg["states"], size=(H, W)).astype(np.uint8)
        for boundary in range(4):
            # Referencia numpy
            ref_frames = []
            g = init.copy()
            for _ in range(steps):
                g = ref_step(g, cfg["states"], cfg["groups"], cfg["rule"], boundary)
                ref_frames.append(g.copy())

            # step()
            ca = make_ca(init, cfg["states"], cfg["groups"], cfg["rule"])
            ca.set_boundary(boundary)
            ok = True
            for s in range(steps):
                ca.step()
                ok = ok and np.array_equal(ca.get_space(), ref_frames[s])
            check(f"{cfg['name']} b{boundary} step", ok)
            del ca

            # run()
            ca = make_ca(init, cfg["states"], cfg["groups"], cfg["rule"])
            ca.set_boundary(boundary)
            frames = ca.run(steps)
            ok = all(np.array_equal(frames[s], ref_frames[s]) for s in range(steps))
            check(f"{cfg['name']} b{boundary} run", ok)
            del ca


def test_ca_run_equals_steps(rng):
    print("── CA: run(37) ≡ 37 × step() (grilla rectangular) ──")
    cfg = ca_configs(rng)[0]
    init = rng.integers(0, 2, size=(33, 47)).astype(np.uint8)

    ca1 = make_ca(init, cfg["states"], cfg["groups"], cfg["rule"])
    frames = ca1.run(37)
    ca2 = make_ca(init, cfg["states"], cfg["groups"], cfg["rule"])
    ok = True
    for s in range(37):
        ca2.step()
        ok = ok and np.array_equal(frames[s], ca2.get_space())
    check("run(37) == 37×step()", ok)
    del ca1, ca2


# ── Referencia numpy del NCA ──────────────────────────────────────────────────

def nca_ref_step(grid, nb, layers, out_size):
    """layers = [(W(out,in), b(out,), act_code), ...]; frontera toroidal."""
    def act(v, code):
        if code == 0:
            return np.maximum(0.0, v)
        if code == 1:
            return 1.0 / (1.0 + np.exp(-v))
        if code == 2:
            return np.tanh(v)
        return v

    n = len(nb)
    H, W_ = grid.shape
    X = np.empty((H, W_, n), dtype=np.float32)
    for i, (dx, dy) in enumerate(nb):
        X[..., i] = np.roll(np.roll(grid, -dy, axis=0), -dx, axis=1)
    v = X.reshape(-1, n)
    for Wm, b, code in layers:
        v = act(v @ Wm.T + b, code).astype(np.float32)
    if out_size == 1:
        return v[:, 0].reshape(H, W_)
    return np.argmax(v, axis=1).astype(np.float32).reshape(H, W_)


def make_nca(grid, nb, layers):
    H, W = grid.shape
    nca = ca_module.NCA(W, H, grid, np.ascontiguousarray(nb, dtype=np.int32), len(nb))
    nca.set_weights(
        [l[0] for l in layers], [l[1] for l in layers],
        [l[0].shape[1] for l in layers], [l[0].shape[0] for l in layers],
        [l[2] for l in layers],
    )
    return nca


def random_layers(rng, sizes, acts):
    layers = []
    for i in range(len(sizes) - 1):
        Wm = (rng.standard_normal((sizes[i + 1], sizes[i])) * 0.5).astype(np.float32)
        b = (rng.standard_normal(sizes[i + 1]) * 0.1).astype(np.float32)
        layers.append((Wm, b, acts[i]))
    return layers


def test_nca_reference(rng):
    print("── NCA: binario y multiclase vs referencia numpy ──")
    nb = np.array([[dx, dy] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], dtype=np.int32)

    # Binario: 9 → 16:relu → 1:sigmoid, 5 pasos independientes
    layers = random_layers(rng, [9, 16, 1], [0, 1])
    ok = True
    for _ in range(5):
        init = rng.random((21, 34)).astype(np.float32)
        nca = make_nca(init, nb, layers)
        nca.step()
        got = np.asarray(nca.get_space())
        ref = nca_ref_step(init, nb, layers, out_size=1)
        ok = ok and np.allclose(got, ref, atol=1e-4)
        del nca
    check("NCA binario (sigmoid) vs numpy", ok)

    # Multiclase: 9 → 16:tanh → 3:linear (argmax en kernel)
    layers = random_layers(rng, [9, 16, 3], [2, 3])
    ok = True
    for _ in range(5):
        init = rng.integers(0, 3, size=(21, 34)).astype(np.float32)
        nca = make_nca(init, nb, layers)
        nca.step()
        got = np.asarray(nca.get_space())
        ref = nca_ref_step(init, nb, layers, out_size=3)
        ok = ok and (got == ref).mean() >= 0.999  # tolerancia por empates float
        del nca
    check("NCA multiclase (argmax) vs numpy", ok)


# ── Modo estocástico ──────────────────────────────────────────────────────────

def test_stochastic_ca(rng):
    print("── CA estocástico ──")
    cfg = ca_configs(rng)[0]  # Life
    init = rng.integers(0, 2, size=(64, 64)).astype(np.uint8)

    # p = 1.0 explícito ≡ determinista (bit a bit)
    ca_det = make_ca(init, cfg["states"], cfg["groups"], cfg["rule"])
    det_frames = ca_det.run(20)
    del ca_det
    ca_p1 = make_ca(init, cfg["states"], cfg["groups"], cfg["rule"])
    ca_p1.set_update_prob(1.0)
    ca_p1.set_seed(42)
    p1_frames = ca_p1.run(20)
    del ca_p1
    check("CA p=1.0 ≡ determinista", np.array_equal(det_frames, p1_frames))

    # Misma semilla → misma salida (run y step)
    def stoch_run(seed, use_step=False):
        ca = make_ca(init, cfg["states"], cfg["groups"], cfg["rule"])
        ca.set_update_prob(0.5)
        ca.set_seed(seed)
        if use_step:
            frames = []
            for _ in range(20):
                ca.step()
                frames.append(np.asarray(ca.get_space()).copy())
            out = np.stack(frames)
        else:
            out = ca.run(20)
        del ca
        return out

    a = stoch_run(42)
    b = stoch_run(42)
    check("CA misma semilla → misma salida", np.array_equal(a, b))

    # run() y step() consumen el mismo stream de RNG
    c = stoch_run(42, use_step=True)
    check("CA estocástico run ≡ step (misma semilla)", np.array_equal(a, c))

    # Semilla distinta → salida distinta
    d = stoch_run(43)
    check("CA semillas distintas → salidas distintas", not np.array_equal(a, d))

    # Fracción actualizada ≈ p (regla flip: vecindad = solo el centro)
    flip_groups = [[[0, 0]]]
    zeros = np.zeros((256, 256), dtype=np.uint8)
    ca = make_ca(zeros, 2, flip_groups, [1, 0])
    ca.set_update_prob(0.3)
    ca.set_seed(7)
    ca.step()
    frac = np.asarray(ca.get_space()).mean()
    check(f"CA fracción actualizada ≈ p (0.3, medido {frac:.4f})",
          abs(frac - 0.3) < 0.01)
    del ca


def test_stochastic_nca(rng):
    print("── NCA estocástico ──")
    nb = np.array([[dx, dy] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], dtype=np.int32)
    layers = random_layers(rng, [9, 16, 1], [0, 1])
    init = rng.random((32, 32)).astype(np.float32)

    # p = 1.0 explícito ≡ determinista
    nca = make_nca(init, nb, layers)
    det = nca.run(10)
    del nca
    nca = make_nca(init, nb, layers)
    nca.set_update_prob(1.0)
    nca.set_seed(42)
    p1 = nca.run(10)
    del nca
    check("NCA p=1.0 ≡ determinista", np.array_equal(det, p1))

    def stoch_run(seed):
        nca = make_nca(init, nb, layers)
        nca.set_update_prob(0.5)
        nca.set_seed(seed)
        out = nca.run(10)
        del nca
        return out

    a = stoch_run(42)
    b = stoch_run(42)
    check("NCA misma semilla → misma salida", np.array_equal(a, b))
    c = stoch_run(43)
    check("NCA semillas distintas → salidas distintas", not np.array_equal(a, c))

    # Fracción actualizada ≈ p (red identidad+1: solo el centro, lineal)
    nb1 = np.array([[0, 0]], dtype=np.int32)
    plus_one = [(np.array([[1.0]], dtype=np.float32),
                 np.array([1.0], dtype=np.float32), 3)]
    zeros = np.zeros((256, 256), dtype=np.float32)
    nca = make_nca(zeros, nb1, plus_one)
    nca.set_update_prob(0.4)
    nca.set_seed(7)
    nca.step()
    frac = np.asarray(nca.get_space()).mean()
    check(f"NCA fracción actualizada ≈ p (0.4, medido {frac:.4f})",
          abs(frac - 0.4) < 0.01)
    del nca


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(1234)
    test_ca_bit_exact(rng)
    test_ca_run_equals_steps(rng)
    test_nca_reference(rng)
    test_stochastic_ca(rng)
    test_stochastic_nca(rng)
    total = CHECKS["pass"] + CHECKS["fail"]
    print(f"\n{CHECKS['pass']}/{total} checks OK, {CHECKS['fail']} fallos")
    sys.exit(1 if CHECKS["fail"] else 0)


if __name__ == "__main__":
    main()
