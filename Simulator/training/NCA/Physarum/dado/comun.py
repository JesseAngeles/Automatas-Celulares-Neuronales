"""Utilidades del experimento del dado oculto (ver PLAN.md).

Maestra = la red de 30 min que se mueve al muestrear (NCA/Physarum/prueba/160_6estados/red.npz).
Estado combinado = visible * DADOS + dado; la paleta repite el color del visible en sus DADOS subestados.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

CARPETA = Path(__file__).resolve().parent
TRAINING = CARPETA.parents[2]
sys.path.insert(0, str(TRAINING / "scripts"))
from barrido_flame import cargar_red                                       # noqa: E402

CA = TRAINING / "CA" / "Physarum" / "estados_cada30min_160"
MAESTRA = TRAINING / "NCA" / "Physarum" / "prueba" / "160_6estados" / "red.npz"
PL = 3                                                                     # primer estado de plasmodio
NV = 6                                                                     # estados visibles
PRUEBA = [f"physarum-{e}-dish5-r2c1" for e in ("001", "002", "008", "009")]
CALIBRACION = [f"physarum-{e}-dish{d}-r0c{d}" for e in ("001", "002", "008", "009") for d in (0, 1)]
PALETA6 = ["#000000", "#2A2A30", "#E6CFA8", "#6E6A12", "#D2AE00", "#FFF46A"]


def cajas_entrenamiento():
    return [p.stem for p in sorted(CA.glob("physarum-*.npy")) if p.stem not in PRUEBA]


def cargar(caja):
    return np.load(CA / f"{caja}.npy")


class Red:
    """Red de main.py evaluada como el núcleo de la GUI (frontera abierta = 0) sobre lotes (B, H, W)."""

    def __init__(self, ruta, dev):
        self.red, nb = cargar_red(str(ruta), dev)
        self.nb = [tuple(map(int, p)) for p in nb]
        self.r = max(max(abs(dx), abs(dy)) for dx, dy in self.nb)

    @torch.no_grad()
    def logits(self, E):
        B, H, W = E.shape
        r = self.r
        Pd = torch.nn.functional.pad(E, (r, r, r, r))
        X = torch.stack([Pd[:, r + dy:r + dy + H, r + dx:r + dx + W] for dx, dy in self.nb], -1).reshape(-1, len(self.nb))
        return torch.cat([self.red(X[s:s + (1 << 18)]) for s in range(0, len(X), 1 << 18)]).view(B, H, W, -1)


# ── Métricas ───────────────────────────────────────────────────────────────────────────────────────────────────

def puntaje_t(real, sim):
    """IoU × (área menor / área mayor) del plasmodio por paso: real (T, H, W), sim (B, T, H, W) → (B, T)."""
    a, b = (real >= PL).unsqueeze(0), sim >= PL
    inter, union = (a & b).flatten(2).sum(-1).float(), (a | b).flatten(2).sum(-1).float()
    iou = torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))
    na, nb = a.flatten(2).sum(-1).float(), b.flatten(2).sum(-1).float()
    r = torch.where(torch.maximum(na, nb) > 0, torch.minimum(na, nb) / torch.maximum(na, nb).clamp(min=1),
                    torch.ones_like(nb))
    return iou * r


def brotes(X, r=3):
    """Plasmodio nuevo en t+1 sin plasmodio a distancia ≤ r en t, por célula de la caja y paso: (…, T, H, W) → (…)."""
    p = (X >= PL).float()
    cerca = torch.nn.functional.max_pool2d(p.flatten(0, -3).unsqueeze(1), 2 * r + 1, 1, r).squeeze(1).view(p.shape) > 0
    nuevo = (X[..., 1:, :, :] >= PL) & ~cerca[..., :-1, :, :]
    d = (X[..., :1, :, :] != 0)
    return nuevo.flatten(-3).sum(-1).float() / (d.flatten(-3).sum(-1).float() * (X.shape[-3] - 1))


def frontera(X):
    """Fracción del plasmodio que toca (4 vecinos) algo que no es plasmodio: (…, H, W) → (…)."""
    p = X >= PL
    q = torch.nn.functional.pad(p, (1, 1, 1, 1))
    interior = q[..., :-2, 1:-1] & q[..., 2:, 1:-1] & q[..., 1:-1, :-2] & q[..., 1:-1, 2:]
    n = p.flatten(-2).sum(-1).float()
    return torch.where(n > 0, (p & ~interior).flatten(-2).sum(-1).float() / n.clamp(min=1), torch.full_like(n, float("nan")))


def interno(X):
    """Fracción del plasmodio (en t y t+1) que cambia de nivel entre pasos: (…, T, H, W) → (…, T-1)."""
    a, b = X[..., :-1, :, :], X[..., 1:, :, :]
    m = (a >= PL) & (b >= PL)
    n = m.flatten(-2).sum(-1).float()
    return torch.where(n > 0, (m & (a != b)).flatten(-2).sum(-1).float() / n.clamp(min=1),
                       torch.full_like(n, float("nan")))


def resumen(real, sim):
    """Métricas de una caja: real (T, H, W), sim (B, T, H, W), ambos en la GPU, estados visibles."""
    d = real[0] != 0
    T = len(real)
    mitad = slice(T // 2, T)
    return {
        "puntaje": puntaje_t(real, sim)[:, 1:].mean(1).cpu().numpy(),
        "area_fin": ((sim[:, -1] >= PL) & d).flatten(1).sum(1).float().cpu().numpy() / d.sum().item(),
        "area_real_fin": ((real[-1] >= PL) & d).sum().item() / d.sum().item(),
        "frontera": torch.nanmean(frontera(sim[:, mitad]), 1).cpu().numpy(),
        "frontera_real": torch.nanmean(frontera(real[mitad])).item(),
        "interno": torch.nanmean(interno(sim[:, mitad]), 1).cpu().numpy(),
        "interno_real": torch.nanmean(interno(real[mitad])).item(),
        "brotes": brotes(sim).cpu().numpy(),
        "brotes_real": brotes(real).item(),
    }


def guardar_json(ruta, obj):
    Path(ruta).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float))


# ── Proceso ideal: maestra + dado oculto ───────────────────────────────────────────────────────────────────────
# Estado combinado E = visible * DADOS + dado. El dado (2 bits a, b) sigue la regla de regla_dado.py; el azar de
# cada célula es u = Φ(Σ_k w_k z(dado_{x+k}) / ‖w‖) con w gaussiana (SIGMA) en la ventana de RADIO_RUIDO, así que
# células vecinas reciben azar parecido. El visible siguiente sale de la maestra por CDF inversa con ese u.

DADOS = 4
SIGMA = 2.0
RADIO_RUIDO = 4                                                            # ventana 9 × 9
NE = NV * DADOS                                                            # 24 estados combinados
PMIN = 0.0                                                                 # no se sortean estados con p < PMIN
LOCAL = True                      # el plasmodio solo aparece si hay plasmodio en la vecindad de la alumna en t
CRECER = 0.0                      # se suma a los logits de plasmodio de la maestra (empujón de crecimiento)


def pesos_ruido(dev):
    x = torch.arange(-RADIO_RUIDO, RADIO_RUIDO + 1, device=dev, dtype=torch.float32)
    w = torch.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / (2 * SIGMA ** 2))
    return w / w.norm()


def vecindad_alumna(maestra):
    """9 × 9 densa (para el ruido) ∪ vecindad de la maestra (anillos hasta radio 6)."""
    nb = [(dx, dy) for dy in range(-RADIO_RUIDO, RADIO_RUIDO + 1) for dx in range(-RADIO_RUIDO, RADIO_RUIDO + 1)]
    return nb + [p for p in maestra.nb if p not in nb]


def z_dado(D):
    return (D.float() - 1.5) / 1.25 ** 0.5


def azar(D, w):
    """D (B, H, W) dados → u (B, H, W) uniforme correlacionado; fuera del espacio el estado es 0 (dado 0)."""
    r = RADIO_RUIDO
    Z = torch.nn.functional.pad(z_dado(D), (r, r, r, r), value=float(z_dado(torch.zeros(1))[0]))
    s = torch.nn.functional.conv2d(Z.unsqueeze(1), w.view(1, 1, *w.shape)).squeeze(1)
    return 0.5 * (1 + torch.erf(s / 2 ** 0.5))


def paso_dado(D):
    """Regla del dado con frontera abierta: a' = a_O ^ (a | a_E) ^ b, b' = b_N ^ (b | b_S) ^ a."""
    a, b = (D >> 1) & 1, D & 1

    def v(X, dx, dy):
        H, W = X.shape[-2:]
        P = torch.nn.functional.pad(X, (1, 1, 1, 1))
        return P[..., 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]

    a2 = v(a, -1, 0) ^ (a | v(a, 1, 0)) ^ b
    b2 = v(b, 0, -1) ^ (b | v(b, 0, 1)) ^ a
    return (a2 << 1) | b2


@torch.no_grad()
def paso_ideal(maestra, E, w):
    """Un paso del proceso que la alumna debe imitar: E (B, H, W) combinado → combinado."""
    V, D = E // DADOS, E % DADOS
    L = maestra.logits(V)
    if CRECER:
        L = torch.cat([L[..., :PL], L[..., PL:] + CRECER], -1)
    p = L.softmax(-1)
    if PMIN > 0:                                                           # quita los eventos raros (brotes)
        p = torch.where(p >= PMIN, p, torch.zeros_like(p))
        p = p / p.sum(-1, keepdim=True)
    if LOCAL:
        nb = vecindad_alumna(maestra)
        r = max(max(abs(dx), abs(dy)) for dx, dy in nb)
        P = torch.nn.functional.pad(V >= PL, (r, r, r, r))
        H, W = V.shape[-2:]
        cerca = torch.stack([P[:, r + dy:r + dy + H, r + dx:r + dx + W] for dx, dy in nb]).any(0)
        p = torch.where((cerca | (V >= PL)).unsqueeze(-1), p,
                        torch.cat([p[..., :PL], torch.zeros_like(p[..., PL:])], -1))
        p = p / p.sum(-1, keepdim=True)
    u = azar(D, w)
    V2 = (p.cumsum(-1) <= u.unsqueeze(-1)).sum(-1).clamp(max=NV - 1)
    return (V2 * DADOS + paso_dado(D)).to(torch.int8)


def combinar(V, g, dev):
    """Visible (…, H, W) → combinado con dados al azar (generador g)."""
    D = torch.randint(0, DADOS, V.shape, generator=g, device=dev, dtype=torch.int8)
    return (V.to(dev).to(torch.int8) * DADOS + D).to(torch.int8)


@torch.no_grad()
def libre_ideal(maestra, E0, T, w):
    """(B, T, H, W) combinados."""
    E = E0
    ev = [E]
    for _ in range(1, T):
        E = paso_ideal(maestra, E, w)
        ev.append(E)
    return torch.stack(ev, 1)


# Paleta de los 24 estados (visible × 4 + dado): un color distinto por estado, parecido dentro de cada visible,
# para que se entienda (amarillos = plasmodio) y se vea que todos los estados están en pantalla.
PALETA24 = [
    "#000000", "#1A1A1A", "#0E1626", "#24160E",                            # borde
    "#2A2A30", "#3C3C46", "#283444", "#3E3230",                            # fondo
    "#E6CFA8", "#D4B482", "#F3E4C8", "#C8A47A",                            # avena
    "#6E6A12", "#8C7E1A", "#55560A", "#76601E",                            # plasmodio tenue
    "#D2AE00", "#E8920A", "#B49400", "#EAC424",                            # plasmodio medio
    "#FFF46A", "#FFDD3A", "#FFFAB4", "#FFC850",                            # plasmodio intenso
]
