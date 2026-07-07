import argparse
import numpy as np
import torch
import torch.nn as nn

from main import build_neighbourhood, parse_layers_arg, build_model, save_weights


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Optimize NCA weights so that initial state → target state in n steps (BPTT)")
    p.add_argument("--initial",      required=True,            help=".npy initial state (H,W) o (H,W,C); una evolución (T,...) usa el frame 0")
    p.add_argument("--target",       required=True,            help=".npy target state (H,W) o (H,W,C); una evolución (T,...) usa el último frame")
    p.add_argument("--steps",        type=int,   required=True,help="n: steps to evolve initial → target")
    p.add_argument("--neighbourhood",default=None,             help='Pairs "dx,dy" separated by spaces')
    p.add_argument("--radius",       type=int,   default=1,    help="Square radius (2r+1)×(2r+1) (default: 1)")
    p.add_argument("--channels",     type=int,   default=1,    help="Float channels per cell (default: 1)")
    p.add_argument("--layers",       default=None,             help='Hidden layers spec: "neurons:act ..." e.g. "64:relu 32:tanh"')
    p.add_argument("--hidden",       type=int,   default=64,   help="Hidden neurons when --layers not set (default: 64)")
    p.add_argument("--out-act",      default="sigmoid",        help="Output activation (default: sigmoid)")
    p.add_argument("--lr",           type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    p.add_argument("--epochs",       type=int,   default=5000, help="Max epochs (default: 5000)")
    p.add_argument("--loss-target",  type=float, default=0.01, help="Stop when loss < value (default: 0.01)")
    p.add_argument("--l2",           type=float, default=0.0,  help="L2 regularization on weights (default: 0)")
    p.add_argument("--output",       default="weights.npz",    help="Output .npz (default: weights.npz)")
    p.add_argument("--print-every",  type=int,   default=50,   help="Print every N epochs (default: 50)")
    p.add_argument("--seed",         type=int,   default=None, help="Random seed (default: random)")
    return p.parse_args()


# ── Estados ───────────────────────────────────────────────────────────────────

def load_state(path: str, channels: int, which: str) -> np.ndarray:
    """Carga un estado (H, W[, C]); si el archivo es una evolución (T, ...),
    usa el frame 0 para --initial y el último para --target. Devuelve (H, W, C)."""
    arr = np.load(path).astype(np.float32)
    state_ndim = 2 if channels == 1 else 3
    if arr.ndim == state_ndim + 1:
        idx = 0 if which == "initial" else -1
        print(f"{which}: {path} es una evolución {arr.shape}; usando frame {idx}")
        arr = arr[idx]
    if arr.ndim != state_ndim or (channels > 1 and arr.shape[-1] != channels):
        expected = "(H, W)" if channels == 1 else f"(H, W, {channels})"
        raise SystemExit(f"Error: --{which} debe ser {expected}; el archivo tiene shape {arr.shape}")
    if channels == 1:
        arr = arr[..., None]
    return arr


# ── Forward diferenciable (réplica del kernel NCA_GPU) ────────────────────────

def nca_forward(model: nn.Sequential, state: torch.Tensor, nb, n_steps: int) -> torch.Tensor:
    """Desenrolla n_steps del NCA sobre state (C, H, W): frontera toroidal y
    entrada del MLP = C canales de cada vecino, consecutivos, en el orden de
    la vecindad (mismo layout que NCA_GPU / extract_patches)."""
    C, H, W = state.shape
    for _ in range(n_steps):
        feats = []
        for dx, dy in nb:
            shifted = torch.roll(state, shifts=(-dy, -dx), dims=(1, 2))
            feats.append(shifted.permute(1, 2, 0).reshape(H * W, C))
        X = torch.cat(feats, dim=1)               # (H*W, n_size*C)
        out = model(X)                            # (H*W, C)
        state = out.permute(1, 0).reshape(C, H, W)
    return state


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Seed: {args.seed}")

    neighbourhood = build_neighbourhood(args)
    n_size        = len(neighbourhood)
    print(f"Vecindad: {n_size} celdas  {neighbourhood.tolist()}")

    channels      = args.channels
    hidden_layers = parse_layers_arg(args.layers) if args.layers else [(args.hidden, 'relu')]
    in_size       = n_size * channels
    arch_str = " → ".join(f"{n}:{a}" for n, a in hidden_layers) + f" → {channels}:{args.out_act}"
    print(f"Arquitectura: {in_size} → {arch_str}")

    initial = load_state(args.initial, channels, "initial")
    target  = load_state(args.target,  channels, "target")
    if initial.shape != target.shape:
        raise SystemExit(f"Error: initial {initial.shape} y target {target.shape} no coinciden")
    print(f"Estado: {initial.shape[1]}×{initial.shape[0]}×{channels}  |  n = {args.steps} pasos")

    state0 = torch.from_numpy(initial).permute(2, 0, 1).contiguous().to(device)  # (C, H, W)
    tgt    = torch.from_numpy(target).permute(2, 0, 1).contiguous().to(device)
    nb     = [(int(dx), int(dy)) for dx, dy in neighbourhood]

    model     = build_model(in_size, hidden_layers, channels, args.out_act).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse       = nn.MSELoss()
    lin_weights = [m.weight for m in model if isinstance(m, nn.Linear)]

    print("Optimizing...")
    try:
        for epoch in range(args.epochs):
            pred = nca_forward(model, state0, nb, args.steps)
            loss = mse(pred, tgt)
            if args.l2 > 0:
                loss = loss + args.l2 * sum((w ** 2).sum() for w in lin_weights)
            optimizer.zero_grad()
            loss.backward()
            # BPTT sobre n pasos: clip del gradiente para evitar explosiones
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_val = loss.item()
            if epoch % args.print_every == 0:
                print(f"epoch: {epoch:>6}  loss: {loss_val:.6f}")
            if loss_val < args.loss_target:
                print(f"Convergió en epoch {epoch}  loss: {loss_val:.6f}")
                break

    except KeyboardInterrupt:
        print("\nInterrumpido.")

    save_weights(args.output, model, neighbourhood, hidden_layers, args.out_act, channels)


if __name__ == "__main__":
    main()
