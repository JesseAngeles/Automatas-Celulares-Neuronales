import argparse
import numpy as np
import torch
import torch.nn as nn


ACT_MAP  = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'linear': nn.Identity}
ACT_CODE = {'relu': 0,       'sigmoid': 1,           'tanh': 2,       'linear': 3}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train NCA weights from CA evolution data")
    p.add_argument("--data",         required=True,            help="Path to .npy file (T, H, W)")
    p.add_argument("--output",       default="weights.npz",    help="Output .npz (default: weights.npz)")
    p.add_argument("--neighbourhood",default=None,             help='Pairs "dx,dy" separated by spaces')
    p.add_argument("--radius",       type=int,   default=1,    help="Square radius (2r+1)×(2r+1) (default: 1)")
    p.add_argument("--layers",       default=None,             help='Hidden layers spec: "neurons:act ..." e.g. "64:relu 32:tanh"')
    p.add_argument("--hidden",       type=int,   default=64,   help="Hidden neurons when --layers not set (default: 64)")
    p.add_argument("--out-size",     type=int,   default=1,    help="Output layer neurons (default: 1)")
    p.add_argument("--out-act",      default="sigmoid",        help="Output activation (default: sigmoid)")
    p.add_argument("--epochs",       type=int,   default=20000,help="Max epochs (default: 20000)")
    p.add_argument("--batch",        type=int,   default=8192, help="Batch size (default: 8192)")
    p.add_argument("--lr",           type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    p.add_argument("--loss-target",  type=float, default=0.01, help="Stop when loss < value (default: 0.01)")
    p.add_argument("--print-every",  type=int,   default=50,   help="Print every N epochs (default: 50)")
    return p.parse_args()


# ── Vecindad ──────────────────────────────────────────────────────────────────

def build_neighbourhood(args) -> np.ndarray:
    if args.neighbourhood:
        pairs = args.neighbourhood.strip().split()
        nb = [tuple(int(v) for v in pair.split(",")) for pair in pairs]
    else:
        r = args.radius
        nb = [(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
    return np.array(nb, dtype=np.int32)   # (n_size, 2)


# ── Capas ─────────────────────────────────────────────────────────────────────

def parse_layers_arg(s: str) -> list:
    """'64:relu 32:tanh' → [(64, 'relu'), (32, 'tanh')]"""
    layers = []
    for part in s.strip().split():
        neurons, act = part.split(':')
        layers.append((int(neurons), act.lower()))
    return layers


# ── Extracción de muestras ────────────────────────────────────────────────────

def extract_patches(data: np.ndarray, neighbourhood: np.ndarray):
    src = data[:-1]
    tgt = data[1:]
    patches = np.stack([
        np.roll(np.roll(src, -int(dy), axis=1), -int(dx), axis=2)
        for dx, dy in neighbourhood
    ], axis=-1)                                   # (T-1, H, W, n_size)
    X = patches.reshape(-1, len(neighbourhood))  # (N, n_size)
    Y = tgt.reshape(-1)                          # (N,)
    return X, Y


# ── Modelo ────────────────────────────────────────────────────────────────────

def build_model(n_size: int, hidden_layers: list, out_size: int, out_act: str) -> nn.Sequential:
    mods = []
    in_f = n_size
    for neurons, act in hidden_layers:
        mods.append(nn.Linear(in_f, neurons))
        mods.append(ACT_MAP[act]())
        in_f = neurons
    mods.append(nn.Linear(in_f, out_size))
    mods.append(ACT_MAP[out_act]())
    return nn.Sequential(*mods)


# ── Guardar pesos ─────────────────────────────────────────────────────────────

def save_weights(path: str, model: nn.Sequential, neighbourhood: np.ndarray,
                 hidden_layers: list, out_act: str):
    linears  = [l for l in model if isinstance(l, nn.Linear)]
    all_acts = [act for _, act in hidden_layers] + [out_act]

    save_dict = {}
    for i, layer in enumerate(linears):
        save_dict[f'weights_{i}'] = layer.weight.data.cpu().numpy()
        save_dict[f'biases_{i}']  = layer.bias.data.cpu().numpy()

    save_dict['layer_in']     = np.array([l.in_features  for l in linears])
    save_dict['layer_out']    = np.array([l.out_features for l in linears])
    save_dict['activations']  = np.array([ACT_CODE[a]   for a in all_acts])
    save_dict['neighbourhood']= neighbourhood

    np.savez(path, **save_dict)
    print(f"Pesos guardados → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    neighbourhood = build_neighbourhood(args)
    n_size        = len(neighbourhood)
    print(f"Vecindad: {n_size} celdas  {neighbourhood.tolist()}")

    hidden_layers = parse_layers_arg(args.layers) if args.layers else [(args.hidden, 'relu')]
    out_size      = args.out_size
    out_act       = args.out_act
    arch_str = " → ".join(f"{n}:{a}" for n, a in hidden_layers) + f" → {out_size}:{out_act}"
    print(f"Arquitectura: {n_size} → {arch_str}")

    print("Leyendo datos...")
    data = np.load(args.data).astype(np.float32)
    print(f"Shape: {data.shape}")

    X_np, Y_np = extract_patches(data, neighbourhood)
    print(f"Muestras: {len(X_np):,}")

    n_classes_data = int(Y_np.max()) + 1
    multiclass = out_size > 1

    if multiclass and out_size != n_classes_data:
        print(f"ADVERTENCIA: --out-size {out_size} pero los datos tienen "
              f"{n_classes_data} clases (0..{n_classes_data-1}). "
              f"Ajustando out_size → {n_classes_data}.")
        out_size = n_classes_data

    X = torch.tensor(X_np, dtype=torch.float32)
    Y = (torch.tensor(Y_np, dtype=torch.long)
         if multiclass
         else torch.tensor(Y_np, dtype=torch.float32).unsqueeze(1))
    N = X.shape[0]

    if multiclass:
        print(f"Modo multiclase: {out_size} estados  (CrossEntropyLoss)")
    else:
        print("Modo binario: 2 estados  (BCELoss)")

    print("Construyendo modelo...")
    model     = build_model(n_size, hidden_layers, out_size, out_act).to(device)
    criterion = nn.CrossEntropyLoss() if multiclass else nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_batches = (N + args.batch - 1) // args.batch

    on_gpu     = False
    data_bytes = X.nbytes + Y.nbytes
    OVERHEAD   = 2 * 1024 ** 3
    if device.type == "cuda":
        total_vram = torch.cuda.get_device_properties(device).total_memory
        if data_bytes * 2 + OVERHEAD < total_vram:
            X, Y   = X.to(device), Y.to(device)
            on_gpu = True
            print(f"Datos en GPU  ({data_bytes/1e9:.2f} GB)")
        else:
            X, Y = X.pin_memory(), Y.pin_memory()
            print(f"Datos en CPU/pin_memory  ({data_bytes/1e9:.2f} GB, "
                  f"VRAM total: {total_vram/1e9:.2f} GB)")

    print("Training...")
    try:
        for epoch in range(args.epochs):
            if on_gpu:
                perm   = torch.randperm(N, device=device)
                X_shuf = X[perm]
                Y_shuf = Y[perm]
                def get_batch(s):
                    return X_shuf[s:s+args.batch], Y_shuf[s:s+args.batch]
            else:
                perm   = torch.randperm(N)
                X_shuf = X[perm]
                Y_shuf = Y[perm]
                def get_batch(s):
                    return (X_shuf[s:s+args.batch].to(device, non_blocking=True),
                            Y_shuf[s:s+args.batch].to(device, non_blocking=True))

            epoch_loss = 0.0
            for start in range(0, N, args.batch):
                X_b, Y_b = get_batch(start)
                pred = model(X_b)
                loss = criterion(pred, Y_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / n_batches
            if epoch % args.print_every == 0:
                print(f"epoch: {epoch:>6}  loss: {avg_loss:.6f}")
            if avg_loss < args.loss_target:
                print(f"Convergió en epoch {epoch}  loss: {avg_loss:.6f}")
                break

    except KeyboardInterrupt:
        print("\nInterrumpido.")

    save_weights(args.output, model, neighbourhood, hidden_layers, out_act)


if __name__ == "__main__":
    main()
