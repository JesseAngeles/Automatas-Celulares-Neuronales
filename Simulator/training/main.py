import argparse
import numpy as np
import torch
import torch.nn as nn


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train NCA weights from CA evolution data")
    p.add_argument("--data",         required=True,            help="Path to .npy file (T, H, W)")
    p.add_argument("--output",       default="weights.npz",    help="Output .npz  (default: weights.npz)")
    p.add_argument("--neighbourhood",default=None,             help='Pares "dx,dy" separados por espacio')
    p.add_argument("--radius",       type=int,   default=1,    help="Radio cuadrado (2r+1)×(2r+1) (default: 1 → 3×3)")
    p.add_argument("--hidden",       type=int,   default=64,   help="Neuronas ocultas (default: 64)")
    p.add_argument("--epochs",       type=int,   default=20000,help="Epochs máximos (default: 20000)")
    p.add_argument("--batch",        type=int,   default=8192, help="Batch size (default: 8192)")
    p.add_argument("--lr",           type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    p.add_argument("--loss-target",  type=float, default=0.01, help="Parar cuando loss < valor (default: 0.01)")
    p.add_argument("--print-every",  type=int,   default=50,   help="Imprimir cada N epochs (default: 50)")
    p.add_argument("--inverse",      action="store_true",      help="Regla inversa: vecindad t+1 → celda t")
    return p.parse_args()


# ── Vecindad ──────────────────────────────────────────────────────────────────

def build_neighbourhood(args) -> np.ndarray:
    if args.neighbourhood:
        pairs = args.neighbourhood.strip().split()
        nb = [tuple(int(v) for v in pair.split(",")) for pair in pairs]
    else:
        r = args.radius
        nb = [(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
    return np.array(nb, dtype=np.int32)   # (n_size, 2) — cada fila [dx, dy]


# ── Extracción de muestras ────────────────────────────────────────────────────

def extract_patches(data: np.ndarray, neighbourhood: np.ndarray, inverse: bool):
    """
    Boundary periódico. Devuelve (X, Y) como arrays planos.
    Forward:  X = vecindad en t,   Y = celda en t+1
    Inverse:  X = vecindad en t+1, Y = celda en t
    """
    src = data[1:]  if inverse else data[:-1]   # frames fuente para X
    tgt = data[:-1] if inverse else data[1:]    # frames objetivo para Y

    patches = np.stack([
        np.roll(np.roll(src, -int(dy), axis=1), -int(dx), axis=2)
        for dx, dy in neighbourhood
    ], axis=-1)                                  # (T-1, H, W, n_size)

    X = patches.reshape(-1, len(neighbourhood))  # (N, n_size)
    Y = tgt.reshape(-1)                          # (N,)
    return X, Y


# ── Modelo ────────────────────────────────────────────────────────────────────

def build_model(n_size: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_size, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
        nn.Sigmoid(),
    )


# ── Guardar pesos ─────────────────────────────────────────────────────────────

def save_weights(path: str, model: nn.Sequential, neighbourhood: np.ndarray):
    np.savez(path,
        weights_0    = model[0].weight.data.cpu().numpy(),
        biases_0     = model[0].bias.data.cpu().numpy(),
        weights_1    = model[2].weight.data.cpu().numpy(),
        biases_1     = model[2].bias.data.cpu().numpy(),
        layer_in     = [model[0].in_features,  model[2].in_features],
        layer_out    = [model[0].out_features, model[2].out_features],
        activations  = [0, 1],
        neighbourhood= neighbourhood,
    )
    print(f"Pesos guardados → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    neighbourhood = build_neighbourhood(args)
    n_size        = len(neighbourhood)
    print(f"Vecindad: {n_size} celdas  {neighbourhood.tolist()}")
    print(f"Regla: {'inversa' if args.inverse else 'forward'}")

    print("Leyendo datos...")
    data = np.load(args.data).astype(np.float32)
    print(f"Shape: {data.shape}")

    X_np, Y_np = extract_patches(data, neighbourhood, args.inverse)
    print(f"Muestras: {len(X_np):,}")

    X = torch.tensor(X_np, dtype=torch.float32)
    Y = torch.tensor(Y_np, dtype=torch.float32).unsqueeze(1)
    N = X.shape[0]

    # Construir modelo primero para medir la VRAM real restante.
    print("Construyendo modelo...")
    model     = build_model(n_size, args.hidden).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_batches = (N + args.batch - 1) // args.batch

    # Decidir si los datos caben en VRAM.
    # Necesitamos 2× data_bytes (datos + copia shuffle) + 2 GB de overhead
    # (CUDA context, cuDNN, activaciones intermedias, display).
    on_gpu     = False
    data_bytes = X.nbytes + Y.nbytes
    OVERHEAD   = 2 * 1024 ** 3   # 2 GB fijo
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
                X_shuf = X[perm]   # shuffle en CPU
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

    save_weights(args.output, model, neighbourhood)


if __name__ == "__main__":
    main()
