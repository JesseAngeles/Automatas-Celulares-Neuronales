import argparse
import csv
import time
import numpy as np
import torch
import torch.nn as nn


ACT_MAP  = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'linear': nn.Identity}
ACT_CODE = {'relu': 0,       'sigmoid': 1,           'tanh': 2,       'linear': 3}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train NCA weights from CA evolution data")
    p.add_argument("--data",         required=True, nargs="+", help="Path(s) to .npy or .mat file (T, H, W) o (N, T, H, W) con N evoluciones; con varios se juntan sus transiciones")
    p.add_argument("--mat-key",      default=None,             help="Nombre de la variable a usar si --data es .mat con varias matrices")
    p.add_argument("--output",       default="weights.npz",    help="Output .npz (default: weights.npz)")
    p.add_argument("--neighbourhood",default=None,             help='Pairs "dx,dy" separated by spaces')
    p.add_argument("--radius",       type=int,   default=1,    help="Square radius (2r+1)×(2r+1) (default: 1)")
    p.add_argument("--layers",       default=None,             help='Hidden layers spec: "neurons:act ..." e.g. "64:relu 32:tanh"')
    p.add_argument("--hidden",       type=int,   default=64,   help="Hidden neurons when --layers not set (default: 64)")
    p.add_argument("--out-size",     type=int,   default=1,    help="Output layer neurons (default: 1)")
    p.add_argument("--out-act",      default="sigmoid",        help="Output activation (default: sigmoid)")
    p.add_argument("--one-hot",      type=int,   default=0,    metavar="ESTADOS", nargs="?", const=-1,
                   help="Codifica cada vecino en one-hot con ESTADOS entradas (vecinos × ESTADOS en la capa de entrada); "
                        "sin número toma los estados de los datos (máximo + 1); 0 = valor crudo (default: 0)")
    p.add_argument("--epochs",       type=int,   default=20000,help="Max epochs (default: 20000)")
    p.add_argument("--batch",        type=int,   default=8192, help="Batch size (default: 8192)")
    p.add_argument("--lr",           type=float, default=1e-3, help="Learning rate inicial (default: 1e-3)")
    p.add_argument("--lr-schedule",  default="exp", choices=["exp", "cosine", "plateau", "none"],
                   help="Decaimiento del learning rate de --lr a --lr-min: exponencial o coseno a lo largo de --epochs, "
                        "ReduceLROnPlateau según la pérdida, o none para dejarlo constante (default: exp)")
    p.add_argument("--lr-min",       type=float, default=1e-6, help="Learning rate final/mínimo (default: 1e-6)")
    p.add_argument("--plateau-patience", type=int, default=20, help="Épocas sin mejorar antes de reducir con plateau (default: 20)")
    p.add_argument("--plateau-factor",   type=float, default=0.5, help="Factor de reducción con plateau (default: 0.5)")
    p.add_argument("--loss-target",  type=float, default=0.01, help="Stop when loss < value (default: 0.01)")
    p.add_argument("--print-every",  type=int,   default=50,   help="Print every N epochs (default: 50)")
    p.add_argument("--seed",         type=int,   default=None, help="Random seed (default: random)")
    p.add_argument("--log-csv",      default=None,             help="Write epoch,loss for each printed message to this CSV path")
    p.add_argument("--open-boundary",action="store_true",      help="Frontera abierta: fuera del espacio todo es 0 (default: periódica)")
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


# ── Carga de datos ────────────────────────────────────────────────────────────

def load_data(path: str, mat_key: str = None) -> np.ndarray:
    """Carga la evolución de entrenamiento desde .npy o .mat."""
    if path.lower().endswith(".mat"):
        import scipy.io
        raw = scipy.io.loadmat(path)
        data = {k: v for k, v in raw.items() if not k.startswith("__")}
        if mat_key:
            if mat_key not in data:
                raise ValueError(f"'{mat_key}' no está en {path}. Disponibles: {', '.join(data)}")
            return data[mat_key]
        if "evolution" in data:
            return data["evolution"]
        if len(data) == 1:
            return next(iter(data.values()))
        raise ValueError(
            f"El archivo .mat contiene varias matrices ({', '.join(data)}); "
            f"usa --mat-key para elegir una."
        )
    return np.load(path)


# ── Extracción de muestras ────────────────────────────────────────────────────

def extract_patches(data: np.ndarray, neighbourhood: np.ndarray, open_boundary: bool = False):
    src = data[:-1]
    tgt = data[1:]
    T1, H, W = src.shape
    n = len(neighbourhood)
    X = np.empty((T1, H, W, n), dtype=np.float32)
    if open_boundary:
        # Frontera abierta: se rellena con ceros, así los vecinos fuera del espacio valen 0
        r = int(np.abs(neighbourhood).max())
        padded = np.pad(src, ((0, 0), (r, r), (r, r)))
    for i, (dx, dy) in enumerate(neighbourhood):
        if open_boundary:
            X[..., i] = padded[:, r + dy:r + dy + H, r + dx:r + dx + W]
        else:
            X[..., i] = np.roll(np.roll(src, -int(dy), axis=1), -int(dx), axis=2)
    X = X.reshape(-1, n)
    Y = tgt.reshape(-1)
    return X, Y


# ── Modelo ────────────────────────────────────────────────────────────────────

class OneHot(nn.Module):
    """(N, vecinos) con estados enteros → (N, vecinos · estados); la entrada del vecino i y estado s es i · estados + s."""
    def __init__(self, states: int):
        super().__init__()
        self.states = states

    def forward(self, x):
        return nn.functional.one_hot(x.long(), self.states).flatten(1).float()


def build_model(n_size: int, hidden_layers: list, out_size: int, out_act: str, one_hot: int = 0) -> nn.Sequential:
    mods = [OneHot(one_hot)] if one_hot else []
    in_f = n_size * one_hot if one_hot else n_size
    for neurons, act in hidden_layers:
        mods.append(nn.Linear(in_f, neurons))
        mods.append(ACT_MAP[act]())
        in_f = neurons
    mods.append(nn.Linear(in_f, out_size))
    mods.append(ACT_MAP[out_act]())
    return nn.Sequential(*mods)


# ── Learning rate ─────────────────────────────────────────────────────────────

def build_scheduler(optimizer, args):
    """Baja el learning rate de --lr a --lr-min; con exp y coseno llega a --lr-min en la última época."""
    if args.lr_schedule == "none":
        return None
    pasos = max(1, args.epochs - 1)
    if args.lr_schedule == "exp":
        gamma = (args.lr_min / args.lr) ** (1 / pasos)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    if args.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pasos, eta_min=args.lr_min)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.plateau_factor, patience=args.plateau_patience, min_lr=args.lr_min)


# ── Guardar pesos ─────────────────────────────────────────────────────────────

def save_weights(path: str, model: nn.Sequential, neighbourhood: np.ndarray,
                 hidden_layers: list, out_act: str, one_hot: int = 0):
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
    save_dict['one_hot_states'] = np.array(one_hot)   # 0 = entrada cruda

    np.savez(path, **save_dict)
    print(f"Pesos guardados → {path}")


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
    else:
        print("Seed: random")

    neighbourhood = build_neighbourhood(args)
    n_size        = len(neighbourhood)
    print(f"Vecindad: {n_size} celdas  {neighbourhood.tolist()}")

    hidden_layers = parse_layers_arg(args.layers) if args.layers else [(args.hidden, 'relu')]
    out_size      = args.out_size
    out_act       = args.out_act
    arch_str = " → ".join(f"{n}:{a}" for n, a in hidden_layers) + f" → {out_size}:{out_act}"
    entrada = f"{n_size} one-hot" if args.one_hot else str(n_size)
    print(f"Arquitectura: {entrada} → {arch_str}")

    print("Leyendo datos...")
    print(f"Frontera: {'abierta' if args.open_boundary else 'periódica'}")
    # Las transiciones se extraen por archivo para no unir el último paso de una evolución con el primero de otra
    partes = []
    for path in args.data:
        data = load_data(path, args.mat_key).astype(np.float32)
        print(f"Shape: {data.shape}  ({path})")
        for evo in (data if data.ndim == 4 else [data]):
            partes.append(extract_patches(evo, neighbourhood, args.open_boundary))
    X_np = np.concatenate([X for X, _ in partes])
    Y_np = np.concatenate([Y for _, Y in partes])
    print(f"Muestras: {len(X_np):,}")

    n_classes_data = int(Y_np.max()) + 1
    multiclass = out_size > 1

    if multiclass and out_size != n_classes_data:
        print(f"ADVERTENCIA: --out-size {out_size} pero los datos tienen "
              f"{n_classes_data} clases (0..{n_classes_data-1}). "
              f"Ajustando out_size → {n_classes_data}.")
        out_size = n_classes_data

    X = torch.from_numpy(X_np)
    Y = (torch.from_numpy(Y_np).long()
         if multiclass
         else torch.from_numpy(Y_np).unsqueeze(1))
    N = X.shape[0]

    if multiclass:
        print(f"Modo multiclase: {out_size} estados  (CrossEntropyLoss)")
    else:
        print("Modo binario: 2 estados  (BCELoss)")

    print("Construyendo modelo...")
    if args.one_hot == -1:
        args.one_hot = int(max(X_np.max(), Y_np.max())) + 1
    if args.one_hot:
        print(f"One-hot: {args.one_hot} estados → {n_size * args.one_hot} entradas")
    if args.one_hot and not (0 <= X_np.min() and X_np.max() < args.one_hot):
        raise ValueError(f"--one-hot {args.one_hot} pero los datos tienen estados entre {X_np.min():g} y {X_np.max():g}")
    model     = build_model(n_size, hidden_layers, out_size, out_act, args.one_hot).to(device)
    criterion = nn.CrossEntropyLoss() if multiclass else nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = build_scheduler(optimizer, args)
    print(f"Learning rate: {args.lr:g}" + ("" if args.lr_schedule == "none"
                                           else f" → {args.lr_min:g} ({args.lr_schedule})"))
    n_batches = (N + args.batch - 1) // args.batch

    # Los datos van a la GPU si caben en la VRAM libre (no la total) dejando OVERHEAD para el modelo y los lotes.
    # Cada lote se toma con X[perm[s:s+b]], así no se copia el arreglo completo en cada época.
    on_gpu     = False
    data_bytes = X.nbytes + Y.nbytes
    OVERHEAD   = 2 * 1024 ** 3
    if device.type == "cuda":
        free_vram, total_vram = torch.cuda.mem_get_info(device)
        if data_bytes + OVERHEAD < free_vram:
            X, Y   = X.to(device), Y.to(device)
            on_gpu = True
            print(f"Datos en GPU  ({data_bytes/1e9:.2f} GB)")
        else:
            X, Y = X.pin_memory(), Y.pin_memory()
            print(f"Datos en CPU/pin_memory  ({data_bytes/1e9:.2f} GB, "
                  f"VRAM libre: {free_vram/1e9:.2f} de {total_vram/1e9:.2f} GB)")

    csv_file = csv_writer = None
    if args.log_csv:
        csv_file   = open(args.log_csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["epoch", "loss", "lr"])

    print("Training...")
    train_start = time.perf_counter()
    try:
        for epoch in range(args.epochs):
            if on_gpu:
                perm = torch.randperm(N, device=device)
                def get_batch(s):
                    idx = perm[s:s+args.batch]
                    return X[idx], Y[idx]
            else:
                perm = torch.randperm(N)
                def get_batch(s):
                    idx = perm[s:s+args.batch]
                    return (X[idx].pin_memory().to(device, non_blocking=True),
                            Y[idx].pin_memory().to(device, non_blocking=True))

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
            lr = optimizer.param_groups[0]["lr"]   # el learning rate con el que se entrenó esta época
            if epoch % args.print_every == 0:
                print(f"epoch: {epoch:>6}  loss: {avg_loss:.6f}  lr: {lr:.2e}")
                if csv_writer:
                    csv_writer.writerow([epoch, avg_loss, lr])
                    csv_file.flush()
            if avg_loss < args.loss_target:
                print(f"Convergió en epoch {epoch}  loss: {avg_loss:.6f}")
                if csv_writer:
                    csv_writer.writerow([epoch, avg_loss, lr])
                    csv_file.flush()
                break
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            elif scheduler is not None:
                scheduler.step()

    except KeyboardInterrupt:
        print("\nInterrumpido.")
    finally:
        if csv_file:
            csv_file.close()
            print(f"Registro CSV guardado → {args.log_csv}")

    train_elapsed = time.perf_counter() - train_start
    print(f"Tiempo de entrenamiento: {train_elapsed:.2f} s "
          f"({train_elapsed/60:.2f} min)")

    save_weights(args.output, model, neighbourhood, hidden_layers, out_act, args.one_hot)


if __name__ == "__main__":
    main()
