"""
Búsqueda de hiperparámetros de entrenamiento del NCA con un algoritmo genético.

Basado en la arquitectura de:
https://github.com/JesseAngeles/Metaheuristicas/blob/master/Algorithms/GeneticAlgorithm.py
(Problem + GeneticAlgorithm con operadores intercambiables de
 selección / cruza / mutación / reemplazo).

Fijo:    vecindad 3x3, out-act sigmoid, epochs 100, loss-target 1e-04
Variable (genoma): 8 ranuras de capa oculta (neuronas 0 = capa ignorada,
                   activación por capa), batch size, learning rate y
                   tamaño de la capa de salida (1 o 2).
Fitness: -loss final del entrenamiento (maximizar).
"""

import argparse
import csv
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from functools import partial

# ── Espacio de búsqueda ───────────────────────────────────────────────────────

NEURONS_OPTS  = [0, 4, 8, 16, 32, 128]      # 0 → la capa se ignora
ACT_OPTS      = ['relu', 'sigmoid', 'tanh', 'linear']
BATCH_OPTS    = [2048, 4096, 8192, 16384, 32768]
LR_OPTS       = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
OUT_SIZE_OPTS = [1, 2]

MAX_LAYERS = 8

# Genoma (lista de índices, longitud fija 2*MAX_LAYERS + 3):
#   [0 .. MAX_LAYERS-1]              → índice en NEURONS_OPTS por capa (0 = sin capa)
#   [MAX_LAYERS .. 2*MAX_LAYERS-1]   → índice en ACT_OPTS por capa
#   [-3]                             → índice en BATCH_OPTS
#   [-2]                             → índice en LR_OPTS
#   [-1]                             → índice en OUT_SIZE_OPTS
GENOME_LEN = 2 * MAX_LAYERS + 3

NEIGHBOURHOOD = "-1,-1 0,-1 1,-1 -1,0 0,0 1,0 -1,1 0,1 1,1"

LOSS_RE = re.compile(r"loss:\s*([0-9.eE+-]+)")


# ── Problema ──────────────────────────────────────────────────────────────────

class TrainingConfigProblem:
    """Define el genoma, su decodificación y la función objetivo (fitness)."""

    def __init__(self, python_bin, train_script, data_path, out_dir,
                 epochs=100, loss_target=1e-5, seed=42, log_path=None):
        self.python_bin   = python_bin
        self.train_script = train_script
        self.data_path    = data_path
        self.out_dir      = out_dir
        self.epochs       = epochs
        self.loss_target  = loss_target
        self.seed         = seed
        self.log_path     = log_path
        self.cache        = {}          # fenotipo → fitness
        self.evaluations  = 0
        self.best         = None        # (fitness, cfg) mejor histórico

    # -- Genoma ---------------------------------------------------------------

    def generateInitialSolution(self):
        genome  = [random.randrange(len(NEURONS_OPTS)) for _ in range(MAX_LAYERS)]
        genome += [random.randrange(len(ACT_OPTS)) for _ in range(MAX_LAYERS)]
        genome.append(random.randrange(len(BATCH_OPTS)))
        genome.append(random.randrange(len(LR_OPTS)))
        genome.append(random.randrange(len(OUT_SIZE_OPTS)))
        return genome

    def getRandomNeighbour(self, genome):
        """Mutación de un punto: cambia un gen a otro valor válido."""
        neighbour = list(genome)
        idx = random.randrange(GENOME_LEN)
        if idx < MAX_LAYERS:
            options = len(NEURONS_OPTS)
        elif idx < 2 * MAX_LAYERS:
            options = len(ACT_OPTS)
        elif idx == GENOME_LEN - 3:
            options = len(BATCH_OPTS)
        elif idx == GENOME_LEN - 2:
            options = len(LR_OPTS)
        else:
            options = len(OUT_SIZE_OPTS)
        new_val = random.randrange(options)
        while options > 1 and new_val == neighbour[idx]:
            new_val = random.randrange(options)
        neighbour[idx] = new_val
        return neighbour

    # -- Decodificación -------------------------------------------------------

    def decode(self, genome):
        layers = []
        for i in range(MAX_LAYERS):
            neurons = NEURONS_OPTS[genome[i]]
            if neurons == 0:            # capa presente en el vector pero ignorada
                continue
            act = ACT_OPTS[genome[MAX_LAYERS + i]]
            layers.append(f"{neurons}:{act}")
        batch    = BATCH_OPTS[genome[-3]]
        lr       = LR_OPTS[genome[-2]]
        out_size = OUT_SIZE_OPTS[genome[-1]]
        return {"layers": layers, "batch": batch, "lr": lr, "out_size": out_size}

    def phenotypeKey(self, genome):
        cfg = self.decode(genome)
        return (tuple(cfg["layers"]), cfg["batch"], cfg["lr"], cfg["out_size"])

    # -- Objetivo -------------------------------------------------------------

    def objective(self, genome):
        key = self.phenotypeKey(genome)
        if key in self.cache:
            return self.cache[key]

        cfg = self.decode(genome)

        if not cfg["layers"]:
            # todas las capas en 0: main.py no puede expresar una red sin ocultas
            print("  [inválido] genoma sin capas ocultas", flush=True)
            self.cache[key] = -float("inf")
            return -float("inf")

        out_path = os.path.join(self.out_dir, f"ga_eval_{self.evaluations}.npz")
        cmd = [
            self.python_bin, "-u", self.train_script,
            "--data", self.data_path,
            "--output", out_path,
            "--neighbourhood", NEIGHBOURHOOD,
            "--layers", " ".join(cfg["layers"]),
            "--out-size", str(cfg["out_size"]),
            "--out-act", "sigmoid",
            "--epochs", str(self.epochs),
            "--batch", str(cfg["batch"]),
            "--lr", str(cfg["lr"]),
            "--loss-target", str(self.loss_target),
            "--print-every", "1",
            "--seed", str(self.seed),
        ]

        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0

        losses = LOSS_RE.findall(result.stdout)
        if result.returncode != 0 or not losses:
            print(f"  [FALLO] {cfg} → returncode={result.returncode}", flush=True)
            print(result.stderr[-500:], flush=True)
            fitness = -float("inf")
            final_loss = float("nan")
        else:
            final_loss = float(losses[-1])
            fitness = -final_loss

        self.cache[key] = fitness
        self.evaluations += 1
        if self.best is None or fitness > self.best[0]:
            self.best = (fitness, cfg)

        arch = " ".join(cfg["layers"])
        print(f"  eval {self.evaluations:>3}  loss={final_loss:.6f}  "
              f"[{arch} | out={cfg['out_size']} batch={cfg['batch']} lr={cfg['lr']}]  "
              f"({elapsed:.1f}s)", flush=True)

        if self.log_path:
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [self.evaluations, arch, cfg["out_size"], cfg["batch"],
                     cfg["lr"], f"{final_loss:.8f}", f"{elapsed:.1f}"])
        return fitness


# ── Operadores genéticos ──────────────────────────────────────────────────────

class SelectionFunctions:
    @staticmethod
    def tournament(population, objective, selection_rate=0.2):
        population_size = len(population)
        parents = []
        for _ in range(population_size):
            k = max(2, math.ceil(population_size * selection_rate))
            candidates = random.sample(population, min(k, population_size))
            scores = [objective(ind) for ind in candidates]
            best = candidates[scores.index(max(scores))]
            parents.append(best)
        return parents


class CrossoverFunctions:
    @staticmethod
    def onePoint(population):
        generation = []
        for i in range(0, len(population) - 1, 2):
            parent1, parent2 = population[i], population[i + 1]
            point = random.randint(1, len(parent1) - 1)
            generation.append(parent1[:point] + parent2[point:])
            generation.append(parent2[:point] + parent1[point:])
        return generation

    @staticmethod
    def uniform(population, crossover_rate=0.5):
        generation = []
        for i in range(0, len(population) - 1, 2):
            parent1, parent2 = population[i], population[i + 1]
            child1, child2 = [], []
            for a, b in zip(parent1, parent2):
                if random.random() < crossover_rate:
                    child1.append(a); child2.append(b)
                else:
                    child1.append(b); child2.append(a)
            generation.extend([child1, child2])
        return generation


class MutationFunctions:
    @staticmethod
    def singlePoint(population, problem, mutation_rate=0.2):
        mutations = []
        for individual in population:
            if random.random() <= mutation_rate:
                mutations.append(problem.getRandomNeighbour(individual))
            else:
                mutations.append(individual)
        return mutations


class ReplaceFunctions:
    @staticmethod
    def elitism(population, replace, objective):
        population_size = len(population)
        sorted_population = sorted(population, key=objective, reverse=True)
        survivors = sorted_population[:population_size - len(replace)]
        survivors += replace
        return survivors


# ── Algoritmo genético ────────────────────────────────────────────────────────

class GeneticAlgorithm:
    def __init__(self, problem, population_size=8):
        self.problem = problem
        self.population_size = population_size
        self.resetProblem()

    def resetProblem(self):
        self.population = [self.problem.generateInitialSolution()
                           for _ in range(self.population_size)]
        self.selection = SelectionFunctions.tournament
        self.crossover = CrossoverFunctions.onePoint
        self.mutation  = MutationFunctions.singlePoint
        self.replace   = ReplaceFunctions.elitism

    def optimize(self, epochs=1, stationary=0.0):
        for gen in range(1, epochs + 1):
            print(f"\n─── Generación {gen}/{epochs} ───", flush=True)

            sample_size = self.population_size - int(self.population_size * stationary)
            sample_size -= sample_size % 2
            population_sample = random.sample(self.population, sample_size)

            population_sample = self.selection(population_sample, self.problem.objective)
            population_sample = self.crossover(population_sample)
            population_sample = self.mutation(population_sample, self.problem)
            self.population   = self.replace(self.population, population_sample,
                                             self.problem.objective)

            best = self.bestIndividual(self.population)
            cfg  = self.problem.decode(best)
            print(f"Mejor hasta ahora: loss={-self.problem.objective(best):.6f}  "
                  f"{' '.join(cfg['layers'])} | out={cfg['out_size']} "
                  f"batch={cfg['batch']} lr={cfg['lr']}",
                  flush=True)

    def bestIndividual(self, population):
        return max(population, key=lambda ind: self.problem.objective(ind))

    # SET functions (mismo patrón que el repo)
    def setSelection(self, selection, selection_rate=None):
        self.selection = partial(selection, selection_rate=selection_rate) if selection_rate else selection

    def setCrossover(self, crossover, crossover_rate=None):
        self.crossover = partial(crossover, crossover_rate=crossover_rate) if crossover_rate else crossover

    def setMutation(self, mutation, mutation_rate=None):
        self.mutation = partial(mutation, mutation_rate=mutation_rate) if mutation_rate else mutation

    def setReplace(self, replace):
        self.replace = replace


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="GA para hiperparámetros del NCA")
    p.add_argument("--data",        default=os.path.join(base, "New Folder", "B3_S23.npy"))
    p.add_argument("--python",      default=os.path.join(base, "..", "GUI", "venv", "bin", "python"))
    p.add_argument("--out-dir",     default=os.path.join(base, "ga_runs"))
    p.add_argument("--population",  type=int,   default=8)
    p.add_argument("--generations", type=int,   default=10)
    p.add_argument("--mutation-rate", type=float, default=0.3)
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--loss-target", type=float, default=1e-4)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--ga-seed",     type=int,   default=None)
    args = p.parse_args()

    if args.ga_seed is not None:
        random.seed(args.ga_seed)

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "ga_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["eval", "layers", "out_size", "batch", "lr",
                                "final_loss", "seconds"])

    problem = TrainingConfigProblem(
        python_bin=os.path.abspath(args.python),
        train_script=os.path.join(base, "main.py"),
        data_path=os.path.abspath(args.data),
        out_dir=args.out_dir,
        epochs=args.epochs,
        loss_target=args.loss_target,
        seed=args.seed,
        log_path=log_path,
    )

    ga = GeneticAlgorithm(problem, population_size=args.population)
    ga.setMutation(MutationFunctions.singlePoint, mutation_rate=args.mutation_rate)

    print("=== Configuration ===")
    print(f"Selection: tournament   Crossover: onePoint")
    print(f"Mutation: singlePoint (rate={args.mutation_rate})   Replace: elitism")
    print(f"Population: {args.population}   Generations: {args.generations}")
    print(f"Epochs por evaluación: {args.epochs}   loss-target: {args.loss_target}\n")

    t0 = time.time()
    print("─── Evaluando población inicial ───", flush=True)
    for ind in ga.population:
        problem.objective(ind)

    ga.optimize(epochs=args.generations)

    # Mejor histórico (puede haberse perdido de la población por el reemplazo)
    best_fitness, cfg = problem.best
    best_loss = -best_fitness

    result = {
        "layers":   cfg["layers"],
        "out_size": cfg["out_size"],
        "batch":    cfg["batch"],
        "lr":       cfg["lr"],
        "final_loss": best_loss,
        "epochs_per_eval": args.epochs,
        "total_evaluations": problem.evaluations,
        "total_seconds": round(time.time() - t0, 1),
    }
    best_path = os.path.join(args.out_dir, "best_config.json")
    with open(best_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n══════════ MEJOR CONFIGURACIÓN ══════════")
    print(f"  --layers {' '.join(cfg['layers'])}")
    print(f"  --out-size {cfg['out_size']}")
    print(f"  --batch {cfg['batch']}")
    print(f"  --lr {cfg['lr']}")
    print(f"  loss final ({args.epochs} épocas): {best_loss:.6f}")
    print(f"  evaluaciones: {problem.evaluations}  "
          f"tiempo total: {result['total_seconds']}s")
    print(f"  guardado en: {best_path}")


if __name__ == "__main__":
    main()
