# controllers.SimulationController

import random
from PyQt5.QtCore import QTimer, QThreadPool, QRunnable, QMutex, QMutexLocker, QWaitCondition
from collections import deque

from models.cellularAutomaton import cellularAutomaton

PREFETCH_SIZE = 10

class PrefetchWorker(QRunnable):
    def __init__(self, automaton, future_record, mutex, condition, needed, generation, get_gen):
        super().__init__()
        self.automaton     = automaton
        self.future_record = future_record
        self.mutex         = mutex
        self.condition     = condition
        self.needed        = needed
        self._generation   = generation
        self._get_gen      = get_gen   # lambda que devuelve la generación actual

    def run(self):
        frames = self.automaton.run(self.needed)

        locker = QMutexLocker(self.mutex)
        # Solo escribir si la generación sigue vigente (sin cambios de config)
        if self._get_gen() == self._generation:
            self.future_record.extend(frames)
        self.condition.wakeAll()


class SimulationController:
    def __init__(self, simulation_widget, config):
        self.simulation_widget = simulation_widget
        self.config = config

        life_neighbourhood = [
            [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]],
            [[0,0]]
        ]
        # Índice = sum_vecinos*2 + celda_central
        # Nacer: (0,3)→idx6=1  Sobrevivir: (1,2)→idx5=1, (1,3)→idx7=1
        life_rule = [0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0]

        self.automaton = cellularAutomaton(
            space         = [[0] * simulation_widget.grid_size for _ in range(simulation_widget.grid_size)],
            states        = 2,
            neighbourhood = life_neighbourhood,
            rule          = life_rule
        )

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_step_button)
        self.is_running = False
        self.interval   = 200
        self.timer.setInterval(self.interval)

        self.past_record:   list  = []
        self.future_record: deque = deque([])

        # Sync
        self.mutex            = QMutex()
        self.condition        = QWaitCondition()
        self.thread_pool      = QThreadPool.globalInstance()
        self._prefetch_running = False
        self._generation       = 0     # se incrementa en cada cambio de config

        self._on_running_changed = None  # callback(is_running: bool)
        self._on_step_changed = None     # callback(step: int)
        self.current_step = 0

        self.simulation_widget.cell_toggled.connect(self.on_cell_toggled)

    # ------------------------------------------------------------------
    # Historial
    # ------------------------------------------------------------------
    def _clear_history(self):
        self._generation += 1          # invalida cualquier worker en curso
        self.past_record.clear()
        self.future_record.clear()
        self.current_step = 0
        if self._on_step_changed:
            self._on_step_changed(0)

    # ------------------------------------------------------------------
    # Prefetch
    # ------------------------------------------------------------------
    def _start_timer(self):
        self.is_running = True
        self.timer.start()
        if self._on_running_changed:
            self._on_running_changed(True)

    def _prefetch(self):
        locker = QMutexLocker(self.mutex)
        if self._prefetch_running or len(self.future_record) >= PREFETCH_SIZE:
            return

        gen    = self._generation
        needed = PREFETCH_SIZE
        self._prefetch_running = True
        locker.unlock()

        worker = PrefetchWorker(
            self.automaton, self.future_record,
            self.mutex, self.condition, needed,
            gen, lambda: self._generation
        )
        worker.setAutoDelete(True)
        original_run = worker.run
        ctrl = self

        def run_and_reset():
            original_run()
            locker2 = QMutexLocker(ctrl.mutex)
            discarded = (ctrl._generation != gen)
            ctrl._prefetch_running = False
            locker2.unlock()
            if discarded:
                # El autómata avanzó pero los frames se descartaron;
                # re-sincronizar al estado que está en pantalla
                ctrl.automaton.setSpace(ctrl.simulation_widget.grid)

        worker.run = run_and_reset
        self.thread_pool.start(worker)

    # ------------------------------------------------------------------
    # Eventos de mouse
    # ------------------------------------------------------------------
    def on_cell_toggled(self, x, y, new_val):
        self.automaton.setSpace(self.simulation_widget.grid)
        self._clear_history()
        self.automaton.setCell(x, y, new_val)
        self.simulation_widget.update_grid(self.automaton.getSpace())

    # ------------------------------------------------------------------
    # Eventos de slider
    # ------------------------------------------------------------------
    def on_speed_change(self, new_speed):
        self.interval = 1000 - new_speed
        self.timer.setInterval(self.interval)

    # ------------------------------------------------------------------
    # Eventos de botones
    # ------------------------------------------------------------------
    def on_step_button(self):
        self.past_record.append(self.simulation_widget.grid.copy())

        locker = QMutexLocker(self.mutex)
        if not self.future_record:
            if self._prefetch_running:
                self.past_record.pop()
                locker.unlock()
                return
            locker.unlock()
            self.automaton.step()
            self.simulation_widget.update_grid(self.automaton.getSpace().copy())
            self.current_step += 1
            if self._on_step_changed:
                self._on_step_changed(self.current_step)
            self._prefetch()
            return

        next_frame = self.future_record.popleft()
        locker.unlock()

        self.simulation_widget.update_grid(next_frame.copy())
        self.current_step += 1
        if self._on_step_changed:
            self._on_step_changed(self.current_step)
        self._prefetch()

    def on_play_button(self):
        if self.is_running:
            self.timer.stop()
            self.is_running = False
            if self._on_running_changed:
                self._on_running_changed(False)
        else:
            self._prefetch()
            QTimer.singleShot(100, self._start_timer)

    def on_backstep_button(self):
        if not self.past_record:
            return
        self.future_record.appendleft(self.simulation_widget.grid.copy())
        self.simulation_widget.update_grid(self.past_record.pop())
        self.current_step -= 1
        if self._on_step_changed:
            self._on_step_changed(self.current_step)

    def on_erase_button(self):
        new_grid = [[0] * self.simulation_widget.grid_size for _ in range(self.simulation_widget.grid_size)]
        self._clear_history()
        self.automaton.setSpace(new_grid)
        self.simulation_widget.update_grid(new_grid)
        if self.is_running:
            self.on_play_button()

    def on_boundary_change(self, mode: int):
        self.automaton.setSpace(self.simulation_widget.grid)
        self._clear_history()
        self.automaton.setBoundary(mode)

    def on_neighbourhood_change(self, neighbourhood):
        # Sincronizar el autómata al frame visible ANTES de limpiar el historial,
        # para que el próximo step parta del estado en pantalla y no de uno prefetched
        self.automaton.setSpace(self.simulation_widget.grid)
        self._clear_history()
        self.automaton.setNeighbourhood(neighbourhood)

    def on_rule_array_change(self, rule_array: list):
        self.automaton.setRule(rule_array)
        self._clear_history()

    def on_rule_change(self, n: int):
        states = self.automaton.getStates()
        r_size = self.automaton.ca.get_r_size()
        # Decodificación en base-k: dígito i = (n // states^i) % states
        rule = [(n // (states ** i)) % states for i in range(r_size)]
        self.automaton.setRule(rule)
        self._clear_history()

    def on_states_change(self, n_states: int, colors):
        # Clampear celdas con estado fuera del nuevo rango → 0
        grid = self.simulation_widget.grid
        for row in grid:
            for i, v in enumerate(row):
                if v >= n_states:
                    row[i] = 0
        self._clear_history()
        self.automaton.setStates(n_states)
        self.automaton.setSpace(grid)
        self.simulation_widget.update_grid(grid)

    def on_record_button(self):
        from PyQt5.QtWidgets import QInputDialog
        n, ok = QInputDialog.getInt(None, "Record", "Number of steps:", 100, 1, 10000)
        if not ok:
            return
        if self.is_running:
            self.on_play_button()
        self.automaton.setSpace(self.simulation_widget.grid)
        self._clear_history()
        for _ in range(n):
            self.past_record.append(self.automaton.getSpace().copy())
            self.automaton.step()
        self.current_step = n
        if self._on_step_changed:
            self._on_step_changed(n)
        self.simulation_widget.update_grid(self.automaton.getSpace().copy())

    def on_export_button(self):
        import numpy as np
        import scipy.io
        from PyQt5.QtWidgets import QFileDialog
        if not self.past_record:
            return
        path, _ = QFileDialog.getSaveFileName(None, "Export evolution", "", "Evolution files (*)")
        if not path:
            return
        frames = self.past_record + [self.simulation_widget.grid]
        arr = np.array(frames, dtype=np.int32)
        base = path.removesuffix(".npy").removesuffix(".mat")
        np.save(base + ".npy", arr)
        scipy.io.savemat(base + ".mat", {"evolution": arr})

    def on_import_button(self):
        import numpy as np
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(None, "Import evolution", "", "NumPy (*.npy)")
        if not path:
            return
        arr = np.load(path)
        if arr.ndim != 3:
            return
        if self.is_running:
            self.on_play_button()
        self._clear_history()
        frames = arr.tolist()
        self.past_record.extend(frames[:-1])
        last = frames[-1]
        self.automaton.setSpace(last)
        self.simulation_widget.update_grid(last)
        self.current_step = len(frames) - 1
        if self._on_step_changed:
            self._on_step_changed(self.current_step)

    def on_random_button(self):
        states   = self.automaton.getStates()
        new_grid = [
            [random.randint(0, states - 1) for _ in range(self.simulation_widget.grid_size)]
            for _ in range(self.simulation_widget.grid_size)
        ]
        self._clear_history()
        self.automaton.setSpace(new_grid)
        self.simulation_widget.update_grid(new_grid)
        if self.is_running:
            self.on_play_button()
