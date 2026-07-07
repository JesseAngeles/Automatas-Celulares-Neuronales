# controllers.SimulationController

import numpy as np
from PyQt5.QtCore import QTimer, QThreadPool, QRunnable, QMutex, QMutexLocker, QWaitCondition
from collections import deque

from models.cellularAutomaton import cellularAutomaton

PREFETCH_SIZE = 20


class PrefetchWorker(QRunnable):
    def __init__(self, automaton, future_record, mutex, condition, needed, generation, get_gen):
        super().__init__()
        self.automaton     = automaton
        self.future_record = future_record
        self.mutex         = mutex
        self.condition     = condition
        self.needed        = needed
        self._generation   = generation
        self._get_gen      = get_gen

    def run(self):
        frames = self.automaton.run(self.needed)   # ndarray (N, H, W) uint8

        locker = QMutexLocker(self.mutex)
        if self._get_gen() == self._generation:
            for i in range(frames.shape[0]):
                self.future_record.append(frames[i])
        self.condition.wakeAll()


class SimulationController:
    def __init__(self, simulation_widget, config):
        self.simulation_widget = simulation_widget
        self.config = config

        n = simulation_widget.grid_size

        life_neighbourhood = [
            [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]],
            [[0,0]]
        ]
        life_rule = [0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0]

        self.automaton = cellularAutomaton(
            space         = np.zeros((n, n), dtype=np.uint8),
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
        self.future_record: deque = deque()

        # Sync
        self.mutex             = QMutex()
        self.condition         = QWaitCondition()
        self.thread_pool       = QThreadPool.globalInstance()
        self._prefetch_running = False
        self._generation       = 0

        self._on_running_changed = None   # callback(is_running: bool)
        self._on_step_changed    = None   # callback(step: int)
        self.current_step        = 0

        self._nca_mode     = False
        self._nca          = None
        self._update_prob  = 1.0             # p estocástico compartido CA/NCA
        self._nca_out_size = 1             # 1 = binario, >1 = N-estados
        self._nca_channels = 1             # canales float por celda (C)
        self._nca_state    = None          # estado completo (H, W, C) si C > 1
        self._display_channel = 0          # canal mostrado cuando C > 1
        self._continuous   = False         # NCA sin discretizar entre pasos
        self._on_nca_mode_changed = None   # callback(nca_mode: bool)
        self._on_timeline_changed = None   # callback(current: int, total: int)
        self._on_continuous_changed = None # callback(continuous: bool, available: bool)
        self._on_channels_changed = None   # callback(channels: int)

        self.simulation_widget.cell_toggled.connect(self.on_cell_toggled)

    # ------------------------------------------------------------------
    # Historial
    # ------------------------------------------------------------------
    def _clear_history(self):
        self._generation += 1
        self.past_record.clear()
        self.future_record.clear()
        self.current_step = 0
        if self._on_step_changed:
            self._on_step_changed(0)
        if self._on_timeline_changed:
            self._on_timeline_changed(0, 0)

    def _emit_step(self):
        if self._on_step_changed:
            self._on_step_changed(self.current_step)
        if self._on_timeline_changed:
            total = len(self.past_record) + len(self.future_record)
            self._on_timeline_changed(len(self.past_record), total)

    # ------------------------------------------------------------------
    # Prefetch
    # ------------------------------------------------------------------
    def _start_timer(self):
        self.is_running = True
        self.timer.start()
        if self._on_running_changed:
            self._on_running_changed(True)

    def _prefetch(self):
        if self._nca_mode:
            return
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
                ctrl.automaton.setSpace(ctrl.simulation_widget.grid)

        worker.run = run_and_reset
        self.thread_pool.start(worker)

    # ------------------------------------------------------------------
    # Eventos de mouse
    # ------------------------------------------------------------------
    def on_cell_toggled(self, x, y, new_val):
        self._clear_history()
        grid = self.simulation_widget.grid
        # El CA (tabla de regla) solo puede sincronizarse con grillas discretas;
        # en modo continuo se re-sincroniza al volver a discreto.
        if grid.dtype == np.uint8:
            self.automaton.setSpace(grid)
        if self._nca_mode and self._nca is not None:
            if self._nca_channels > 1:
                # El pintado edita el canal visible; el resto del vector se conserva
                self._nca_state[..., self._display_channel] = grid
                self._nca.set_space(self._nca_state)
            else:
                self._nca.set_space(np.ascontiguousarray(grid, dtype=np.float32))

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
        if self._nca_mode and self._nca is not None:
            if self._nca_channels > 1:
                # Multi-canal: el historial guarda el estado completo (H, W, C);
                # el widget muestra solo el canal seleccionado
                self.past_record.append(self._nca_state)
                self._nca.step()
                self._nca_state = np.ascontiguousarray(
                    self._nca.get_space(), dtype=np.float32)
                self.simulation_widget.update_grid(
                    np.ascontiguousarray(self._nca_state[..., self._display_channel]))
                self.current_step += 1
                self._emit_step()
                return
            self.past_record.append(self.simulation_widget.grid.copy())
            self._nca.step()
            raw = self._nca.get_space()                       # (H, W) float32
            if self._continuous:
                display = raw.copy()   # el device conserva el estado continuo
            elif self._nca_out_size == 1:
                display = (raw > 0.5).astype(np.uint8)
                self._nca.set_space(display.astype(np.float32))
            else:
                display = raw.astype(np.uint8)                # argmax ya en kernel
                self._nca.set_space(raw)
            self.simulation_widget.update_grid(np.ascontiguousarray(display))
            self.current_step += 1
            self._emit_step()
            return

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
            self._emit_step()
            self._prefetch()
            return

        next_frame = self.future_record.popleft()
        locker.unlock()

        # next_frame es una vista del ndarray (N, H, W) del run() anterior.
        # La copiamos para no retener el array grande completo por un solo frame.
        self.simulation_widget.update_grid(np.ascontiguousarray(next_frame))
        self.current_step += 1
        self._emit_step()
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
        if self._nca_mode and self._nca_channels > 1:
            self.future_record.appendleft(self._nca_state)
            prev = self.past_record.pop()
            self._nca_state = prev
            self._nca.set_space(prev)
            self.simulation_widget.update_grid(
                np.ascontiguousarray(prev[..., self._display_channel]))
        else:
            self.future_record.appendleft(self.simulation_widget.grid.copy())
            prev = self.past_record.pop()
            self.simulation_widget.update_grid(prev)
        self.current_step -= 1
        self._emit_step()

    def on_erase_button(self):
        n = self.simulation_widget.grid_size
        if self._nca_mode and self._nca_channels > 1:
            self._clear_history()
            self._nca_state = np.zeros((n, n, self._nca_channels), dtype=np.float32)
            self._nca.set_space(self._nca_state)
            self.simulation_widget.update_grid(
                np.zeros((n, n), dtype=np.float32))
            if self.is_running:
                self.on_play_button()
            return
        dtype = np.float32 if self._continuous else np.uint8
        new_grid = np.zeros((n, n), dtype=dtype)
        self._clear_history()
        if new_grid.dtype == np.uint8:
            self.automaton.setSpace(new_grid)
        if self._nca_mode and self._nca is not None:
            self._nca.set_space(new_grid.astype(np.float32, copy=False))
        self.simulation_widget.update_grid(new_grid)
        if self.is_running:
            self.on_play_button()

    def on_stochastic_p_change(self, p: float):
        p = float(p)
        if p == self._update_prob:
            return
        self._update_prob = p
        # Cambio de configuración: sincronizar al frame visible e invalidar prefetch
        grid = self.simulation_widget.grid
        if grid.dtype == np.uint8:
            self.automaton.setSpace(grid)
        self._clear_history()
        self.automaton.setUpdateProb(p)
        if self._nca is not None:
            self._nca.set_update_prob(p)

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
        rule = [(n // (states ** i)) % states for i in range(r_size)]
        self.automaton.setRule(rule)
        self._clear_history()

    def on_states_change(self, n_states: int, colors):
        grid = self.simulation_widget.grid
        # Clampear estados fuera de rango → 0 (vectorizado)
        grid = np.where(grid >= n_states, 0, grid).astype(np.uint8, copy=False)
        self._clear_history()
        self.automaton.setStates(n_states)
        self.automaton.setSpace(grid)
        self.simulation_widget.update_grid(grid)

    def on_record_button(self):
        from PyQt5.QtWidgets import QInputDialog
        n, ok = QInputDialog.getInt(None, "Record", "Number of steps:", 100, 1, 100000)
        if not ok:
            return
        if self.is_running:
            self.on_play_button()

        if self._nca_mode and self._nca is not None:
            if self._nca_channels > 1:
                # Multi-canal: run() batcheado, historial de estados completos
                self._nca.set_space(self._nca_state)
                self._clear_history()
                self.past_record.append(self._nca_state)
                frames = self._nca.run(n)                # (n, H, W, C) float32
                for i in range(n - 1):
                    self.past_record.append(np.ascontiguousarray(frames[i]))
                self._nca_state = np.ascontiguousarray(frames[n - 1])
                self.simulation_widget.update_grid(
                    np.ascontiguousarray(self._nca_state[..., self._display_channel]))
                self.current_step = n
                self._emit_step()
                return

            self._nca.set_space(
                np.ascontiguousarray(self.simulation_widget.grid, dtype=np.float32))
            self._clear_history()
            self.past_record.append(self.simulation_widget.grid.copy())

            if self._continuous:
                # Sin re-inyección discreta entre pasos: run() batcheado en GPU
                frames = self._nca.run(n)                # (n, H, W) float32
                for i in range(n - 1):
                    self.past_record.append(np.ascontiguousarray(frames[i]))
                self.simulation_widget.update_grid(np.ascontiguousarray(frames[n - 1]))
                self.current_step = n
                self._emit_step()
                return

            display = self.simulation_widget.grid
            for i in range(n):
                self._nca.step()
                raw = self._nca.get_space()              # (H, W) float32
                if self._nca_out_size == 1:
                    display = (raw > 0.5).astype(np.uint8)
                    self._nca.set_space(display.astype(np.float32))
                else:
                    display = raw.astype(np.uint8)
                    self._nca.set_space(raw)
                if i < n - 1:
                    self.past_record.append(np.ascontiguousarray(display))
            self.simulation_widget.update_grid(np.ascontiguousarray(display))
            self.current_step = n
            self._emit_step()
            return

        self.automaton.setSpace(self.simulation_widget.grid)
        self._clear_history()
        frames = self.automaton.run(n)   # ndarray (n, H, W) uint8
        self.past_record.append(self.simulation_widget.grid.copy())
        for i in range(n - 1):
            self.past_record.append(np.ascontiguousarray(frames[i]))
        self.simulation_widget.update_grid(np.ascontiguousarray(frames[n - 1]))
        self.current_step = n
        self._emit_step()

    def on_export_button(self):
        import scipy.io
        from PyQt5.QtWidgets import QFileDialog
        if not self.past_record:
            return
        path, _ = QFileDialog.getSaveFileName(None, "Export evolution", "", "Evolution files (*)")
        if not path:
            return
        last = (self._nca_state
                if self._nca_mode and self._nca_channels > 1
                else self.simulation_widget.grid)
        frames = self.past_record + [last]
        arr = np.stack(frames)  # (T, H, W) o (T, H, W, C)
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32, copy=False)  # continuo: sin truncar
        else:
            arr = arr.astype(np.int32, copy=False)
        base = path.removesuffix(".npy").removesuffix(".mat")
        np.save(base + ".npy", arr)
        scipy.io.savemat(base + ".mat", {"evolution": arr})

    def on_import_button(self):
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        path, _ = QFileDialog.getOpenFileName(None, "Import evolution", "", "NumPy (*.npy)")
        if not path:
            return
        arr = np.load(path)
        if arr.ndim != 3:
            return
        if self.is_running:
            self.on_play_button()

        if np.issubdtype(arr.dtype, np.floating):
            # Evolución continua: solo reproducible en modo NCA continuo
            if not (self._nca_mode and self._continuous):
                QMessageBox.information(
                    None, "Evolución continua",
                    "El archivo contiene estados continuos (float).\n"
                    "Carga pesos NCA y activa Output: Continuous para importarlo."
                )
                return
            arr = arr.astype(np.float32, copy=False)
        else:
            arr = arr.astype(np.uint8, copy=False)

        n = self.simulation_widget.grid_size
        fH, fW = arr.shape[1], arr.shape[2]
        if fH != n or fW != n:
            QMessageBox.information(
                None, "Tamaño ajustado",
                f"La evolución importada es {fW}×{fH} y la grilla actual es {n}×{n}.\n"
                f"{'Se recortará' if fH > n or fW > n else 'Se centrará'} en el espacio disponible."
            )
            fitted = np.zeros((arr.shape[0], n, n), dtype=arr.dtype)
            src_h = min(fH, n)
            src_w = min(fW, n)
            off_y = (n - src_h) // 2
            off_x = (n - src_w) // 2
            fitted[:, off_y:off_y + src_h, off_x:off_x + src_w] = arr[:, :src_h, :src_w]
            arr = fitted

        self._clear_history()
        for i in range(arr.shape[0] - 1):
            self.past_record.append(np.ascontiguousarray(arr[i]))
        last = np.ascontiguousarray(arr[-1])
        if last.dtype == np.uint8:
            self.automaton.setSpace(last)
        if self._nca_mode and self._nca is not None:
            self._nca.set_space(np.ascontiguousarray(last, dtype=np.float32))
        self.simulation_widget.update_grid(last)
        self.current_step = arr.shape[0] - 1
        self._emit_step()

    def on_random_button(self):
        n = self.simulation_widget.grid_size
        if self._nca_mode and self._nca_channels > 1:
            self._clear_history()
            self._nca_state = np.random.random(
                (n, n, self._nca_channels)).astype(np.float32)
            self._nca.set_space(self._nca_state)
            self.simulation_widget.update_grid(
                np.ascontiguousarray(self._nca_state[..., self._display_channel]))
            if self.is_running:
                self.on_play_button()
            return
        if self._continuous:
            new_grid = np.random.random((n, n)).astype(np.float32)
        else:
            states = self.automaton.getStates()
            new_grid = np.random.randint(0, states, size=(n, n), dtype=np.uint8)
        self._clear_history()
        if new_grid.dtype == np.uint8:
            self.automaton.setSpace(new_grid)
        if self._nca_mode and self._nca is not None:
            self._nca.set_space(new_grid.astype(np.float32, copy=False))
        self.simulation_widget.update_grid(new_grid)
        if self.is_running:
            self.on_play_button()

    def on_load_nca_button(self):
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        from CA import NCA as NCA_CUDA

        path, _ = QFileDialog.getOpenFileName(None, "Load NCA weights", "", "NumPy (*.npz)")
        if not path:
            return
        if self.is_running:
            self.on_play_button()

        try:
            data = np.load(path)
            n_layers     = int(data["layer_in"].shape[0])
            neighbourhood = data["neighbourhood"].astype(np.int32)  # (n_size, 2)
            n_size        = len(neighbourhood)
            channels      = int(data["channels"]) if "channels" in data.files else 1

            n = self.simulation_widget.grid_size
            grid_float = self.simulation_widget.grid.astype(np.float32)

            if channels > 1:
                # Estado inicial: canal 0 = grilla visible, resto en 0
                state = np.zeros((n, n, channels), dtype=np.float32)
                state[..., 0] = grid_float
                self._nca = NCA_CUDA(n, n, state, neighbourhood, n_size, channels)
                self._nca_state = state
            else:
                self._nca = NCA_CUDA(n, n, grid_float, neighbourhood, n_size)
                self._nca_state = None
            self._nca_channels    = channels
            self._display_channel = 0
            self._nca.set_weights(
                [data[f"weights_{i}"] for i in range(n_layers)],
                [data[f"biases_{i}"]  for i in range(n_layers)],
                data["layer_in"].tolist(),
                data["layer_out"].tolist(),
                data["activations"].tolist(),
            )
            self._nca_out_size = int(data["layer_out"][-1])
            if self._update_prob < 1.0:
                self._nca.set_update_prob(self._update_prob)

            self._nca_mode = True
            if channels > 1:
                # Multi-canal es continuo por construcción: display float (canal 0)
                self._continuous = False
                self.simulation_widget.update_grid(
                    np.ascontiguousarray(self._nca_state[..., 0]))
            elif self._continuous:
                # Cargar pesos vuelve al modo discreto; si la grilla venía continua,
                # se discretiza para que display y modo queden consistentes
                self._continuous = False
                disc = (np.asarray(self.simulation_widget.grid) > 0.5).astype(np.uint8)
                self.automaton.setSpace(disc)
                self._nca.set_space(disc.astype(np.float32))
                self.simulation_widget.update_grid(disc)
            self._clear_history()
            if self._on_nca_mode_changed:
                self._on_nca_mode_changed(self._nca_mode, nca_loaded=True)
            if self._on_continuous_changed:
                # Continuo solo para salida escalar: out_size > 1 ya es argmax discreto
                self._on_continuous_changed(False, self._nca_out_size == 1 and channels == 1)
            if self._on_channels_changed:
                self._on_channels_changed(channels)

        except Exception as e:
            QMessageBox.critical(None, "Error loading NCA", str(e))

    def on_timeline_jump(self, pos: int):
        locker = QMutexLocker(self.mutex)
        current_pos = len(self.past_record)
        if pos == current_pos:
            return
        current_frame = (self._nca_state
                         if self._nca_mode and self._nca_channels > 1
                         else self.simulation_widget.grid.copy())
        full = self.past_record + [current_frame] + list(self.future_record)
        if pos < 0 or pos >= len(full):
            return
        delta = pos - current_pos
        self.past_record = list(full[:pos])
        self.future_record = deque(full[pos + 1:])
        self._generation += 1
        locker.unlock()

        self.current_step += delta
        target = np.ascontiguousarray(full[pos])
        if target.ndim == 3:
            # Frame multi-canal (H, W, C): restaurar estado completo del NCA
            self._nca_state = target
            if self._nca is not None:
                self._nca.set_space(target)
            self.simulation_widget.update_grid(
                np.ascontiguousarray(target[..., self._display_channel]))
            self._emit_step()
            return
        if target.dtype == np.uint8:
            self.automaton.setSpace(target)
        if self._nca_mode and self._nca is not None and self._nca_channels == 1:
            self._nca.set_space(target.astype(np.float32, copy=False))
        self.simulation_widget.update_grid(target)
        self._emit_step()

    def on_mode_toggle(self):
        if self._nca is None:
            return
        if self.is_running:
            self.on_play_button()
        # Salir del modo NCA fuerza salida discreta (el CA no acepta floats)
        if self._nca_mode and (self._continuous or self._nca_channels > 1):
            self._continuous = False
            disc = (np.asarray(self.simulation_widget.grid) > 0.5).astype(np.uint8)
            self.simulation_widget.update_grid(disc)
        self._nca_mode = not self._nca_mode
        grid = self.simulation_widget.grid
        self.automaton.setSpace(grid)
        if self._nca_channels > 1:
            # Re-sembrar el NCA multi-canal: canal 0 = grilla visible, resto en 0
            state = np.zeros((*grid.shape, self._nca_channels), dtype=np.float32)
            state[..., 0] = grid
            self._nca_state = state
            self._nca.set_space(state)
            if self._nca_mode:
                self.simulation_widget.update_grid(
                    np.ascontiguousarray(state[..., self._display_channel]))
        else:
            self._nca.set_space(grid.astype(np.float32))
        self._clear_history()
        if self._on_nca_mode_changed:
            self._on_nca_mode_changed(self._nca_mode, nca_loaded=True)
        if self._on_continuous_changed:
            self._on_continuous_changed(
                self._continuous, self._nca_mode and self._nca_out_size == 1)

    def on_display_channel_change(self, channel: int):
        if self._nca_channels <= 1:
            return
        self._display_channel = max(0, min(int(channel), self._nca_channels - 1))
        if self._nca_mode and self._nca_state is not None:
            self.simulation_widget.update_grid(
                np.ascontiguousarray(self._nca_state[..., self._display_channel]))

    def on_continuous_toggle(self):
        if self._nca is None or self._nca_out_size != 1 or not self._nca_mode:
            return
        if self.is_running:
            self.on_play_button()
        self._continuous = not self._continuous
        grid = self.simulation_widget.grid
        if self._continuous:
            new_grid = np.ascontiguousarray(grid, dtype=np.float32)
        else:
            new_grid = (np.asarray(grid) > 0.5).astype(np.uint8)
            self.automaton.setSpace(new_grid)
        self._nca.set_space(new_grid.astype(np.float32, copy=False))
        self._clear_history()
        self.simulation_widget.update_grid(new_grid)
        if self._on_continuous_changed:
            self._on_continuous_changed(self._continuous, True)

