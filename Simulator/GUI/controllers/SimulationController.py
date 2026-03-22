# controllers.SimulationController

import random
from PyQt5.QtCore import QTimer, QThreadPool, QRunnable, QMutex, QMutexLocker, QWaitCondition
from collections import deque

from models.cellularAutomaton import cellularAutomaton

PREFETCH_SIZE = 10

class PrefetchWorker(QRunnable):
    def __init__(self, automaton, future_record, mutex, condition, needed):
        super().__init__()
        self.automaton = automaton
        self.future_record = future_record
        self.mutex = mutex
        self.condition = condition
        self.needed = needed

    def run(self):
        # Correr el thread en paralelo
        frames = self.automaton.run(self.needed) 

        locker = QMutexLocker(self.mutex)
        self.future_record.extend(frames)
        self.condition.wakeAll()


class SimulationController:
    def __init__(self, simulation_widget, config):
        self.simulation_widget = simulation_widget

        self.config = config

        self.automaton = cellularAutomaton(
            space = [[0] * simulation_widget.grid_size for _ in range(simulation_widget.grid_size)],
            states = 2,
            neighbourhood = [[-1,-1],[0,-1],[1,-1]],
            rule = [0,1,0,1,1,0,1,0] 
            )
        
        # Timer config
        self.timer = QTimer()
        self.timer .timeout.connect(self.on_step_button)
        self.is_running = False
        self.interval = 200
        self.timer.setInterval(self.interval)

        self.past_record: list = []
        self.future_record:deque = deque([])         

        # Sync
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.thread_pool = QThreadPool.globalInstance()
        self._prefetch_running = False

        # Event click
        self.simulation_widget.cell_toggled.connect(self.on_cell_toggled)

    def _clear_history(self):
        self.past_record.clear()
        self.future_record.clear()

    def _start_timer(self):
        self.is_running = True
        self.timer.start()

    def _prefetch(self):
        locker = QMutexLocker(self.mutex)
        if self._prefetch_running or len(self.future_record) >= PREFETCH_SIZE:
            return
        
        needed = PREFETCH_SIZE
        self._prefetch_running = True
        locker.unlock()

        worker = PrefetchWorker(self.automaton, self.future_record,self.mutex, self.condition, needed)
        worker.setAutoDelete(True)
        original_run = worker.run

        def run_and_reset():
            original_run()
            locker2 =   QMutexLocker(self.mutex)
            self._prefetch_running = False
        
        worker.run = run_and_reset

        self.thread_pool.start(worker)

    # Mouse events
    def on_cell_toggled(self, x, y, new_val):
        self.automaton.setSpace(self.simulation_widget.grid)
        self._clear_history()
        self.automaton.setCell(x, y, new_val)
        current = self.automaton.getSpace()
        self.simulation_widget.update_grid(current)

    # Slider events
    def on_speed_change(self, new_speed):
        self.interval = 1000 - new_speed
        self.timer.setInterval(self.interval) 

    # Buttons events
    def on_step_button(self):
        # Se añade al actual al pasado
        self.past_record.append(self.simulation_widget.grid.copy())
        
        locker = QMutexLocker(self.mutex)
        if not self.future_record:
            self.past_record.pop()
            locker.unlock()
            self._prefetch()
            return

        next_frame = self.future_record.popleft()
        locker.unlock()

        self.simulation_widget.update_grid(next_frame.copy())
        self._prefetch()

    def on_play_button(self):
        if self.is_running:
            self.timer.stop()
            self.is_running = False
        else:
            self._prefetch()
            QTimer.singleShot(100, self._start_timer)

    def on_backstep_button(self):
        if not self.past_record:
            return
        
        self.future_record.appendleft(self.simulation_widget.grid.copy())
        self.simulation_widget.update_grid(self.past_record.pop())

    def on_erase_button(self):
        new_grid = [[0] * self.simulation_widget.grid_size for _ in range(self.simulation_widget.grid_size)]
        self._clear_history()
        self.automaton.setSpace(new_grid)
        self.simulation_widget.update_grid(new_grid)
        if self.is_running:
            self.on_play_button()

    def on_random_button(self):
        states = self.automaton.getStates()
        new_grid = [
            [random.randint(0, states - 1) for _ in range(self.simulation_widget.grid_size)]
            for _ in range(self.simulation_widget.grid_size)
        ]
        self._clear_history()
        self.automaton.setSpace(new_grid)
        self.simulation_widget.update_grid(new_grid)
        if self.is_running:
            self.on_play_button()