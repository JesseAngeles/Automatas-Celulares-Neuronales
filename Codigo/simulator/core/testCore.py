import sys
import os

# Añade la ruta del build donde está el .so
sys.path.append(os.path.join(os.path.dirname(__file__), "build"))

import CASimulatorCore

sim = CASimulatorCore.CASimulator(10)
sim.step()
print(sim.get_grid())