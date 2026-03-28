"""GUI Launcher Script."""

import tkinter as tk
from main import run_simulation_core
from utils.gui import CircuitSimulatorGUI

if __name__ == "__main__":
    root = tk.Tk()
    app = CircuitSimulatorGUI(root, run_simulation_core)
    root.mainloop()
