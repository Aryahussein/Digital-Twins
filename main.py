from dc_solver import run_dc
from ac_solver import run_ac
from tran_solver import run_tran

NETLIST_FILE = "test_circuit.sp"

def detect_analysis(netlist_file):
    with open(netlist_file) as f:
        for line in f:
            line = line.strip().lower()
            if line.startswith('.op'):
                return 'dc'
            if line.startswith('.ac'):
                return 'ac'
            if line.startswith('.tran'):
                return 'tran'
    raise RuntimeError("No analysis directive (.op, .ac, or .tran) found")

if __name__ == "__main__":
    analysis = detect_analysis(NETLIST_FILE)

    if analysis == 'dc':
        run_dc(NETLIST_FILE)
    elif analysis == 'ac':
        run_ac(NETLIST_FILE)
    elif analysis == 'tran':
        run_tran(NETLIST_FILE)

print("Detected analysis:", analysis)
