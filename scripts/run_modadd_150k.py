"""Run full modular addition grokking pipeline: 150K step training + Lanczos Hessian + visualization."""
import sys
sys.path.insert(0, ".")
from src.run_modadd import run

run(n_steps=150000, lanczos_k=50)
