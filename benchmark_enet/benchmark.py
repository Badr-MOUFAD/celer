from benchopt.benchmark import Benchmark
from benchopt.runner import run_benchmark


enet_benchmark = Benchmark('./benchmark_enet')

run_benchmark(enet_benchmark, plot_result=True)
