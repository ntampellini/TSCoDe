import cProfile
import time
from pstats import Stats

from tscode.embedder import Embedder


def profiled_wrapper(filename, name):

    datafile = f"tscode_{name}_cProfile.dat"
    cProfile.run("Embedder(filename, args.name).run()", datafile)

    with open(f"tscode_{name}_cProfile_output_time.txt", "w") as f:
        p = Stats(datafile, stream=f)
        p.sort_stats("time").print_stats()

    with open(f"tscode_{name}_cProfile_output_cumtime.txt", "w") as f:
        p = Stats(datafile, stream=f)
        p.sort_stats("cumtime").print_stats()