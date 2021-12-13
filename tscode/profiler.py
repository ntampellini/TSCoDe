import cProfile
from pstats import Stats
from tscode.embedder import Docker
from tscode.run import RunEmbedding

def profiled_wrapper(filename, name):

    datafile = f"TSCoDe_{name}_cProfile.dat"
    cProfile.run("RunEmbedding(Embedder(filename, args.name))", datafile)

    with open(f"TSCoDe_{name}_cProfile_output_time.txt", "w") as f:
        p = Stats(datafile, stream=f)
        p.sort_stats("time").print_stats()

    with open(f"TSCoDe_{name}_cProfile_output_cumtime.txt", "w") as f:
        p = Stats(datafile, stream=f)
        p.sort_stats("cumtime").print_stats()