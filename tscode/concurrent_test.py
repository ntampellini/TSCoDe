from concurrent.futures import ProcessPoolExecutor
from tscode.utils import read_xyz, time_to_string
from tscode.calculators._xtb import xtb_opt
from time import perf_counter
import sys
from multiprocessing import cpu_count
import plotext as plt

procs_set = {
    'GFN-FF'   : (1, 2, 4, 8, 16),
    'GFN2-XTB' : (4, 8, 16, 32)
    }

data = read_xyz(sys.argv[1])

cpus = cpu_count()
print(f'--> Detected {cpus} CPUs (considering hyperthreading)')

for method in procs_set.keys():

    timings = {p:[] for p in procs_set[method]}
    thread_info = {p:[] for p in procs_set[method]}

    for procs in procs_set[method]: 
        for threads in (1, 2, 4, 8):

            if procs * threads <= cpus:

                results = []

                results, processes = [], []
                with ProcessPoolExecutor(max_workers=threads) as executor:

                    t_start = perf_counter()
                    
                    for i in range(8):

                        p = executor.submit(
                                xtb_opt,
                                data.atomcoords[0],
                                data.atomnos,
                                method=method,
                                solvent='THF',
                                procs=procs,
                                conv_thr="tight",
                                title=f'process_temp_{i}',
                            )
                        processes.append(p)

                    for p in processes:
                        results.extend(p.result())

                    t_end = perf_counter()
                    timings[procs].append(t_end-t_start)
                    thread_info[procs].append(threads)

                    print(f'{method} --> {procs:2} cores, {threads:2} threads: {time_to_string(t_end-t_start)}')

    plt.theme("pro")
    plt.title(method)
    plt.plotsize(100,25)
    for procs in procs_set[method]:
        times = timings.get(procs, None)
        if times is not None:
            plot = plt.scatter(thread_info[procs], times, label=f'{procs:2} cores')

    plt.xlabel("# Threads")
    plt.ylabel(f"Time (s)")
    plt.show()
    plt.cld()