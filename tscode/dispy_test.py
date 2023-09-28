import dispy
import socket
import numpy as np
from tscode.optimization_methods import optimize
from tscode.utils import graphize, read_xyz, time_to_string

if __name__ == '__main__':

    data = read_xyz(f'mol.xyz')
    graph = graphize(data.atomcoords[0], data.atomnos)

    def node_wrapper(func, **kwargs):
        node = socket.gethostname()
        print(node)
        return func(**kwargs), node

    queue = dispy.JobCluster(node_wrapper, depends=[])
    jobs = []

    for i in range(5):

        process = queue.submit(
                                optimize,
                                    coords=data.atomcoords[0],
                                    atomnos=data.atomnos,
                                    calculator="ORCA",
                                    method="R2SCAN-3C",
                                    maxiter=3,
                                    constrained_indices=np.array([6,7]),
                                    mols_graphs=graph,
                                    procs=16,
                                    max_newbonds=0,
                                    check=False,

                                    logfunction=print,
                                    title=f'Candidate_{i+1}',)
        
        jobs.append(process)

    for job in jobs:
        *results, node = job()
        print(f'Completed job on node {node} in {time_to_string(job.end_time-job.start_time)}')

    queue.print_status()