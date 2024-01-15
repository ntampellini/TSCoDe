# coding=utf-8
'''

TSCoDe: Transition State Conformational Docker
Copyright (C) 2021-2024 Nicolò Tampellini

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

https://github.com/ntampellini/TSCoDe

Nicolo' Tampellini - nicolo.tampellini@yale.edu

'''
import argparse
import os
import sys

__version__ = '0.4.5'

if __name__ == '__main__':


    usage = '''python -m tscode [-h] [-s] [-t] inputfile [-n NAME]
        
        positional arguments:
          inputfile               Input filename, can be any text file.

        optional arguments:
          -h, --help              Show this help message and exit.
          -s, --setup             Guided setup of the calculation settings.
          -t, --test              Perform some tests to check the TSCoDe installation.
          -n, --name NAME         Custom name for the run.
          -cl,--command_line      Read instructions from command line instead of from inputfile.
          -c, --cite              Print citation links.
          -p, --profile           Profile the run through cProfiler.
          -b, --benchmark FILE    Benchmark the geometry optimization of FILE to get the optimal number of procs/threads.
          --procs                 Number of processors to be used by each optimization job.
          --threads               Number of parallel threads used.
          '''

    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("-s", "--setup", help="Guided setup of the calculation settings.", action="store_true")
    parser.add_argument("-t", "--test", help="Perform some tests to check the TSCoDe installation.", action="store_true")
    parser.add_argument("-cl", "--command_line", help="Read instructions from command line instead of from inputfile.", action="store")
    parser.add_argument("inputfile", help="Input filename, can be any text file.", action='store', nargs='?', default=None)
    parser.add_argument("-n", "--name", help="Custom name for the run.", action='store', required=False)
    parser.add_argument("-c", "--cite", help="Print the appropriate document links for citation purposes.", action='store_true', required=False)
    parser.add_argument("-p", "--profile", help="Profile the run through cProfiler.", action='store_true', required=False)
    parser.add_argument("-b", "--benchmark", help=("Benchmark the geometry optimization of FILE to get the optimal number " +
                        "of procs/threads."), action='store', required=False, default=False)
    # parser.add_argument("-r", "--restart", help="Restarts previous run from an embedder.pickle object.", action='store', required=False, default=False)
    parser.add_argument("--procs", help="Number of processors to be used by each optimization job.", action='store', required=False, default=None)
    parser.add_argument("--threads", help="Number of parallel threads used.", action='store', required=False, default=None)

    args = parser.parse_args()

    if (not (args.test or args.setup or args.command_line or args.benchmark)) and args.inputfile is None:
        parser.error("One of the following arguments are required: inputfile, -t, -s, -b.")

    if args.benchmark:
        from tscode.concurrent_test import run_concurrent_test
        run_concurrent_test(args.benchmark)
        sys.exit()

    if args.setup:
        from tscode.modify_settings import run_setup
        run_setup()
        sys.exit()

    if args.cite:
        print('No citation link is available for TSCoDe yet. You can link to the code on https://www.github.com/ntampellini/TSCoDe')
        sys.exit()

    if args.test:
        from tscode.tests import run_tests
        run_tests()
        sys.exit()

    if args.command_line:
        
        filename = 'input_TSCoDe.txt'
        with open(filename, 'w') as f:
            f.write(args.command_line)

        args.inputfile = filename

    filename = os.path.realpath(args.inputfile)

    from tscode.embedder import Embedder

    if args.profile:
        from tscode.profiler import profiled_wrapper
        profiled_wrapper(filename, args.name)
        sys.exit()

    # if args.restart:
    #     import pickle
    #     with open(args.restart, 'rb') as _f:
    #         embedder = pickle.load(_f)
        # initialize embedder from pickle file

        embedder.run()
        # run the program

    # import faulthandler
    # faulthandler.enable()

    embedder = Embedder(filename, stamp=args.name, procs=args.procs, threads=args.threads)
    # initialize embedder from input file

    embedder.run()
    # run the program