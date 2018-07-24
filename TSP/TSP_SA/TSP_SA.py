#coding:utf-8
import numpy as np
import json
import math
from util import *
from args import *
import pandas as pd
distance_x = [
    178,272,176,171,650,499,267,703,408,437,491,74,532,
    416,626,42,271,359,163,508,229,576,147,560,35,714,
    757,517,64,314,675,690,391,628,87,240,705,699,258,
    428,614,36,360,482,666,597,209,201,492,294]
distance_y = [
    170,395,198,151,242,556,57,401,305,421,267,105,525,
    381,244,330,395,169,141,380,153,442,528,329,232,48,
    498,265,343,120,165,50,433,63,491,275,348,222,288,
    490,213,524,244,114,104,552,70,425,227,331]
coor = np.array([[a,b] for a,b in zip(distance_x,distance_y)])

def main():
    # Get arguments
    args = parse_args()

    # Use the corresponding data file
    if args.file:
        filename = args.file
    elif args.data == 'nctu':
        filename = "data/nctu.csv"
    elif args.data == 'nthu':
        filename = "data/nthu.csv"
    elif args.data == 'thu':
        filename = "data/thu.csv"
    else:
        print("ERROR: undefined data file")
    #coordinates = np.loadtxt(filename, delimiter=',')
    coordinates = coor
    # Constant Definitions
    NUM_NEW_SOLUTION_METHODS = 3
    SWAP, REVERSE, TRANSPOSE = 0, 1, 2

    # Params Initial
    num_location = coordinates.shape[0]
    markov_step = args.markov_coefficient * num_location
    T_0, T, T_MIN = args.init_temperature, args.init_temperature, 1
    T_NUM_CYCLE = 1

    # Build distance matrix to accelerate cost computing
    distmat = get_distmat(coordinates)

    # States: New, Current and Best
    sol_new, sol_current, sol_best = (np.arange(num_location), ) * 3
    cost_new, cost_current, cost_best = (float('inf'), ) * 3

    # Record costs during the process
    costs = []

    # previous cost_best
    prev_cost_best = cost_best

    # counter for detecting how stable the cost_best currently is
    cost_best_counter = 0

    # Simulated Annealing
    while T > T_MIN and cost_best_counter < args.halt:
        for i in np.arange(markov_step):
            # Use three different methods to generate new solution
            # Swap, Reverse, and Transpose
            choice = np.random.randint(NUM_NEW_SOLUTION_METHODS)
            if choice == SWAP:
                # swap two points in the genes
                sol_new = swap(sol_new)
            elif choice == REVERSE:
                # choose a continuous segments and reverse them
                sol_new = reverse(sol_new)
            elif choice == TRANSPOSE:
                # insert a segment after some point
                sol_new = transpose(sol_new)
            else:
                print("ERROR: new solution method %d is not defined" % choice)

            # Get the total distance of new route
            cost_new = sum_distmat(sol_new, distmat)

            if accept(cost_new, cost_current, T):
                # Update sol_current
                sol_current = sol_new.copy()
                cost_current = cost_new

                if cost_new < cost_best:
                    sol_best = sol_new.copy()
                    cost_best = cost_new
            else:
                sol_new = sol_current.copy()

        # Lower the temperature
        alpha = 1 + math.log(1 + T_NUM_CYCLE)
        T = T_0 / alpha
        costs.append(cost_best)
        # print T, alpha
        # Increment T_NUM_CYCLE
        T_NUM_CYCLE += 1

        # Detect stability of cost_best
        if isclose(cost_best, prev_cost_best, abs_tol=1e-12):
          cost_best_counter += 1
        else:
          # Not stable yet, reset
          cost_best_counter = 0

        # Update prev_cost_best
        prev_cost_best = cost_best

        # Monitor the temperature & cost
        print("Temperature:", "%.2f°C" % round(T, 2),
              " Distance:", "%.2fm" % round(cost_best, 2),
              " Optimization Threshold:", "%d" % cost_best_counter)

    # Show final cost & route
    print("Final Distance:", round(costs[-1], 2))
    print("Best Route", sol_best)

    route = json.dumps(sol_best.tolist())
    payloads = [ costs[-1], route, markov_step ]

    # Save the result to sqlite
    save_sqlite(payloads)

    #export to path.json for google map
    export2json(filename, sol_best)

    # Plot cost function and TSP-route
    plot(sol_best.tolist(), coordinates, costs)

if __name__ == "__main__":
    main()
