from algo import *

def choice(dataset_number):

    if dataset_number == 1:
        # run the algorithm on Web-Stanford dataset
        G = load_data("Stanford")
        return G
    elif dataset_number == 2:
        # run the algorithm on Web-BerkStan dataset
        G = load_data("BerkStan")
        return G
    else:
        raise ValueError("Invalid choice. Please choose a valid option.")

# main function
if __name__ == "__main__":

    dataset_number = int(input("Choose the dataset to work with. The options are:\n\t [1] Web-Stanford\n\t [2] Web-BerkStan\nType your number of choice: "))

    G = choice(dataset_number)
    alphas = [0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]


    ### STANDARD PAGERANK ALGORITHM ###
    iter_dict = dict.fromkeys(alphas, 0)
    list_of_pageranks = [] # list of pageranks dict for each alpha

    start1 = time.time()
    for alpha in alphas:
        x, iter, tol = pagerank(G, alpha, tol=1e-9)
        iter_dict[alpha] = iter
        list_of_pageranks.append(x)
    end1 = time.time()

    total_iter = sum(iter_dict.values())

    print("\nSTANDARD PAGERANK ALGORITHM\n")
    print("\tCPU time (s):", round(end1 - start1,1))
    print("\tMatrix-vector multiplications:", total_iter)
    print("\tAlpha:", alphas)
    print("\tTolerance:", tol)
    print()

    # check if there are entries in the list of pageranks that are empty dict, if so, print the corresponding alpha saying that the algorithm did not converge for that alpha
    for i in range(len(list_of_pageranks)):
        if not list_of_pageranks[i]:
            print("The algorithm did not converge for alpha =", alphas[i])

    ### SHIFTED PAGERANK ALGORITHM ###
    start2 = time.time()
    x, mv, alphas, tol = shifted_pow_pagerank(G, alphas, tol=1e-9)
    end2 = time.time()
    print("\nSHIFTED PAGERANK ALGORITHM\n")
    print("\tCPU time (s):", round(end2 - start2,1))
    print("\tMatrix-vector multiplications:", mv)
    print("\tAlphas:", alphas)
    print("\tTolerance:", tol)
