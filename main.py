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

    start1 = time.time()
    prank, iterations, alpha, tol = pagerank(G)
    end1 = time.time()
    print("STANDARD PAGERANK ALGORITHM\n")
    print("\tCPU time (s):", round(end1 - start1,1))
    print("\tIterations:", iterations)
    print("\tAlpha:", alpha)
    print("\tTolerance:", tol)

    start2 = time.time()
    shifted_pagerank, iterations, alphas, tol = shifted_pow_pagerank(G)
    end2 = time.time()
    print("\nSHIFTED PAGERANK ALGORITHM\n")
    print("\tCPU time (s):", round(end2 - start2,1))
    print("\tIterations:", iterations)
    print("\tAlphas:", alphas)
    print("\tTolerance:", tol)
