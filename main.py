#! /usr/bin/python

from algo import *
import warnings
import argparse
warnings.filterwarnings("ignore")

# df = pd.DataFrame(columns=["method" "alpha", "cpu_time", "mv", "tol"])

def run_standard_pagerank(G, alphas):
    print("\nStarting the standard pagerank algorithm...")

    iter_dict = dict.fromkeys(alphas, 0)
    list_of_pageranks = []

    start1 = time.time()
    for alpha in alphas:
        x, iter, tol = pagerank(G, alpha, tol=1e-6)
        iter_dict[alpha] = iter
        list_of_pageranks.append(x)
    end1 = time.time()

    total_iter = sum(iter_dict.values())
    cpu_time = round(end1 - start1,1)
    mv = total_iter

    print("\nSTANDARD PAGERANK ALGORITHM RESULTS:\n")
    print("\tCPU time (s):", cpu_time)
    print("\tMatrix-vector multiplications:", mv)
    print("\tAlpha(s):", alphas)
    print("\tTolerance:", tol)

    for i in range(len(list_of_pageranks)):
        if not list_of_pageranks[i]:
            print("The algorithm did not converge for alpha =", alphas[i])


    # df.loc[len(df)] = ["Power Method", alphas, cpu_time, mv, tol]
    # df.to_csv(args.dataset + "_results.tsv", sep="\t", index=False)
    # print("\nThe results are saved in the file:", args.dataset + "_results.tsv")


def run_shifted_powe(G, alphas):
    print("\nStarting the shifted power method... (this may take a while)")

    start2 = time.time()
    x, mv, alphas, tol = shifted_pow_pagerank(G, alphas, tol=1e-6)
    end2 = time.time()
    cpu_time = round(end2 - start2,1)

    print("\nSHIFTED PAGERANK ALGORITHM RESULTS:\n")
    print("\tCPU time (s):", cpu_time)
    print("\tMatrix-vector multiplications:", mv)
    print("\tAlphas(s):", alphas)
    print("\tTolerance:", tol)

    # df.loc[len(df)] = ["Shifted Power Method", alphas, cpu_time, mv, tol]
    # df.to_csv(args.dataset + "_results.tsv", sep="\t", index=False)
    # print("\nThe results are saved in the file:", args.dataset + "_results.tsv")

# main function
if __name__ == "__main__":
  
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Stanford", help="Choose the dataset to work with. The options are: Stanford, NotreDame, BerkStan")
    parser.add_argument("--algo", type=str, default="both", help="Choose the algorithm to use. The options are: pagerank, shifted_pagerank, both")
    args = parser.parse_args()

    G = load_data(args.dataset)

    alphas = [0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

    if args.algo == "pagerank":
        run_standard_pagerank(G, alphas)
    elif args.algo == "shifted_pagerank":
        run_shifted_powe(G, alphas)
    elif args.algo == "both":
        run_standard_pagerank(G, alphas)
        run_shifted_powe(G, alphas)



