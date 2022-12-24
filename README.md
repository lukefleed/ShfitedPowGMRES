# ShfitedPowGMRES

> Relation of the project: [view](https://github.com/lukefleed/ShfitedPowGMRES/blob/main/tex/main.pdf) / [download](https://github.com/lukefleed/ShfitedPowGMRES/raw/main/tex/main.pdf)

This repository contains the code of my attempt to replicate the results obtained in `[1]`. The scripts are all written in python and are heavily build around the libraries SciPy and NumPy. To install all the required packages with `pip` run the following command in terminal

```bash
pip install -r requirements.txt
```

At the moment, the standard and shifted power method to compute the PageRank with multiple damping factors are fully implemented (as described in `[1]`). To run the program, go into the folder `src/` and execute the `./main.py` file. It takes as input two arguments:

- `--dataset`: the options are `BerkStan` and `Stanford`. This commands selects the web-graph to run the algorithms on. 
- `--algo`: the options are `power`, `shifted`, `both`. If you choose the last option, it will first run the standard power method and then the shifted one. 

Here an example of what's described above.

```bash
cd src
sudo chmod +x main.py
```

Now we can run the program

```bash
./main.py --dataset Stanford --algo both
```

## Under development 

In the `testing/` folder there are two python notebook that contains the attempt on replicating the results obtained in `[1]` for the shifted GMRES method. The implementation of the Arnoldi process is fully working. On the other hand, there are several problems on the shifted GMRES algorithm that I can't figure out.  


## References

`[1]` _Zhao-Li Shen, Meng Su, Bruno Carpentieri, and Chun Wen. Shifted power-gmres method accelerated by extrapolation for solving pagerank with multiple damping factors. Applied Mathematics and Computation, 420:126799, 2022_