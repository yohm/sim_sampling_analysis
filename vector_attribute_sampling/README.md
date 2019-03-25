# Vector attribute model

## Model definition

./vec_attr_sampling.out N k q_sigma q_tau c _seed

First, an Erdos-Renyi random network of size `N` and average degree `k` is constructed.

Then, a tuple of two integers, $(\sigma_{i}, \tau_{i})$, is assigned randomly to each node.
sigma and tau are independently drawn from the uniform distribution $[0,q_{\sigma}-1]$ and $[0,q_{\tau}-1]$, respectively.

After affinity values are assigned to each node, then sample the links with the probability dependent on the affinity of the nodes in both ends.
See section 3.B of the [paper](https://arxiv.org/abs/1902.04707) for details. 

# How to Run

## Compiling

Clone git submodules.

```sh
git submodule update --init --recursive
```

Run `make` twice in "simulator" and "network\_analysis" directories.

```sh
cd simulator && make && cd ../network_analysis && make && cd ..
```

If you would like to specify the compiler explicitly, set `CXX` environment variables.

```
env CXX=g++ make
```

