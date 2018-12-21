# A model of network sampling by communication channel

A simulation code for the model for sampling social network by communication channel.
The model is proposed in paper entitled "What Big Data tells: Sampling the social network by communication channels" published in Phys.Rev.E [link](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.94.052319).
The paper is also on [arXiv](https://arxiv.org/abs/1511.08749).

# Model definition

First, a Erdos-Renyi random network of size `N` and average degree `k0` is constructed.

Then, a scalar value , called "affinity" of node i, is assigned randomly to each node.
The affinity values are taken from the Weibull distribution with exponent `alpha` and a characteristic scale `f0`.

After affinity values are assigned to each node, then sample the links with the probability dependent on the affinity of the nodes in both ends.
The sampling probability is given by `p_{ij} = p(fi,fj)`, where p(x,y) is the generalized mean with exponent `beta`.
See section V of the paper for more accurate description.

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

There are two kinds of simulators.
The first one requires the value of *f0* as an input parameter and then conduct sampling.
The other one requires the degree after the sampling and its acceptable margin after the sampling.
Samplings are conducted iteratively to tune *f0* so that the sampled network are close enough to the given degree.

## Running

To run the first simulator, prepare a JSON file specifying the input parameters like the following

```json
{
  "N": 1000,
  "k": 20.0,
  "f0": 0.3,
  "alpha": 1.0,
  "beta": 0.0,
  "_seed": 1234
}
```

and then run the script as follows.

```
ruby runner/run_power_mean_sampling.rb parameters.json
```

After you ran the simulation, you'll find the output files in the current directory.

To run the second simulator, the format of the input JSON is slightly different.

```json
{
  "N": 2000,
  "k0": 150.0,
  "expected_k": 15.0,
  "dk": 2.0,
  "alpha": 1.0,
  "beta": 0.0,
  "_seed": 1234
}
```

and then run the script as follows.

```
ruby runner/run_power_mean_sampling_tuned_f0.rb parameters.json
```

# Installer

Run `install.sh` will compile the codes and then registers two simulators to OACIS.
To use it in OACIS docker container, run the container from `oacis/oacis_jupyter` image and run the following. (Container name may be different from yours.)

```sh
docker exec -it -u oacis my_oacis bash -c "git clone --recursive https://github.com/yohm/sim_power_mean_sampling.git && sim_power_mean_sampling/install.sh"
```

