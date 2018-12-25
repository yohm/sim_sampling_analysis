# A model of network sampling by communication channel

## Model definition

# How to Run

## Compiling

Run `make` twice in "simulator" and "network\_analysis" directories.

```sh
cd simulator && make && cd ../network_analysis && make && cd ..
```

## Running

To run the first simulator, prepare a JSON file specifying the input parameters. See a sample "runner/sample_stoch_block_sampling.json".

Then, run the script as follows.

```
ruby runner/run_stoch_block_sampling.rb parameters.json
```

After you ran the simulation, you'll find the output files in the current directory.

# Installer

The script "install.sh" will compile the codes and then registers the simulator to OACIS.
Run the script after setting "OACIS_ROOT" environment variable.
