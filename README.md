# Model of CA1 section of Hippocampus
### Goal is to show SPWR in CA1
## Building network
```
python build_network.py
```
## Building via batch is helpful if network takes a long time to build
```
sbatch batch_build.sh
```
## Running Network on 60 cores(20 min for just a raster, 3 hours for LFP)
```
mpirun -n 60 nrniv -mpi -python run_network.py simulation_config.json
```
## Batch run
### Email results will get sent to Greg Glicket's email by default
```
sbatch batch_run.sh
```
## Running on NSG takes about 3 min for raster and 20 for LFP on 3 node
### Step one
#### Build network on local machine
### Step two
#### Compress CA1_model folder into a zip
### Step three
#### Upload zip to NSG in the data folder
### Step four
#### Create a new task and use the Neuron on Expanse toolkit
### Step five
#### Change parameters and then click box for if your code is python and change the input file name to run_network.py and then change number of nodes to how many core you want
### Step six
#### Submit network to run on NSG!
