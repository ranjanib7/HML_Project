# Systolic CNN AcceLErator Simulator with Sparse functiality (SCALE Sim) v3_latest

[![Documentation Status](https://readthedocs.org/projects/scale-sim-project/badge/?version=latest)](https://scale-sim-project.readthedocs.io/en/latest/?badge=latest)


Our code integrates SCNN  a novel Sparse CNN (SCNN) accelerator architecture aimed at enhancing the performance and energy efficiency of Convolutional Neural Networks (CNNs) into SCALESim. SCALESim \cite{samajdar2018scalesim} serves as a Systolic Array simulator, aiding designers in fine-tuning accelerator parameters for executing diverse models and conducting Data Space Exploration (DSE). However, its capability to furnish optimal performance metrics for Sparse CNNs is limited. This limitation arises from the inclusion of compute cycles that process multiplications for operands with zero values. Eliminating such redundant computations can effectively reduce overall compute cycles. We investigate the impact of sparsity percentages in the input and filter matrices on both compute cycles and mapping efficiency.

![scnn_design_2](https://github.com/ranjanib7/HML_Project/assets/36790410/fd58e4bf-c2bc-4b7e-bdc4-423e18265865)


## Getting started in 30 seconds



### *Launching a run*

SCALE-Sim can be run by using the ```scale.py``` script from the repository and providing the paths to the architecture configuration, and the topology descriptor csv file.

```$ python3 scale.py -c <path_to_config_file> -t <path_to_topology_file> -p <path_to_output_log_dir>```

Try it now in this jupyter [notebook](https://github.com/scalesim-project/scalesim-tutorial-materials/blob/main/scaledemo.ipynb).

### *Running from source*

The above method uses the installed package for running the simulator.
In cases where you would like to run directly from the source, the following command should be used instead.

```$ python3 <scale_sim_repo_root>/scalesim/scale.py -c <path_to_config_file> -t <path_to_topology_file>```

If you are running from sources for the first time and do not have all the dependencies installed, please install them first  using the following command.

```$ pip3 install -r <scale_sim_repo_root>/requirements.txt```

## Tool inputs

SCALE-Sim uses two input files to run, a configuration file and a topology file.

### Configuration file

The configuration file is used to specify the architecture and run parameters for the simulations.
The following shows a sample config file:

![sample config](https://github.com/scalesim-project/scale-sim-v2/blob/main/documentation/resources/config-file-example.png "sample config")

The config file has three sections. The "*general*" section specifies the run name, which is user specific. The "*architecture_presets*" section describes the parameter of the systolic array hardware to simulate.
The "*run_preset*" section specifies if the simulator should run with user specified bandwidth, or should it calculate the optimal bandwidth for stall free execution.

The detailed documentation for the config file could be found **here (TBD)**

### Topology file

The topology file is a *CSV* file which decribes the layers of the workload topology. The layers are typically described as convolution layer parameters as shown in the example below.

![sample topo](https://github.com/scalesim-project/scale-sim-v2/blob/main/documentation/resources/topo-file-example.png "sample topo")

For other layer types, SCALE-Sim also accepts the workload desciption in M, N, K format of the equivalent GEMM operation as shown in the example below.

![sample mnk topo](https://github.com/scalesim-project/scale-sim-v2/blob/doc/anand/readme/documentation/resources/topo-mnk-file-example.png "sample mnk topo")

The tool however expects the inputs to be in the convolution format by default. When using the mnk format for input, please specify using the  ```-i gemm``` switch, as shown in the example below.

```$ python3 <scale sim repo root>/scalesim/scale.py -c <path_to_config_file> -t <path_to_mnk_topology_file> -i gemm```

### Output

Here is an example output dumped to stdout when running Yolo Tiny (whose configuration is in yolo_tiny.csv):
![screen_out](https://github.com/scalesim-project/scale-sim-v2/blob/doc/anand/readme/documentation/resources/output.png "std_out")

Also, the simulator generates read write traces and summary logs at ```<run_dir>/../scalesim_outputs/```. The user can also provide a custom location using ```-p <custom_output_directory>``` when using `scalesim.py` file.
There are three summary logs:

* COMPUTE_REPORT.csv: Layer wise logs for compute cycles, stalls, utilization percentages etc.
* BANDWIDTH_REPORT.csv: Layer wise information about average and maximum bandwidths for each operand when accessing SRAM and DRAM
* DETAILED_ACCESS_REPORT.csv: Layer wise information about number of accesses and access cycles for each operand for SRAM and DRAM.

In addition cycle accurate SRAM/DRAM access logs are also dumped and could be accesses at ```<outputs_dir>/<run_name>/``` eg `<run_dir>/../scalesim_outputs/<run_name>`


## Developers

Main devs:
* Ranjani Balasubramanyam(@rbalasub32)
* Abhipsa Panigrahi(@apanigrahi33)
* Aakash Venkataraman (@avenkata44)
* Ashwin Kulkarni (@akulkarni379)
* Ketan Anand (@kanand)


## Individual Contributions

Ashwin Kulkarni:
Compute unit implementation: Worked on the streaming of weights, and tiling of inputs, and fetched the data that should be stored in the accumulator matrix.
Worked on the development of the systolic array conversion to multiplier arrays for SCNN.
Ideation - Responsible for literature review to decide between SCNN and VEGETA architectures for integration with SCALESim.
Presentation and Poster - Collaborated on the development and presentation of Poster.
      
Ranjani Balasubramanyam
Worked on converting the MAC Units in SCALESim to multiplier arrays and generating the updated operand matrices to incorporate SCNN Cartesian Product computation.
Worked on integrating sparsity calculation and implementing design changes to the existing SCALESim to improve performance for sparse networks.
Worked on modifying the handling accumulator and multiplication cycles in the updated framework to account for parallel multiplication and accumulations in the SCNN Dataflow.
Contributed to the development of poster and report.
Presentation

Ketan Anand
Compute unit implementation: Work on the streaming of weights, tiling, get the data to be stored in the accumulator matrix
Convolution part, give coordinates of valid convolution outputs, handle storing these in accumulator.
Analytical calculation of compute cycles for multiplication and accumulation
Performance anaylsis over different filter and input sparsities
Poster development and presentation

Abhipsa Panigrahi
Worked on Algorithm to calculate the coordinates of valid convolution outputs, and route these to the proper accumulator buffers
Implementation of compute cycle calculation for multiplication and accumulation
Validation of design: Performance analysis to study the trend shown by number of compute cycles, execution times and mapping efficiency with changes in IFMAP sizes, filter sizes and sparsities in both
Ideation - Analytical calculation of compute cycles
Contributed to Poster and Report development
Presentation
    
Aakash Venkataraman
Handle the convolution portion, give coordinates of valid convolution outputs, and handle the storing of these values in the accumulator matrix.
Helped in the development and validating of the algorithm to handle sparse input matrix.    
Worked on accumulator implementation for mapping inputs. 
Literature review for VEGETA vs SCNN.
Modifying how the cycles are computed in existing ScaleSIM code. 
Presentation and Poster development




