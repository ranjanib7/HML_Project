# Efficient Sparse CNN Exploration with SCALE-Sim

[![Documentation Status](https://readthedocs.org/projects/scale-sim-project/badge/?version=latest)](https://scale-sim-project.readthedocs.io/en/latest/?badge=latest)


Our code integrates SCNN  a novel Sparse CNN (SCNN) accelerator architecture aimed at enhancing the performance and energy efficiency of Convolutional Neural Networks (CNNs) into SCALESim. SCALESim \cite{samajdar2018scalesim} serves as a Systolic Array simulator, aiding designers in fine-tuning accelerator parameters for executing diverse models and conducting Data Space Exploration (DSE). However, its capability to furnish optimal performance metrics for Sparse CNNs is limited. This limitation arises from the inclusion of compute cycles that process multiplications for operands with zero values. Eliminating such redundant computations can effectively reduce overall compute cycles. We investigate the impact of sparsity percentages in the input and filter matrices on both compute cycles and mapping efficiency.

![scnn_design_2](https://github.com/ranjanib7/HML_Project/assets/36790410/fd58e4bf-c2bc-4b7e-bdc4-423e18265865)


### *Launching a run*

SCALE-Sim can be run by using the ```scale.py``` script from the scalesim repository and providing the paths to the architecture configuration, and the topology descriptor csv file.

```$ python3 scale.py -c <path_to_config_file> -t <path_to_topology_file> -p <path_to_output_log_dir>```

For our testing purposes, we have used the scale.cfg in the configs directory and the topology file used is test.csv in topology/conv_nets
Since this project is an extension of SCALE-Sim, in addition to the modifications in the cfg file, we have added a new parameter in the topology file test.csv, to represent the Sparsity in the IFMAP and Filter matrices as a percentage.
The modifications made in the topology file are:
![modifications_code](https://github.com/ranjanib7/HML_Project/assets/36790410/42575881-fd0b-4448-aec4-72483fd71a70)

Sample run after running python/scale.py with the above changes for a 64x64 systolic array are:
![sample_run](https://github.com/ranjanib7/HML_Project/assets/36790410/90bbb2cf-dec0-4dc3-af39-8dcd2b7c90c1)


### Output

Here is an example output dumped to stdout when running test.csv with the following configurations: IFMAP Dimensions - 3x3, Filter Dimensions: 2x2, Array Dimensions: 3x3, Sparsity for Filter and Input is set to 0.
![matrices](https://github.com/ranjanib7/HML_Project/assets/36790410/ad02c9f1-c03a-49f6-927d-a01ab874293e)

### Results

We have used the Total Compute Cycles (SCNN) variable to log the number of cycles and mapping efficiency. To calculate the time consumed my the process, the script was run as "time python scale.py"
![image](https://github.com/ranjanib7/HML_Project/assets/36790410/4522969f-0067-4be3-82ec-3a88e4c6451e)


![result1](https://github.com/ranjanib7/HML_Project/assets/36790410/b58505b2-8bf0-409f-b2f4-febcc4e1dc00)

![result2](https://github.com/ranjanib7/HML_Project/assets/36790410/d38c3479-114c-4fb0-8461-12a55d8cb765)

![result3](https://github.com/ranjanib7/HML_Project/assets/36790410/a7360b4c-43d4-485d-b2c1-0760d9e2ec61)

![result4](https://github.com/ranjanib7/HML_Project/assets/36790410/13de078d-e8a3-486c-9b89-20636ca823e0)


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




