Lab course: Deep Learning for Computer Vision -- Protein Function Prediction from 3D data
============================
 **Authors:** Atanas Mirchev, Georgi Dikov
 
 **Submitted on** 12.03.2017

## Description
#### Overview
This code is produced during the winter fall of 2016/2017. 
It is a framework for end-to-end training of 3D Convolutional Neural Networks
which aims at discriminating between different protein classes from 3D molecular representation. 
The main datasets used are the [Protein Data Bank](http://pdb101.rcsb.org) and the 
[Enzyme Structures Database](https://www.ebi.ac.uk/thornton-srv/databases/enzymes/).  

The workflow of the framework can be summarized in the following steps:

1. Protein codes (PDB IDs) of interest are fetched from the Enzyme Structures Database
2. Protein descriptor files (pdb****.ent) are downloaded from the Protein Data Bank
3. 3D maps of protein electron density are computed and stored (optionally 3D maps of electrostatic potential 
can be computed, though it is not recommended as the algorithm used to compute the protein's
partial charges is inadequate for such large molecules.)
4. The computed 3D maps and proteins are then split into test, validation and training sets 
5. A 3D ConvNet is built and a training is initiated
6. Validation is performed regularly and best model parameters are backed-up. 
7. After the training is finished, the model is evaluated on a test set and ROC and 
training history curves are generated.
 
#### Framework structure
see `doc/dependency_grpah.pdf`

## Requirements and Dependencies
#### Python dependencies: 
   * RDKit, Bio, prody
   * lasagne, theano, sklearn
   * logging, colorlog
   * seaborn, tvtk, mayavi
   * abc, yaml
   
#### GPU dependencies:
   * CuDNN, CUDA

## Experiment Configuraiton
In order to run one of the experiments devised during the lab course, please use the dedicated
`experiment_launcher.py`. Otherwise, you might want to create a custom `config.yaml` using 
the `default_config.yaml` 
as a template. 

## Result Documentation 
After each experiment involving 2 classes, a ROC curve can be produced. 
This feature is automated but has to be activated. 
All additional experiment results can be found in a dedicated directory 
named after the experiment name. 


