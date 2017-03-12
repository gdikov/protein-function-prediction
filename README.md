Lab course: Deep Learning for Computer Vision -- Protein Function Prediction from 3D data
============================
 **Authors:** Atanas Mirchev, Georgi Dikov
 
 **Submitted on** 12.03.2017

## Description
#### Overview
This code is produced during the winter term of 2016/2017.
It is a framework for end-to-end training of 3D Convolutional Neural Networks
which aims at discriminating between different protein classes from 3D molecular representation. 
The main datasets used are the [Protein Data Bank](http://pdb101.rcsb.org) and the 
[Enzyme Structures Database](https://www.ebi.ac.uk/thornton-srv/databases/enzymes/).  

The workflow of the framework can be summarized in the following steps:

1. Protein codes (PDB IDs) of interest are fetched from the Enzyme Structures Database
2. Protein descriptor files (pdb****.ent) are downloaded from the Protein Data Bank
3. 3D maps of protein electron density are computed and stored (optionally 3D maps of electrostatic potential 
can be computed, though it is not recommended as the algorithm used to compute the protein's
partial charges is inadequate for such large molecules; the ESP computation is thus disabled by default)
4. The computed 3D maps and proteins are then split into test, validation and training sets 
5. A 3D ConvNet for multi-label classification is built and a training is initiated
6. Validation is performed regularly and best model parameters are backed-up. 
7. After the training is finished, the model is evaluated on a test set and ROC and 
training history curves are generated.
 
#### Framework structure
See `doc/dependency_graph.pdf`, it is the full dependency chart between all system components.

## Requirements and Dependencies
#### Python dependencies:
    * theano
    * Lasagne
    * matplotlib
    * mayavi
    * nolearn
    * numpy
    * ProDy
    * requests
    * scipy
    * seaborn
    * mayavi
    * beautifulsoup4
    * rdkit
    * scikit_learn
    * PyYAML
    * colorlog

Also see `requirements.txt` for versions that are guaranteed to work.

#### GPU dependencies:
   * CUDA, CuDNN

## Experiment configuraiton
In order to re-run one of the experiments devised by us during the lab course, please use the dedicated `experiment_launcher.py` and follow the instructions. You will be given the choice to pick from a set of predefined experiments:

```
cd ./project_dir
python experiment_launcher.py
```

Alternatively, you might want to create a custom `config.yaml` using the `experiments/example_config.yaml` as a template. Then you can start an experiment with this `config.yaml` either by running the `experiment_launcher.py` and choosing that from the menu, or by manually adjusting `main.py` to your needs.

Please consult the contents of `experiments/example_config.yaml` for configuration options.

## Data management
Note that every experiment that requires a different pre-processing of the data, or a different set of proteins, should have its own unique data directory, specified in the `config.yaml` as `data.dir`. The structure of the `data_dir` will be as follows:

* `data_dir/raw` - contains the downloaded `pdb****.ent` files, before the pre-processing, in separate folders for each protein.
* `data_dir/processed` - contains the processed versions of the proteins (usually `.memmap` files, e.g. the precomputed molecule 3D maps)
* `data_dir/train` - the train set (subset of `data_dir/processed`) that was split up; also contains the validation set
* `data_dir/test` - the test set (subset of `data_dir/processed`) that was split up
* `data_dir/models` - here all trained models and their evaluation (ROC curves, test results, etc.) are stored, in separate folders for each model. Also, each model's folder will contain the exact `config.yaml` file used for the training of the model, as well as information about the actual distribution of proteins in the train / test sets (the latter infos are stored in `test_prot_codes.pickle`, `train_prot_codes.pickle`, `val_prot_codes.pickle`). That way the training is fully reproducible.

It is perfectly fine to leave `data_dir` unchanged in the `config.yaml` if you want to run multiple experiments on the same data set, without changing the pre-processing of the data. Then for each run, a different model (and hence a different model directory under `data_dir/models`) will be created. You can later inspect the results of the particular run by checking the model's directory.

TIP: if you'd like to use the same protein data for experiments that require different pre-processing, but do not want to download it multiple times, you can soft-link the `raw` directory between experiments:
```
ln -s ./experiment1/raw ./experiment2/raw
```
That way, our data manager will not download proteins which have been already downloaded.

## Result documentation 
After each experiment, a ROC curve can be produced. In the case of 2 classes, the curve is a standard binary classification ROC. In the case of more than two classes, the plot will contain micro- and macro-average ROC curves. See the documentation in `protfun/visualizer/roc_view.py` for more details. Additionally, a plot of the training progress will be produced (`accuracy_history.png` and `loss_history.png`) and activations of the hidden layers for a small number of proteins from the test set will be saved (`activations.pickle`). The results from the run on the test set are also saved (`test_targets.pickle`, `test_predictions.pickle`, `test_proteins.pickle`). All of this information is stored under the specific model's folder under `data_dir/models`.

The generation of experiment results is automatic if you use the `experiment_launcher.py`, otherwise you can do it manually to suit your needs. Check the code under `protfun/visualizer/experiment_visualizer.py` to get an idea of how it is all put together.


## Additional visualizations

In order to make sure that the input files are being correctly pre-processed and 
that the 3D electron density (and optionally electrostatics potential) maps are 
meaningfully generated, there are additional visualization tools which are not 
invoked during an end-to-end training. They can be found in `protfun/visualizer/molview.py`.
To generate a sample electron density visualization run:
```python
from protfun.visualizer.molview import MoleculeView

# path_to_grid_3d is the full path to the precomputed 3D density map
# grid_3d is a NxNxN numpy array with the precomputed 3D density map
viewer = MoleculeView(data_dir=path_to_grid_3d,
                      data={"density": grid_3d},
                      info={"name": "Molecule Name"})
viewer.density3d()
```

Also, you can check the distribution of the molecule "diameters" (largest distance between two atoms, in angstroms) under `doc/diameter_disr*.png`, a small snippet of how it was produced can be found in `protfun/utils/check_diameter.py`
