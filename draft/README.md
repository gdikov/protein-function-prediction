#### Preprocessing and 3D-representation

* **Step 1**: read the SDF files, process the molecules and store them into .memmap files. This is done just once.
* **Step 2**: read the .memfiles and store their contents into Theano tensors.
* **Step 3**: compute electrostatic potential and electron density at each grid point. Do this dynamically on the GPU,
this step will be always executed.