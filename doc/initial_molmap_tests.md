#### Preprocessing and 3D-representation

* **Step 1**: read the SDF files, process the molecules and store them into .memmap files. This is done just once.
* **Step 2**: read the .memfiles and store their contents into Theano tensors.
* **Step 3**: compute electrostatic potential and electron density at each grid point. Do this dynamically on the GPU,
this step will be always executed.


##### Some crunched numbers (on beetz10, no other load, 2GB gpu memory:)
> old layer, 40 molecules, batch_size = 1 -> 36.5 to 36.7 seconds, ~ 200-300 MB max memory

> old layer, 40 molecules, batch_size = 2 -> 24.5 seconds, up to ~ 1.400 GB memory

> old layer, batch_size = 3 -> cannot fit into memory any more

> new layer, 40 molecules, batch_size = 1 (not scan) -> 42.3 seconds, ~ 400 MB max memory

> new layer, 40 molecules, batch_size = 2 -> 23.6 to 24 seconds, ~ 500 MB ? max memory

> new layer, 39 molecules, batch_size = 3 -> 16 to 16.23 seconds, 1000 MB max memory

> new layer, 40 molecules, batch_size = 4 -> 12.9 to 13.17 seconds, 1.4 GB max memory

> new layer, 40 molecules, batch_size = 4, T.switch instead of elem-wise mult -> 12.8 seconds, 1.4 GB max memory

> new layer, batch_size = 5 -> cannot fit into memory any more

##### More crunched numbers (on beetz7, no other load, 2GB gpu memory, whole proteins, grid_size=16x16x16):
> minibatch 1, 40 molecules -> 36 - 37 seconds
> minibatch 2, 40 molecules -> 21 - 22 seconds
> minibatch 3, 39 molecules -> 13 - 14 seconds
> minibatch 4, 40 molecules -> out of memory on the small video card
