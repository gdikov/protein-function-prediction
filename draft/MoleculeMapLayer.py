import theano
import theano.tensor as T
from lasagne import init
import lasagne
import numpy as np
import os
import gzip
import theano.tensor.nlinalg as Tnlin
floatX = theano.config.floatX
intX = np.int32 # FIXME is this the best choice? (changing would require removing and recreating memmap files)



class MoleculeMapLayer(lasagne.layers.Layer):
    '''
    This is a Lasagne layer to calculate 3D maps (electrostatic potential, and
    electron density estimated from VdW radii) of molecules (using Theano,
    i.e. on the GPU if the user wishes so).
    At initialization, the layer is told whether it should use the file
    with active or inactive compounds. When called, the layer input is an array
    of molecule indices (both for actives and inactives - the layer selects the
    respective half depending on whether it was initialized for actives or
    inactives), and the output are the 3D maps.
    Currently works faster (runtime per sample) if `minibatch_size=1` because
    otherwise `theano.tensor.switch` is slow.
    '''

    def __init__(self, incoming, active_or_inactive, minibatch_size=None, **kwargs):
        # input to layer are indices of molecule
        
        super(MoleculeMapLayer, self).__init__(incoming, **kwargs) # see creating custom layer!
        
        if minibatch_size is None:
            minibatch_size = 1
            print "minibatch_size not provided - assuming it is `{:d}`.  If this is wrong, please provide the correct one, otherwise dropout will not work.".format(minibatch_size)
        
        # Molecule file name
        if active_or_inactive == 0:
            filename = "1798_inactives_cleaned.sdf"
        elif active_or_inactive == 1:
            filename = "1798_actives_cleaned.sdf"
            
        # Create a sensible output prefix from the input file name
        split_path = os.path.splitext(filename)
        while split_path[1]==".gz" or split_path[1] ==".sdf":
            split_path = os.path.splitext(split_path[0])
        prefix = split_path[0]
        
        # TODO get the max number of atoms and not to set it manually to 135!
        max_natom = 135

        try: # use memmap files - fast and does not require installing rdkit :)
            
            coords = np.memmap(prefix+'_coords.memmap', mode='r', dtype=floatX).reshape((-1, max_natom, 3))
            charges = np.memmap(prefix+'_charges.memmap', mode='r', dtype=floatX).reshape((-1, max_natom))
            vdwradii = np.memmap(prefix+'_vdwradii.memmap', mode='r', dtype=floatX).reshape((-1, max_natom))
            n_atoms = np.memmap(prefix+'_n_atoms.memmap', mode='r', dtype=intX)
            atom_mask = np.memmap(prefix+'_atom_mask.memmap', mode='r', dtype=floatX).reshape((-1, max_natom))
            self.total_nmol = n_atoms.size
            
        except IOError: # memmap files not found - make them
            print "Making memmap files..."
        
            import rdkit.Chem as Chem
            import rdkit.Chem.rdPartialCharges as rdPC
            import rdkit.Chem.rdMolTransforms as rdMT
            
            # Make sure the .sdf molecule file exists
            if not os.path.isfile(filename) :
                print "File \"" + filename + "\" does not exist"
    
            # Open up the file containing the molecules
            infile=None
            if os.path.splitext(filename)[1] == ".gz" :
                infile=gzip.open(filename,"r")
            else :
                infile=open(filename,"r")
    
    
            # the SDF parser object, reads in molecules
            # there is also a random-access version of this, but it must be given
            # a filename instead of a file stream (called SDMolSupplier, or FastSDMolSupplier)
            # defined using: import rdkit.Chem as Chem
            sdread=Chem.ForwardSDMolSupplier(infile,removeHs=False)
    
            # Periodic table object, needed for getting VDW radii
            pt=Chem.GetPeriodicTable()
    
            
            
            mol_number = 0
            n_atoms = []
            #n_atoms = theano.TypedListType(T.lvector)()
            molecules = [x for x in sdread]
            self.total_nmol = len(molecules)
            
    
            coords = np.zeros(shape=(self.total_nmol,max_natom,3), dtype=floatX)
            charges = np.zeros(shape=(self.total_nmol,max_natom), dtype=floatX)
            vdwradii = np.zeros(shape=(self.total_nmol,max_natom), dtype=floatX)
            atom_mask = np.zeros(shape=(self.total_nmol,max_natom), dtype=floatX)
    
            
            for mol in molecules:
                # compute the atomic partial charges
                rdPC.ComputeGasteigerCharges(mol)
    
                # get the conformation of the molecule and number of atoms (3D coordinates)
                conformer=mol.GetConformer()
                
                n_atoms.append(mol.GetNumAtoms())
                atom_mask[mol_number, 0:n_atoms[mol_number]] = 1;
                
                # calculate the center of the molecule
                # Centroid is the center of coordinates (center of mass of unit-weight atoms)
                # Center of mass would require atomic weights for each atom: pt.GetAtomicWeight()
                center=rdMT.ComputeCentroid(conformer,ignoreHs=False)
    
                # Get atom coordinates, charges, and vdw radii
                atoms=mol.GetAtoms()
    
                #print "n of atoms ", n_atoms[mol_number]
                #print "max natoms", np.amax(n_atoms)
                for ano in range(0,n_atoms[mol_number]) :
                
                    atom=atoms[ano]
    
                    # this gets the atom coordinates and subtracts the center of mass of the molecule
                    # TODO this can probably be sped up by reading this into an ndarray and vectorizing the operation; not required fixing because this is not the bottleneck
                    coord=conformer.GetAtomPosition(ano)
                    coords[mol_number,ano,0]=coord.x-center.x
                    coords[mol_number,ano,1]=coord.y-center.y
                    coords[mol_number,ano,2]=coord.z-center.z
                    # atomic charge
                    charges[mol_number,ano]=float(atom.GetProp("_GasteigerCharge"))
                    # vdw radius
                    vdwradii[mol_number,ano]=pt.GetRvdw(atom.GetAtomicNum())
    
                # Euclidean distances for each of the 70x70x70 points in the grid to each of the N atoms
                # -> shape is actually (N,1,70,70,70) but the singleton dimension disappears due to np.sum(...,keepdims=False), so (N,70,70,70)
    
                #print "mol_number", mol_number
                #print "number of atoms", n_atoms[mol_number]
                mol_number +=1
                
            n_atoms = np.asarray(n_atoms, dtype=intX)
            
            
            # transfer data to memmap files http://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html
            tmp = np.memmap(prefix+'_coords.memmap', shape=coords.shape, mode='w+', dtype=floatX); tmp[:] = coords[:]; del tmp
            tmp = np.memmap(prefix+'_charges.memmap', shape=charges.shape, mode='w+', dtype=floatX); tmp[:] = charges[:]; del tmp
            tmp = np.memmap(prefix+'_vdwradii.memmap', shape=vdwradii.shape, mode='w+', dtype=floatX); tmp[:] = vdwradii[:]; del tmp
            tmp = np.memmap(prefix+'_n_atoms.memmap', shape=n_atoms.shape, mode='w+', dtype=intX); tmp[:] = n_atoms[:]; del tmp
            tmp = np.memmap(prefix+'_atom_mask.memmap', shape=atom_mask.shape, mode='w+', dtype=floatX); tmp[:] = atom_mask[:]; del tmp
            
        
        print "Total number of molecules: ", self.total_nmol
        
        # Set the grid size and resolution in Angstroms of each dimension
        grid_size=float(34.5) # FIXME choose this number such that grid coordinates are nice
        resolution=float(0.5)
        # Beginning and end coordinates, and number of intermediate points for each dimension
        # in the coordinate grid
        # this positions the center of the grid at (0,0,0)
        endx=grid_size/2
        # startx=-endx # commented out in order to have less variables (endx is used extensively)
        self.stepx=int(grid_size/resolution)+1 # +1 because N Angstroms "-" contain N+1 grid points "x": x-x-x-x-x-x-x
        # an ndarray of grid coordinates: cartesian coordinates of each voxel
        # this will be consistent across all molecules if the grid size doesn't change
        grid_coords = lasagne.utils.floatX(np.mgrid[-endx:endx:self.stepx*1j,-endx:endx:self.stepx*1j,-endx:endx:self.stepx*1j])

        # print np.bincount(n_atoms)
        
        self.min_dist_from_border = 5 # in Angstrom; for random translations; TODO ok to have it on CPU?

        # share variables (on GPU)
        # general
        self.grid_coords = self.add_param(grid_coords, grid_coords.shape , 'grid_coords', trainable=False)
        endx_on_GPU = True
        if endx_on_GPU:
            endx = np.asarray([[[endx]]], dtype=floatX) # list brackets required, otherwise error later (maybe due to array shape)
            self.endx = self.add_param(endx, endx.shape, 'endx', trainable=False)
            self.min_dist_from_border = np.asarray([[[self.min_dist_from_border]]], dtype=floatX)
            self.min_dist_from_border = self.add_param(self.min_dist_from_border, self.min_dist_from_border.shape, 'min_dist_from_border', trainable=False)
            self.endx = T.Rebroadcast((1,True),(2,True),)(self.endx)
            self.min_dist_from_border = T.Rebroadcast((1,True),(2,True),)(self.min_dist_from_border)
        else:
            self.endx = endx # TODO ok to have it on CPU?
        # instance-specific
        self.minibatch_size = minibatch_size
        self.active_or_inactive = active_or_inactive # to know which row of the inputs (molecule numbers) to use
        # .sdf-file-contents-specific
        self.coords = self.add_param(coords, coords.shape , 'coords', trainable=False)
        self.charges = self.add_param(charges, charges.shape, 'charges', trainable=False)
        self.vdwradii = self.add_param(vdwradii, vdwradii.shape, 'vdwradii', trainable=False)
        self.n_atoms = self.add_param(n_atoms, n_atoms.shape, 'n_atoms', trainable=False)
        self.atom_mask = self.add_param(atom_mask, atom_mask.shape, 'atom_mask', trainable=False)



    def get_output_shape_for(self, input_shape):
        return (self.minibatch_size, 2, self.stepx, self.stepx, self.stepx)


    def get_output_for(self, molecule_numbers01, **kwargs):
        
        def getshape(X):
            try:
                return X.shape.eval({molecule_numbers01 : np.require(((1,)*self.minibatch_size,(2,)*self.minibatch_size), dtype=intX)})
            except theano.compile.function_module.UnusedInputError:
                return X.shape.eval()

        def getval(X):
            try:
                return X.eval({molecule_numbers01 : np.require(((1,)*self.minibatch_size,(2,)*self.minibatch_size), dtype=intX)})
            except theano.compile.function_module.UnusedInputError:
                return X.eval()

        
        # select relevant molecule number, depending on active_or_inactive
        molecule_number = molecule_numbers01[self.active_or_inactive]
        # see also stackoverflow.com/questions/2640147
        
        random_streams = theano.sandbox.rng_mrg.MRG_RandomStreams() # dynamic generation of values on GPU

        # random rotation matrix Q
        # randn_matrix = lasagne.random.get_rng().normal(0, 1, size=(3,3)).astype(floatX) # fixed value
        randn_matrix = random_streams.normal((3,3), dtype=floatX) # dynamic generation of values on GPU
        Q, R = Tnlin.qr(randn_matrix) # see Golkov MSc thesis, Lemma 1
        # Mezzadri 2007 "How to generate random matrices from the classical compact groups"
        Q = T.dot(Q, Tnlin.AllocDiag()(T.sgn(R.diagonal()))) # stackoverflow.com/questions/30692742
        Q = Q * Tnlin.Det()(Q) # stackoverflow.com/questions/30132036

        # apply rotation matrix to coordinates of current molecule
        coords_current = T.dot(self.coords[molecule_number], Q)
        
        
        # random translation
        random_translations = True
        print "random_translations:", random_translations
        if random_translations:
            coords_min = T.min(coords_current, axis=1, keepdims=True)
            coords_max = T.max(coords_current, axis=1, keepdims=True)
            # order of summands important, otherwise error (maybe due to broadcastable properties)
            transl_min = (-self.endx + self.min_dist_from_border) - coords_min
            transl_max = (self.endx - self.min_dist_from_border) - coords_max
            rand01 = random_streams.uniform((self.minibatch_size,1,3), dtype=floatX) # unifom random in open interval ]0;1[
            rand01 = T.Rebroadcast((1,True),)(rand01)
            rand_translation = rand01*(transl_max-transl_min)+transl_min
            # coords_current = coords_current + rand01 # FIXME
            coords_current = coords_current + rand_translation
            # coords_current = T.add(coords_current, rand_translation)
            # reshape from (1,135,3) to (135, 3)
            # coords_current = coords_current[0,:,:,:]
            # print "coords_current", getshape(coords_current)
        
        
        #coords_current = T.Rebroadcast((0,True),(1,True),(2,True),(3,True),(4,True),(5,True),)(coords_current)

        # select subarray for current molecule; extend to 5D using `None`
        cha = self.charges[molecule_number,:,None,None,None]
        vdw = self.vdwradii[molecule_number,:,None,None,None]
        ama = self.atom_mask[molecule_number,:,None,None,None]

        if self.minibatch_size==1:
            # minibatch size 1: ignore trailing empty molecule slots
            # Apparently theano cannot select along several dimensions at the same time (maybe due to ndim of index vars?)
            # (although `self.vdwradii[molecule_number,T.arange(natoms),None,None,None]` does work within T.maximum),
            # so here is the second step:
            natoms = self.n_atoms[molecule_number[0]] # [0] for correct number of dimensions of natoms (such that T.arange(natoms) works); only for minibatch_size==1
            cha = cha[:,T.arange(natoms),:,:,:]
            vdw = vdw[:,T.arange(natoms),:,:,:]
            ama = ama[:,T.arange(natoms),:,:,:]
            # coords_current = coords_current[0,:,:] # reshape from (1,135,3) to (135, 3)
            coords_current = coords_current[:,T.arange(natoms),:]


        # TODO make self.grid_coords and others as required from the start instead of reshaping here (delete memmap files first)
        
        # pairwise distances from all atoms to all grid points
        distances = T.sqrt(T.sum((self.grid_coords[None,None,:,:,:,:] - coords_current[:,:,:,None,None,None])**2, axis=2))
        
        # "distance" from atom to grid point should never be smaller than the vdw radius of the atom (otherwise infinite proximity possible)
        distances_esp_cap = T.maximum(distances, vdw)
        
        # TODO: if MBsize>1: [grids with T.switch], else: [grids without T.switch but with the above "relics" - works faster]
        
        # grids_0: electrostatic potential in each of the 70x70x70 grid points
        # (sum over all N atoms, i.e. axis=0, so that shape turns from (N,1,70,70,70) to (1,1,70,70,70))
        # keepdims so that we have (1,70,70,70) instead of (70, 70, 70)
        # grids_1: vdw value in each of the 70x70x70 grid points (sum over all N atoms, i.e. axis=0, so that shape turns from (N,1,70,70,70) to (1,1,70,70,70))
        if self.minibatch_size==1:
            # For `minibatch_size==1`, the selection via `T.arange(natoms)` above is a faster alternative to `T.swich`.
            grids_0 = T.sum(cha/distances_esp_cap, axis=1, keepdims=True)
            grids_1 = T.sum(T.exp((-distances**2)/vdw**2), axis=1, keepdims=True)
        else:
            # T.switch: stackoverflow.com/questions/26621341
            grids_0 = T.sum(T.switch(ama, cha/distances_esp_cap, 0), axis=1, keepdims=True)        
            grids_1 = T.sum(T.switch(ama, T.exp((-distances**2)/vdw**2), 0), axis=1, keepdims=True)
        
        grids = T.concatenate([grids_0, grids_1], axis=1)
        
        # TODO is T.set_subtensor faster than T.concatenate?

        #print "grids: ", getshape(self.grids)
        return grids

