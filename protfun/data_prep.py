import os
import csv
import StringIO
import numpy as np
import theano

floatX = theano.config.floatX
intX = np.int32  # FIXME is this the best choice? (changing would require removing and recreating memmap files)


class DataSetup():
    """ safemode should prevent redoing time consuming tasks such as downloading and memmapping
    the PDB and GOs. If turned off it will overwrite the data. """
    def __init__(self, foldername='data', safemode=True, prot_codes=None, split_test=0.1):

        self.safemode = safemode

        path_to_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
        self.foldername = os.path.join(path_to_data, foldername)

        self.num_gene_ontologies = 0
        self.prot_codes = prot_codes
        self.split_test = split_test

        self.pdb_files = []
        self._setup()

    def _setup(self):

        pdb_dir = os.path.join(self.foldername, "pdb")
        go_dir = os.path.join(self.foldername, "go")
        moldata_dir = os.path.join(self.foldername, "moldata")

        if not os.path.exists(pdb_dir):
            os.makedirs(pdb_dir)
        if not os.path.exists(go_dir):
            os.makedirs(go_dir)
        if not os.path.exists(moldata_dir):
            os.makedirs(moldata_dir)

        if self.safemode:
            # checking for pdb related files
            filelist = [f for f in os.listdir(pdb_dir)
                        if f.endswith('.gz') or f.endswith('.pdb') or f.endswith('.ent')]
            if filelist:
                print("INFO: Safemode is ON and the Protein Data Base seems to be downloaded. "
                      "Skipping download.")
            else:
                print("INFO: Proceeding to download the Protein Data Base...")
                self._download_dataset(pdb_dir=pdb_dir, ontologies_dir=go_dir, pdb_codes=self.prot_codes)

            # checking for molecule data memmaps
            filelist = [f for f in os.listdir(moldata_dir) if f.endswith('.memmap')]
            if filelist:
                print("INFO: Safemode is ON and the molecule data files (memmap) seem to be there. "
                      "Skipping molecule info memmap generation.")
            else:
                self._store_molecule_info(moldata_dir=moldata_dir, sanitize=True)

        else:
            print("INFO: Proceeding to download the Protein Data Base...")
            self._download_dataset(pdb_dir=pdb_dir, ontologies_dir=go_dir)
            print("INFO: Creating molecule data memmap files...")
            self._store_molecule_info(moldata_dir=moldata_dir, sanitize=True)

        # TODO integrate splitting in test and training datasets
        return self.num_gene_ontologies

    """ TODO add documentation """
    def _download_dataset(self, pdb_dir, ontologies_dir=None, pdb_codes=None):
        # TODO: control the number of molecules to download if the entire DB is too large
        # download the Protein Data Base
        from Bio.PDB import PDBList
        pl = PDBList(pdb=pdb_dir)
        pl.flat_tree = 1
        if pdb_codes is not None:
            for code in pdb_codes:
                pl.retrieve_pdb_file(pdb_code=code)
        else:
            pl.download_entire_pdb()

        self.pdb_files = [os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir)
                          if f.endswith(".ent") or f.endswith(".pdb")]

        # TODO remove the pdb file from self.pdb_files if get_gene_onotlogy fails somewhere
        def _get_gene_ontologies():
            from prody.proteins.header import parsePDBHeader
            import requests

            all_go_ids = []

            def parse_gene_ontology(tsv_text):
                f = StringIO.StringIO(tsv_text)
                reader = csv.reader(f, dialect="excel-tab")
                # skip the header
                next(reader)
                try:
                    return zip(*[line for line in reader])[6]
                except IndexError:
                    # protein has no GO terms associated with it
                    return ["unknown"]

            for f in self.pdb_files:
                polymers = parsePDBHeader(f, "polymers")
                uniprot_ids = set()
                for polymer in polymers:
                    for dbref in polymer.dbrefs:
                        if dbref.database == "UniProt":
                            uniprot_ids.add(dbref.accession)

                go_ids = []
                for uniprot_id in uniprot_ids:
                    url = "http://www.ebi.ac.uk/QuickGO/GAnnotation?protein=" + uniprot_id + "&format=tsv"
                    response = requests.get(url)
                    go_ids += parse_gene_ontology(response.text)

                all_go_ids.append(go_ids)

            return all_go_ids

        # download the gene ontologies for each protein
        if ontologies_dir is not None:
            with open(os.path.join(self.foldername, "go_ids.csv"), "wb") as f:
                # log the gene ontology IDs into a csv file
                gene_ontologies = _get_gene_ontologies()
                csv.writer(f).writerows(gene_ontologies)

    """TODO add documentation """
    def _store_molecule_info(self, moldata_dir, sanitize=True):

        # memmap files not found, create them
        print "INFO: Creating memmap files with molecule data..."

        import rdkit.Chem as Chem
        import rdkit.Chem.rdPartialCharges as rdPC
        import rdkit.Chem.rdMolTransforms as rdMT

        n_atoms = []

        def _get_molecules():
            import rdkit.Chem as Chem
            import rdkit.Chem.rdmolops as rdMO

            # get molecules
            # sanitize the data for incorrectly built molecules
            if sanitize:
                res = []
                for f in self.pdb_files:
                    mol_from_pdb = Chem.MolFromPDBFile(molFileName=f, removeHs=False, sanitize=True)
                    if mol_from_pdb is None:
                        self.pdb_files.remove(f)
                        print("WARNING: Bad pdb file found. Protein will be removed.")
                        continue
                    res.append(rdMO.AddHs(mol_from_pdb, addCoords=True))
            else:
                res = [Chem.MolFromPDBFile(molFileName=f, removeHs=False, sanitize=True) for f in self.pdb_files]
                res = [rdMO.AddHs(mol, addCoords=True) for mol in res if mol is not None]
            return res

        molecules = _get_molecules()

        # Periodic table object, needed for getting VDW radii
        pt = Chem.GetPeriodicTable()

        self.molecules_count = len(molecules)
        max_atoms = max([mol.GetNumAtoms() for mol in molecules])

        coords = np.zeros(shape=(self.molecules_count, max_atoms, 3), dtype=floatX)
        charges = np.zeros(shape=(self.molecules_count, max_atoms), dtype=floatX)
        vdwradii = np.ones(shape=(self.molecules_count, max_atoms), dtype=floatX)
        atom_mask = np.zeros(shape=(self.molecules_count, max_atoms), dtype=floatX)

        def _save_to_memmap(filename, data, dtype):
            tmp = np.memmap(filename, shape=data.shape, mode='w+', dtype=dtype)
            print("INFO: Saving memmap. Shape of {0} is {1}".format(filename, data.shape))
            tmp[:] = data[:]
            tmp.flush()
            del tmp

        mol_index = -1
        for mol in molecules:
            mol_index += 1
            # TODO: add sanitiziation
            # compute the atomic partial charges
            if sanitize:
                try:
                    rdPC.ComputeGasteigerCharges(mol, throwOnParamFailure=True)
                except ValueError:
                    print("WARNING: Bad Gasteiger charge evaluation. Protein will be removed.")
                    # comment this removal if molecules aren't use any more
                    molecules.remove(mol)
                    del self.pdb_files[mol_index]
                    mol_index -= 1
                    continue
            else:
                rdPC.ComputeGasteigerCharges(mol)

            # get the conformation of the molecule and number of atoms (3D coordinates)
            conformer = mol.GetConformer()

            # calculate the center of the molecule
            # Centroid is the center of coordinates (center of mass of unit-weight atoms)
            # Center of mass would require atomic weights for each atom: pt.GetAtomicWeight()
            center = rdMT.ComputeCentroid(conformer, ignoreHs=False)

            atoms_count = mol.GetNumAtoms()
            atoms = mol.GetAtoms()

            n_atoms.append(atoms_count)
            atom_mask[mol_index, 0:atoms_count] = 1

            def get_coords(i):
                coord = conformer.GetAtomPosition(i)
                return np.asarray([coord.x, coord.y, coord.z])

            # set the coordinates, charges and VDW radii
            coords[mol_index, 0:atoms_count] = np.asarray(
                [get_coords(i) for i in range(0, atoms_count)]) - np.asarray(
                [center.x, center.y, center.z])
            charges[mol_index, 0:atoms_count] = np.asarray(
                [float(atom.GetProp("_GasteigerCharge")) for atom in atoms])
            vdwradii[mol_index, 0:atoms_count] = np.asarray([pt.GetRvdw(atom.GetAtomicNum()) for atom in atoms])

        n_atoms = np.asarray(n_atoms, dtype=intX)

        _save_to_memmap(os.path.join(moldata_dir, 'max_atoms.memmap'), np.asarray([max_atoms], dtype=intX), dtype=intX)
        _save_to_memmap(os.path.join(moldata_dir, 'coords.memmap'), coords, dtype=floatX)
        _save_to_memmap(os.path.join(moldata_dir, 'charges.memmap'), charges, dtype=floatX)
        _save_to_memmap(os.path.join(moldata_dir, 'vdwradii.memmap'), vdwradii, dtype=floatX)
        _save_to_memmap(os.path.join(moldata_dir, 'n_atoms.memmap'), n_atoms, dtype=intX)
        _save_to_memmap(os.path.join(moldata_dir, 'atom_mask.memmap'), atom_mask, dtype=floatX)

        del molecules

    """TODO add documentation """
    def load_dataset(self):
        # TODO: make it work meaningfully
        data = np.random.randn(10, 10)
        labels = np.random.randn(10, 10)
        train_data_mask, test_data_mask = self._split_dataset(data)
        data_dict = {'x_train': data[train_data_mask], 'y_train': labels[train_data_mask],
                     'x_val': data[train_data_mask], 'y_val': labels[train_data_mask],
                     'x_test': data[test_data_mask], 'y_test': labels[test_data_mask]}

        print("INFO: Data loaded")
        # TODO set the num_gene_ontologies to the filtered number of different ontologies
        return data_dict, self.num_gene_ontologies

    def _split_dataset(self, data_ids=None):
        # TODO use self.split_test
        return np.random.randint(0,10,2), np.random.randint(0,10,2)
