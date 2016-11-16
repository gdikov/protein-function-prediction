import os
import csv
import StringIO
import numpy as np
import theano

floatX = theano.config.floatX
intX = np.int32  # FIXME is this the best choice? (changing would require removing and recreating memmap files)


class DataSetup(object):
    """
    Sets up the data set by downloading PDB proteins and doing initial processing into memmaps.
    """

    def __init__(self, foldername='data', download_again=False, process_again=True, prot_codes=None, split_test=0.1):
        """

        :param foldername: the directory that will contain the data set
        :param update: whether the data set should be updated (downloaded again & memmaps generated).
        :param prot_codes: which protein codes the dataset should contain. Only makes sense if redownload=True.
        :param split_test: ration of training vs. test data
        """

        self.data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../", foldername)
        self.pdb_dir = os.path.join(self.data_dir, "pdb")
        self.go_dir = os.path.join(self.data_dir, "go")
        self.memmap_dir = os.path.join(self.data_dir, "moldata")

        if not os.path.exists(self.pdb_dir):
            os.makedirs(self.pdb_dir)
        if not os.path.exists(self.go_dir):
            os.makedirs(self.go_dir)
        if not os.path.exists(self.memmap_dir):
            os.makedirs(self.memmap_dir)

        self.prot_codes = prot_codes
        self.test_train_ratio = split_test
        self.pdb_files = []
        self._setup(download_again, process_again)

    def _setup(self, download_again, process_again):
        if download_again:
            print("INFO: Proceeding to download the Protein Data Base...")
            self._download_dataset()
        else:
            # checking for pdb related files
            pdb_list = [f for f in os.listdir(self.pdb_dir) if
                        f.endswith('.gz') or f.endswith('.pdb') or f.endswith('.ent')]
            if not pdb_list:
                print("WARNING: %s does not contain any PDB files. " % self.pdb_dir +
                      "Run the DataSetup with update=True to download them.")
            self.pdb_files = [os.path.join(self.pdb_dir, f) for f in pdb_list]

        if process_again:
            print("INFO: Creating molecule data memmap files...")
            self._preprocess_dataset()
        else:
            # checking for molecule data memmaps
            memmap_list = [f for f in os.listdir(self.memmap_dir) if f.endswith('.memmap')]
            if not memmap_list:
                print("WARNING: %s does not contain any memmap files. " % self.pdb_dir +
                      "Run the DataSetup with update=True to recreate them.")

    def _download_dataset(self):
        """
        Downloads the PDB database (or a part of it) as PDB files.
        """
        # TODO: control the number of molecules to download if the entire DB is too large
        # download the Protein Data Base
        from Bio.PDB import PDBList
        pl = PDBList(pdb=self.pdb_dir)
        pl.flat_tree = 1
        if self.prot_codes is not None:
            failed = 0
            attempted = len(self.prot_codes)
            for code in self.prot_codes:
                try:
                    pl.retrieve_pdb_file(pdb_code=code)
                except IOError:
                    failed += 1
                    continue
            print("INFO: Downloaded {0}/{1} molecules".format(attempted-failed, attempted))
        else:
            pl.download_entire_pdb()

        self.pdb_files = [os.path.join(self.pdb_dir, f) for f in os.listdir(self.pdb_dir)
                          if f.endswith(".ent") or f.endswith(".pdb")]

    def _preprocess_dataset(self):
        """
        Does pre-processing of the downloaded PDB files.
        numpy.memmap's are created for molecules (from the PDB files with no errors)
        A .csv file is created with all GO (Gene Ontology) IDs for the processed molecules.
        :return:
        """

        molecule_processor = MoleculeProcessor()
        go_processor = GeneOntologyProcessor()

        molecules = list()
        go_targets = list()

        # also create a list of all deleted files to be inspected manually later
        erroneous_pdb_files = []

        # process all PDB files
        for f in self.pdb_files[:]:
            # process molecule from file
            mol = molecule_processor.process_molecule(f)
            if mol is None:
                print("INFO: removing PDB file %s for invalid molecule" % f)
                # remove from disk as it could be miscounted later if setup() is called with update=False
                erroneous_pdb_files.append((f, "invalid molecule"))
                os.remove(f)
                self.pdb_files.remove(f)
                continue

            # process gene ontology (GO) target label from file
            go_ids = go_processor.process_gene_ontologies(f)
            if go_ids is None or len(go_ids) == 0:
                print("INFO: removing PDB file %s because it has no gene ontologies associated with it." % f)
                erroneous_pdb_files.append((f, "no associated gene ontologies"))
                os.remove(f)
                self.pdb_files.remove(f)
                continue

            molecules.append(mol)
            go_targets.append(go_ids)

        n_atoms = np.array([mol["atoms_count"] for mol in molecules])
        max_atoms = n_atoms.max()
        molecules_count = len(molecules)

        # save the error pdb files log
        with open(os.path.join(self.pdb_dir, "erroneous_pdb_files.log"), "wb") as f:
            for er in erroneous_pdb_files:
                f.write(str(er) + "\n")

        # after pre-processing, the PDB files should match the final molecules
        assert molecules_count == len(self.pdb_files)

        # create numpy arrays for the final data
        coords = np.zeros(shape=(molecules_count, max_atoms, 3), dtype=floatX)
        charges = np.zeros(shape=(molecules_count, max_atoms), dtype=floatX)
        vdwradii = np.ones(shape=(molecules_count, max_atoms), dtype=floatX)
        atom_mask = np.zeros(shape=(molecules_count, max_atoms), dtype=floatX)

        for i, mol in enumerate(molecules):
            coords[i, 0:mol["atoms_count"]] = mol["coords"]
            charges[i, 0:mol["atoms_count"]] = mol["charges"]
            vdwradii[i, 0:mol["atoms_count"]] = mol["vdwradii"]
            atom_mask[i, 0:mol["atoms_count"]] = 1

        n_atoms = np.asarray(n_atoms, dtype=intX)

        # save the final molecules into memmap files
        def save_to_memmap(filename, data, dtype):
            tmp = np.memmap(filename, shape=data.shape, mode='w+', dtype=dtype)
            print("INFO: Saving memmap. Shape of {0} is {1}".format(filename, data.shape))
            tmp[:] = data[:]
            tmp.flush()
            del tmp

        save_to_memmap(os.path.join(self.memmap_dir, 'max_atoms.memmap'), np.asarray([max_atoms], dtype=intX),
                       dtype=intX)
        save_to_memmap(os.path.join(self.memmap_dir, 'coords.memmap'), coords, dtype=floatX)
        save_to_memmap(os.path.join(self.memmap_dir, 'charges.memmap'), charges, dtype=floatX)
        save_to_memmap(os.path.join(self.memmap_dir, 'vdwradii.memmap'), vdwradii, dtype=floatX)
        save_to_memmap(os.path.join(self.memmap_dir, 'n_atoms.memmap'), n_atoms, dtype=intX)
        save_to_memmap(os.path.join(self.memmap_dir, 'atom_mask.memmap'), atom_mask, dtype=floatX)

        # save the final GO targets into a .csv file
        with open(os.path.join(self.go_dir, "go_ids.csv"), "wb") as f:
            csv.writer(f).writerows(go_targets)

    def load_dataset(self):

        print("INFO: Loading total of {0} pdb files.".format(len(self.pdb_files)))

        data_size = len(self.pdb_files)

        # split into test and training data
        test_ids = np.random.randint(0, data_size, int(self.test_train_ratio * data_size))
        # get all but the indices of the test_data
        train_ids = np.setdiff1d(np.arange(data_size), test_ids, assume_unique=True)

        labels, go_dict_id2name = self._load_labels()

        assert labels.shape[0] == data_size, "labels count %d != molecules count %d" % (labels.shape[0], data_size)

        # for the sake of completion, generate prot_id2name dictionary
        prot_dict_id2name = {prot_id: prot_name[-8:-4] for prot_id, prot_name in enumerate(self.pdb_files)}

        validation_portion = train_ids.size / 5
        data_dict = {'x_id2name': prot_dict_id2name, 'y_id2name': go_dict_id2name,
                     'x_train': train_ids[validation_portion:], 'y_train': labels[train_ids[validation_portion:]],
                     'x_val': train_ids[:validation_portion], 'y_val': labels[train_ids[:validation_portion]],
                     'x_test': test_ids, 'y_test': labels[test_ids]}

        print("INFO: Train and test data loaded")

        return data_dict

    def _load_labels(self, label_type='enzyme_classes'):
        """ call the corresponding label generating function.
        :return: label matrix, id2name dictionary for label decoding
        """
        if label_type == 'enzyme_classes':
            return self._load_enz_labels()
        elif label_type == 'gene_onotlogies':
            return self._load_go_labels()
        else:
            print("ERROR: Unknown label_type argument value")
            raise ValueError

    def _load_enz_labels(self):
        """ find if each protein belongs to one of the classes
        :returns: binary matrix with samples as rows and class association as columns,
        dictionary which decodes the column id to class name."""
        path_to_enz = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/enzymes/3_4_21.labels")
        with open(path_to_enz, 'r') as f:
            class21 = set([e.strip() for e in f.readlines()])
        path_to_enz = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/enzymes/3_4_24.labels")
        with open(path_to_enz, 'r') as f:
            class24 = set([e.strip() for e in f.readlines()])

        label_dict21 = np.array([int(x[-8:-4] in class21) for x in self.pdb_files])
        label_dict24 = np.array([int(x[-8:-4] in class24) for x in self.pdb_files])
        # the id2name dictionary here represents the column id-class mapping
        return np.vstack((label_dict21, label_dict24)).T, {0:"class_21", 1:"class_24"}

    def _load_go_labels(self):
        """ find the number of different GO Ids and
        create a binary matrix with rows representing different proteins and columns all the gene ontology terms
        :return: label matrix, id2name dictionary for label decoding"""

        go_ids = set()
        with open(os.path.join(self.go_dir, "go_ids.csv"), 'r') as gene_ontologies:
            gos_all_mols = csv.reader(gene_ontologies)
            for gos_per_mol in gos_all_mols:
                go_ids.update(gos_per_mol)

        go_name2id = dict(zip(go_ids, np.arange(len(go_ids))))
        prot_gos_matrix = np.zeros((len(self.pdb_files), len(go_ids)))

        # csv.reader return a iterator so we need to call it again along with the file opening
        with open(os.path.join(self.go_dir, "go_ids.csv"), 'r') as gene_ontologies:
            gos_all_mols = csv.reader(gene_ontologies)
            for prot_id, gos_per_mol in enumerate(gos_all_mols):
                prot_gos_matrix[prot_id, np.asarray([go_name2id[go] for go in gos_per_mol])] = 1

        go_id2name = {y: x for x, y in go_name2id.iteritems()}

        return prot_gos_matrix, go_id2name

class MoleculeProcessor(object):
    """
    MoleculeProcessor can produce a ProcessedMolecule from the contents of a PDB file.
    """

    def __init__(self):
        import rdkit.Chem as Chem
        self.periodic_table = Chem.GetPeriodicTable()

    def process_molecule(self, pdb_file):
        """
        Processes a molecule from the passed PDB file if the file contents has no errors.
        :param pdb_file: path to the PDB file to process the molecule from.
        :return: a ProcessedMolecule object
        """
        import rdkit.Chem as Chem
        import rdkit.Chem.rdPartialCharges as rdPC
        import rdkit.Chem.rdMolTransforms as rdMT
        import rdkit.Chem.rdmolops as rdMO

        # read a molecule from the PDB file
        try:
            mol = Chem.MolFromPDBFile(molFileName=pdb_file, removeHs=False, sanitize=True)
        except IOError:
            print("WARNING: Could not read PDB file.")
            return None

        if mol is None:
            print("WARNING: Bad pdb file found.")
            return None

        try:
            # add missing hydrogen atoms
            rdMO.AddHs(mol, addCoords=True)

            # compute partial charges
            rdPC.ComputeGasteigerCharges(mol, throwOnParamFailure=True)
        except ValueError:
            print("WARNING: Bad Gasteiger charge evaluation.")
            return None

        # get the conformation of the molecule
        conformer = mol.GetConformer()

        # calculate the center of the molecule
        center = rdMT.ComputeCentroid(conformer, ignoreHs=False)

        atoms_count = mol.GetNumAtoms()
        atoms = mol.GetAtoms()

        def get_coords(i):
            coord = conformer.GetAtomPosition(i)
            return np.asarray([coord.x, coord.y, coord.z])

        # set the coordinates, charges, VDW radii and atom count
        res = {}
        res["coords"] = np.asarray(
            [get_coords(i) for i in range(0, atoms_count)]) - np.asarray(
            [center.x, center.y, center.z])

        res["charges"] = np.asarray(
            [float(atom.GetProp("_GasteigerCharge")) for atom in atoms])

        res["vdwradii"] = np.asarray([self.periodic_table.GetRvdw(atom.GetAtomicNum()) for atom in atoms])

        res["atoms_count"] = atoms_count
        return res


class GeneOntologyProcessor(object):
    """
    GeneOntologyProcessor can read a list of GO (Gene Ontology) from a PDB file.
    """

    def process_gene_ontologies(self, pdb_file):
        """
        Processes a PDB file and returns a list with GO ids that can be associated with it.
        :param pdb_file: the path to the PDB file that is to be processed.
        :return: a list of GO ids for the molecule contained in the PDB file.
        """
        from prody.proteins.header import parsePDBHeader
        import requests

        polymers = parsePDBHeader(pdb_file, "polymers")
        uniprot_ids = set()
        for polymer in polymers:
            for dbref in polymer.dbrefs:
                if dbref.database == "UniProt":
                    uniprot_ids.add(dbref.accession)

        go_ids = []
        for uniprot_id in uniprot_ids:
            url = "http://www.ebi.ac.uk/QuickGO/GAnnotation?protein=" + uniprot_id + "&format=tsv"
            response = requests.get(url)
            go_ids += self._parse_gene_ontology(response.text)

        return go_ids

    @staticmethod
    def _parse_gene_ontology(tsv_text):
        f = StringIO.StringIO(tsv_text)
        reader = csv.reader(f, dialect="excel-tab")
        # skip the header
        next(reader)
        try:
            return zip(*[line for line in reader])[6]
        except IndexError:
            # protein has no GO terms associated with it
            return ["unknown"]
