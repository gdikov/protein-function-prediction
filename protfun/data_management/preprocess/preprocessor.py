import os
import colorlog as log
import numpy as np
import csv
import pickle
import StringIO


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
            log.warning("Could not read PDB file.")
            return None

        if mol is None:
            log.warning("Bad pdb file found.")
            return None

        try:
            # add missing hydrogen atoms
            mol = rdMO.AddHs(mol, addCoords=True)

            # compute partial charges
            rdPC.ComputeGasteigerCharges(mol, throwOnParamFailure=True)
        except ValueError:
            log.warning("Bad Gasteiger charge evaluation.")
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
        res = {
            "coords": np.asarray([get_coords(i) for i in range(0, atoms_count)]) - np.asarray(
                [center.x, center.y, center.z]),
            "charges": np.asarray([float(atom.GetProp("_GasteigerCharge")) for atom in atoms]),
            "vdwradii": np.asarray([self.periodic_table.GetRvdw(atom.GetAtomicNum()) for atom in atoms]),
            "atoms_count": atoms_count
        }
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


class Preprocessor():
    def __init__(self, protein_codes, data_path):
        if isinstance(protein_codes, dict):
            self.label_type = 'enzyme_categorical'
        else:
            self.label_type = 'gene_ontological'
        self.prot_codes = protein_codes
        self.data_dir = data_path

    def process(self):
        """
        Does pre-processing of the downloaded PDB files.
        numpy.memmap's are created for molecules (from the PDB files with no errors)
        """

        molecule_processor = MoleculeProcessor()
        go_processor = GeneOntologyProcessor()

        molecules = list()
        go_targets = list()

        # also create a list of all deleted files to be inspected manually later
        erroneous_pdb_files = []

        # process all PDB codes, use [:] trick to create a copy for the iteration,
        # as removing is not allowed during iteration
        if isinstance(self.prot_codes, dict):
            valid_codes = dict()
        else:
            valid_codes = []
            raise NotImplementedError

        if self.label_type == 'enzyme_categorical':
            for cls in self.prot_codes.keys():
                valid_codes[cls] = []
                for pc in self.prot_codes[cls]:
                    f_path = os.path.join(self.data_dir, 'pdb' + pc.lower() + '.ent')
                    # process molecule from file
                    mol = molecule_processor.process_molecule(f_path)
                    if mol is None:
                        log.warning("Ignoring PDB file {} for invalid molecule".format(pc))
                        erroneous_pdb_files.append((f_path, "invalid molecule"))
                        # self.prot_codes.remove(pc)
                        continue
                    molecules.append(mol)
                    valid_codes[cls].append(pc)
        else:
            valid_codes = []
            for pc in self.prot_codes:
                f_path = os.path.join(self.data_dir, 'pdb' + pc.lower() + '.ent')
                # process molecule from file
                mol = molecule_processor.process_molecule(f_path)
                if mol is None:
                    log.warning("Ignoring PDB file {} for invalid molecule".format(pc))
                    erroneous_pdb_files.append((f_path, "invalid molecule"))
                    continue

                # process gene ontology (GO) target label from file
                if self.label_type == 'gene_ontological':
                    go_ids = go_processor.process_gene_ontologies(f_path)
                    if go_ids is None or len(go_ids) == 0:
                        log.warning("Ignoring PDB file %s because it has no gene ontologies associated with it." % pc)
                        erroneous_pdb_files.append((pc, "no associated gene ontologies"))
                        continue
                    go_targets.append(go_ids)
                molecules.append(mol)
                valid_codes.append(pc)

        return valid_codes

        # if self.label_type == 'protein_geneontological':
        #     # save the final GO targets into a .csv file
        #     with open(os.path.join(self.go_dir, "go_ids.csv"), "wb") as f:
        #         csv.writer(f).writerows(go_targets)
        #
        # n_atoms = np.array([mol["atoms_count"] for mol in molecules])
        # max_atoms = n_atoms.max()
        # molecules_count = len(molecules)
        #
        # # save the error pdb files log
        # with open(os.path.join(self.pdb_dir, "erroneous_pdb_files.log"), "wb") as f:
        #     for er in erroneous_pdb_files:
        #         f.write(str(er) + "\n")
        #
        # # save the correctly preprocessed enzymes
        # with open(self.enz_dir + "/preprocessed_enzymes.pickle", "wb") as f:
        #     pickle.dump(self.prot_codes, f)
        #
        # # after pre-processing, the PDB files should match the final molecules
        # assert molecules_count == len(self.prot_codes), "incorrect number of processed proteins: {} vs. {}".format(
        #     molecules_count, len(self.prot_codes))
        #
        # # create numpy arrays for the final data
        # coords = np.zeros(shape=(molecules_count, max_atoms, 3), dtype=floatX)
        # charges = np.zeros(shape=(molecules_count, max_atoms), dtype=floatX)
        # vdwradii = np.ones(shape=(molecules_count, max_atoms), dtype=floatX)
        # atom_mask = np.zeros(shape=(molecules_count, max_atoms), dtype=floatX)
        #
        # for i, mol in enumerate(molecules):
        #     coords[i, 0:mol["atoms_count"]] = mol["coords"]
        #     charges[i, 0:mol["atoms_count"]] = mol["charges"]
        #     vdwradii[i, 0:mol["atoms_count"]] = mol["vdwradii"]
        #     atom_mask[i, 0:mol["atoms_count"]] = 1
        #
        # n_atoms = np.asarray(n_atoms, dtype=intX)
        #
        # # save the final molecules into memmap files
        # def save_to_memmap(filename, data, dtype):
        #     tmp = np.memmap(filename, shape=data.shape, mode='w+', dtype=dtype)
        #     log.info("Saving memmap. Shape of {0} is {1}".format(filename, data.shape))
        #     tmp[:] = data[:]
        #     tmp.flush()
        #     del tmp
        #
        # save_to_memmap(os.path.join(self.memmap_dir, 'max_atoms.memmap'), np.asarray([max_atoms], dtype=intX),
        #                dtype=intX)
        # save_to_memmap(os.path.join(self.memmap_dir, 'coords.memmap'), coords, dtype=floatX)
        # save_to_memmap(os.path.join(self.memmap_dir, 'charges.memmap'), charges, dtype=floatX)
        # save_to_memmap(os.path.join(self.memmap_dir, 'vdwradii.memmap'), vdwradii, dtype=floatX)
        # save_to_memmap(os.path.join(self.memmap_dir, 'n_atoms.memmap'), n_atoms, dtype=intX)
        # save_to_memmap(os.path.join(self.memmap_dir, 'atom_mask.memmap'), atom_mask, dtype=floatX)

def create_memmaps_for_enzymes(enzyme_dir, moldata_dir, pdb_dir):
    def save_to_memmap(filename, data, dtype):
        tmp = np.memmap(filename, shape=data.shape, mode='w+', dtype=dtype)
        log.info("Saving memmap. Shape of {0} is {1}".format(filename, data.shape))
        tmp[:] = data[:]
        tmp.flush()
        del tmp

    # import theano
    floatX = np.float32 #theano.config.floatX

    # For each enzyme in enzymes dir, create a memmap file in moldata taking the info from the pdb_dir
    leaf_classes = [x for x in os.listdir(enzyme_dir) if x.endswith('.proteins')]
    for cls in leaf_classes:
        path_to_cls = os.path.join(enzyme_dir, cls)
        with open(path_to_cls, 'r') as f:
            prot_codes_in_cls = [pc.strip() for pc in f.readlines()]
            for pc in prot_codes_in_cls:
                path_to_pdb = os.path.join(pdb_dir, 'pdb' + pc.lower() + '.ent')
                enzyme_memmap_filename = os.path.join(moldata_dir, pc.upper() + '.memmap')
                # TODO: memmap the pdb
                data = np.array([])
                save_to_memmap(enzyme_memmap_filename, data=data, dtype=floatX)