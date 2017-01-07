import abc
import os
import colorlog as log
import numpy as np
import csv
import StringIO
import theano
import lasagne

from protfun.layers import MoleculeMapLayer

floatX = theano.config.floatX
intX = np.int32


class DataProcessor(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, from_dir, target_dir):
        self.from_dir = from_dir
        self.target_dir = target_dir

    @abc.abstractmethod
    def process(self):
        raise NotImplementedError


class EnzymeDataProcessor(DataProcessor):
    """
    Enzyme protein data processor.

    Does pre-processing of the downloaded PDB files.
    numpy.memmap's are created for molecules (from the PDB files with no errors)
    """

    def __init__(self, from_dir, target_dir, protein_codes, process_grids=True, process_memmaps=True,
                 force_recreate=False):
        super(EnzymeDataProcessor, self).__init__(from_dir=from_dir, target_dir=target_dir)
        self.prot_codes = protein_codes
        self.process_grids = process_grids
        self.process_memmaps = process_memmaps
        self.force_recreate = force_recreate
        self.molecule_processor = PDBMoleculeProcessor()
        self.grid_processor = GridProcessor()

    def process(self):
        # will store the valid proteins for each enzyme class, which is the key in the dict()
        valid_codes = dict()

        for cls in self.prot_codes.keys():
            valid_codes[cls] = []
            for pc in self.prot_codes[cls]:
                prot_dir = os.path.join(self.target_dir, pc.upper())
                f_path = os.path.join(self.from_dir, pc.upper(), 'pdb' + pc.lower() + '.ent')

                # if required, process the memmaps for the molecules again
                if self.process_memmaps and (not self.memmaps_exists(prot_dir) or self.force_recreate):
                    # attempt to process the molecule from the PDB file
                    mol = self.molecule_processor.process_molecule(f_path)
                    if mol is None:
                        log.warning("Ignoring PDB file {} for invalid molecule".format(pc))
                        continue

                    # persist the molecule and add the resulting memmaps to mol_info if processing was successful
                    self._persist_processed(prot_dir=prot_dir, mol=mol)

                # if required, process the molecule grids as well
                if self.process_grids and (not self.grid_exists(prot_dir) or self.force_recreate):
                    grid = self.grid_processor.process(prot_dir)
                    if grid is None:
                        log.warning("Ignoring PDB file {}, grid could not be processed".format(pc))
                        continue
                    if not os.path.exists(prot_dir):
                        os.makedirs(prot_dir)
                    self.save_to_memmap(file_path=os.path.join(prot_dir, "grid.memmap"), data=grid, dtype=floatX)

                # copy the PDB file to the target directory
                os.system("cp %s %s" % (f_path, os.path.join(prot_dir, 'pdb' + pc.lower() + '.ent')))

                valid_codes[cls].append(pc)

        return valid_codes

    def _persist_processed(self, prot_dir, mol):
        if not os.path.exists(prot_dir):
            os.makedirs(prot_dir)
        # generate and save the memmaps
        coords = mol["coords"]
        charges = mol["charges"]
        vdwradii = mol["vdwradii"]

        self.save_to_memmap(os.path.join(prot_dir, 'coords.memmap'),
                            coords, dtype=floatX)
        self.save_to_memmap(os.path.join(prot_dir, 'charges.memmap'),
                            charges, dtype=floatX)
        self.save_to_memmap(os.path.join(prot_dir, 'vdwradii.memmap'),
                            vdwradii, dtype=floatX)

    @staticmethod
    def save_to_memmap(file_path, data, dtype):
        tmp = np.memmap(file_path, shape=data.shape, mode='w+', dtype=dtype)
        log.info("Saving memmap. Shape of {0} is {1}".format(file_path, data.shape))
        tmp[:] = data[:]
        tmp.flush()
        del tmp

    @staticmethod
    def memmaps_exists(prot_dir):
        return os.path.exists(os.path.join(prot_dir, 'coords.memmap')) and \
               os.path.exists(os.path.join(prot_dir, 'charges.memmap')) and \
               os.path.exists(os.path.join(prot_dir, 'vdwradii.memmap'))

    @staticmethod
    def grid_exists(prot_dir):
        return os.path.exists(os.path.join(prot_dir, "grid.memmap"))


class GODataProcessor(DataProcessor):
    """
    Gene ontology data processor
    """

    def __init__(self, from_dir, target_dir):
        super(GODataProcessor, self).__init__(from_dir=from_dir, target_dir=target_dir)
        self.molecule_processor = PDBMoleculeProcessor()
        self.go_processor = GeneOntologyProcessor()

    def process(self):
        # valid_codes = []
        # for pc in self.prot_codes:
        #     f_path = os.path.join(self.data_dir, 'pdb' + pc.lower() + '.ent')
        #     # process molecule from file
        #     mol = molecule_processor.process_molecule(f_path)
        #     if mol is None:
        #         log.warning("Ignoring PDB file {} for invalid molecule".format(pc))
        #         erroneous_pdb_files.append((f_path, "invalid molecule"))
        #         continue
        #
        #     # process gene ontology (GO) target label from file
        #     if self.label_type == 'gene_ontological':
        #         go_ids = go_processor.process_gene_ontologies(f_path)
        #         if go_ids is None or len(go_ids) == 0:
        #             log.warning("Ignoring PDB file %s because it has no gene ontologies associated with it." % pc)
        #             erroneous_pdb_files.append((pc, "no associated gene ontologies"))
        #             continue
        #         go_targets.append(go_ids)
        #     molecules.append(mol)
        #     valid_codes.append(pc)
        #
        #     # save the final GO targets into a .csv file
        #     with open(os.path.join(self.go_dir, "go_ids.csv"), "wb") as f:
        #         csv.writer(f).writerows(go_targets)
        raise NotImplementedError


class PDBMoleculeProcessor(object):
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


class GridProcessor(object):
    def __init__(self):
        dummy_coords_input = lasagne.layers.InputLayer(shape=(1, None, None))
        dummy_charges_input = lasagne.layers.InputLayer(shape=(1, None))
        dummy_vdwradii_input = lasagne.layers.InputLayer(shape=(1, None))
        dummy_natoms_input = lasagne.layers.InputLayer(shape=(1,))

        self.processor = MoleculeMapLayer(incomings=[dummy_coords_input, dummy_charges_input,
                                                     dummy_vdwradii_input, dummy_natoms_input],
                                          minibatch_size=1,
                                          rotate=False)

    def process(self, prot_dir):
        try:
            coords = np.memmap(os.path.join(prot_dir, 'coords.memmap'), mode='r', dtype=floatX).reshape((1, -1, 3))
            charges = np.memmap(os.path.join(prot_dir, 'charges.memmap'), mode='r', dtype=floatX).reshape((1, -1))
            vdwradii = np.memmap(os.path.join(prot_dir, 'vdwradii.memmap'), mode='r', dtype=floatX).reshape((1, -1))
            n_atoms = np.array(coords.shape[1], dtype=intX).reshape((1,))
        except IOError:
            return None
        mol_info = [theano.shared(coords),
                    theano.shared(charges),
                    theano.shared(vdwradii),
                    theano.shared(n_atoms)]
        grid = self.processor.get_output_for(mols_info=mol_info).eval()
        return grid
