import abc
import os
import colorlog as log
import numpy as np
import csv
import StringIO
import theano
import lasagne
import cPickle
import itertools

import prody as pd
import rdkit.Chem as Chem
import rdkit.Chem.rdPartialCharges as rdPC
import rdkit.Chem.rdMolTransforms as rdMT
import rdkit.Chem.rdmolops as rdMO

from protfun.layers import MoleculeMapLayer

floatX = theano.config.floatX
intX = np.int32
CNS = 24  # number of sidechain channels (20 amino, all, nonhydro, hydro, backbone)


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
                 force_recreate=False, add_sidechain_channels=True, use_esp=False):
        super(EnzymeDataProcessor, self).__init__(from_dir=from_dir, target_dir=target_dir)
        self.prot_codes = protein_codes
        self.process_grids = process_grids
        self.process_memmaps = process_memmaps
        self.force_recreate = force_recreate
        self.use_esp = use_esp
        self.add_sidechain_channels = add_sidechain_channels
        if add_sidechain_channels:
            self.molecule_processor = PDBSideChainProcessor()
            self.grid_processor = GridSideChainProcessor()
        else:
            self.molecule_processor = PDBMoleculeProcessor()
            self.grid_processor = GridProcessor()

    def process(self):
        # will store the valid proteins for each enzyme class, which is the key in the dict()
        valid_codes = dict()
        invalid_codes = set()
        invalid_codes_path = os.path.join(self.from_dir, 'invalid_codes.pickle')
        if os.path.exists(invalid_codes_path):
            with open(invalid_codes_path, 'r') as f:
                invalid_codes = cPickle.load(f)

        prot_codes = list(itertools.chain.from_iterable(self.prot_codes.values()))
        prot_codes = list(set(prot_codes))
        prot_codes = sorted(prot_codes)
        for i, pc in enumerate(prot_codes):
            # skip if we know this protein cannot be processed
            if pc in invalid_codes:
                continue

            prot_dir = os.path.join(self.target_dir, pc.upper())
            f_path = os.path.join(self.from_dir, pc.upper(), 'pdb' + pc.lower() + '.ent')

            # if required, process the memmaps for the protein again
            if self.process_memmaps and (not self.memmaps_exists(prot_dir,
                                                                 num_channels=CNS if self.add_sidechain_channels else 1)
                                         or self.force_recreate):
                # attempt to process the molecule from the PDB file
                mol = self.molecule_processor.process_molecule(f_path, use_esp=self.use_esp)
                if mol is None:
                    log.warning("Ignoring PDB file {} for invalid molecule".format(pc))
                    invalid_codes.add(pc)
                    continue
                # persist the molecule and add the resulting memmaps to mol_info if processing was successful
                self._persist_processed(prot_dir=prot_dir, mol=mol)
            else:
                log.info("Skipping already processed PDB file: {}".format(pc))

            # if required, process the ESP and density grids as well
            if self.process_grids and (not self.grid_exists(prot_dir) or self.force_recreate):
                grid = self.grid_processor.process(prot_dir)
                if grid is None:
                    log.warning("Ignoring PDB file {}, grid could not be processed".format(pc))
                    invalid_codes.add(pc)
                    continue
                if not os.path.exists(prot_dir):
                    os.makedirs(prot_dir)
                # persist the computed grid as a memmap file
                self.save_to_memmap(file_path=os.path.join(prot_dir, "grid.memmap"), data=grid, dtype=floatX)

            # copy the PDB file to the target directory
            if not os.path.exists(os.path.join(prot_dir, 'pdb' + pc.lower() + '.ent')):
                os.system("cp %s %s" % (f_path, os.path.join(prot_dir, 'pdb' + pc.lower() + '.ent')))

        # persist the invalid codes for next time
        with open(invalid_codes_path, 'wb') as f:
            cPickle.dump(invalid_codes, f)

        log.info("Total proteins: {} Invalid proteins: {}".format(len(prot_codes), len(invalid_codes)))
        for cls, prots in self.prot_codes.items():
            valid_codes[cls] = [pc for pc in prots if pc not in invalid_codes]

        return valid_codes

    def _persist_processed(self, prot_dir, mol):
        if not os.path.exists(prot_dir):
            os.makedirs(prot_dir)
        # generate and save the memmaps
        for key, value in mol.items():
            self.save_to_memmap(os.path.join(prot_dir, '{0}.memmap'.format(key)),
                                value, dtype=floatX)

    @staticmethod
    def save_to_memmap(file_path, data, dtype):
        if data.size == 0:
            data = np.array([np.nan])
        tmp = np.memmap(file_path, shape=data.shape, mode='w+', dtype=dtype)
        log.info("Saving memmap. Shape of {0} is {1}".format(file_path, data.shape))
        tmp[:] = data[:]
        tmp.flush()
        del tmp

    @staticmethod
    def memmaps_exists(prot_dir, num_channels=1):
        if num_channels > 1:
            if not os.path.exists(prot_dir):
                return False
            memmaps = [f for f in os.listdir(prot_dir) if f.endswith('.memmap')]
            # TODO this is hardcoded and semi-correct sanity check: refactor!
            return len([f for f in memmaps if f.startswith('coords')]) == num_channels and \
                   len([f for f in memmaps if f.startswith('vdwradii')]) == num_channels
        else:
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

    def process_molecule(self, pdb_file, use_esp=False):
        """
        Processes a molecule from the passed PDB file if the file contents has no errors.
        :param pdb_file: path to the PDB file to process the molecule from.
        :return: a ProcessedMolecule object
        """

        # TODO: this is the old code using rdkit for the charge computations. Gasteiger is an inappropriate algorithm
        # read a molecule from the PDB file
        try:
            mol = Chem.MolFromPDBFile(molFileName=pdb_file, removeHs=False, sanitize=True)
        except IOError:
            log.warning("Could not read PDB file.")
            return None

        if mol is None:
            log.warning("Bad pdb file found.")
            return None

        if use_esp:
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
            # "charges": np.asarray([float(atom.GetProp("_GasteigerCharge")) for atom in atoms]),
            "vdwradii": np.asarray([self.periodic_table.GetRvdw(atom.GetAtomicNum()) for atom in atoms]),
            "atoms_count": atoms_count
        }
        return res


class PDBSideChainProcessor(object):
    def __init__(self):
        self.periodic_table = Chem.GetPeriodicTable()

    def process_molecule(self, pdb_file, use_esp=False):
        hydro_file_name = '_hydrogenized.'.join(os.path.basename(pdb_file).split('.'))
        hydrogenized_pdb_file = os.path.join(os.path.dirname(pdb_file), hydro_file_name)
        try:
            mol_rdkit = Chem.MolFromPDBFile(molFileName=pdb_file, removeHs=False, sanitize=True)
            if mol_rdkit is not None:
                mol_rdkit = rdMO.AddHs(mol_rdkit, addCoords=True)
                # get the conformation of the molecule
                conformer = mol_rdkit.GetConformer()
                # calculate the center of the molecule
                center = rdMT.ComputeCentroid(conformer, ignoreHs=False)
                mol_center = np.asarray([center.x, center.y, center.z])
            else:
                raise ValueError
            pdbw = Chem.rdmolfiles.PDBWriter(fileName=hydrogenized_pdb_file)
            pdbw.write(mol_rdkit)
            pdbw.flush()
            pdbw.close()
            del mol_rdkit, pdbw
        except (IOError, ValueError):
            log.warning("Bad PDB file.")
            return None

        try:
            mol = pd.parsePDB(hydrogenized_pdb_file)
        except IOError:
            log.warning("Could not read PDB file.")
            return None

        if mol is None:
            log.warning("Bad pdb file found.")
            return None

        std_amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                           'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                           'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                           'SER', 'THR', 'TRP', 'TYR', 'VAL']

        canonical_notation = lambda x: x[0].upper() + x[1:].lower() if len(x) > 1 else x
        res = {'coords': mol.getCoords() - mol_center,
               'vdwradii': np.asarray([self.periodic_table.GetRvdw(
                   self.periodic_table.GetAtomicNumber(canonical_notation(atom)))
                                       for atom in mol.getElements()])}

        # find the data for all the 20 amino acids
        for aa in std_amino_acids:
            all_aas_in_mol = mol.select('resname ' + aa)
            if all_aas_in_mol is not None:
                mask = all_aas_in_mol.getIndices()
            else:
                mask = np.array([], dtype=np.int32)
            res['coords_' + aa] = res['coords'][mask, :]
            res['vdwradii_' + aa] = res['vdwradii'][mask]

        # find the data for the backbones
        backbone_mask = mol.backbone.getIndices()
        res['coords_backbone'] = res['coords'][backbone_mask, :]
        res['vdwradii_backbone'] = res['vdwradii'][backbone_mask]

        # find the data for the heavy atoms (i.e. no H atoms)
        heavy_mask = mol.heavy.getIndices()
        res['coords_heavy'] = res['coords'][heavy_mask, :]
        res['vdwradii_heavy'] = res['vdwradii'][heavy_mask]

        # find the data for the heavy atoms (i.e. no H atoms)
        hydro_mask = mol.hydrogen.getIndices()
        res['coords_hydro'] = res['coords'][hydro_mask, :]
        res['vdwradii_hydro'] = res['vdwradii'][hydro_mask]

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
        dummy_vdwradii_input = lasagne.layers.InputLayer(shape=(1, None))
        dummy_natoms_input = lasagne.layers.InputLayer(shape=(1,))
        self.processor = MoleculeMapLayer(incomings=[dummy_coords_input, dummy_vdwradii_input, dummy_natoms_input],
                                          minibatch_size=1, rotate=False)

    def process(self, prot_dir):
        try:
            coords = np.memmap(os.path.join(prot_dir, 'coords.memmap'), mode='r', dtype=floatX).reshape((1, -1, 3))
            vdwradii = np.memmap(os.path.join(prot_dir, 'vdwradii.memmap'), mode='r', dtype=floatX).reshape((1, -1))
            n_atoms = np.array(coords.shape[1], dtype=intX).reshape((1,))
        except IOError:
            return None
        mol_info = [theano.shared(coords),
                    theano.shared(vdwradii),
                    theano.shared(n_atoms)]
        grid = self.processor.get_output_for(mols_info=mol_info).eval()
        return grid


class GridSideChainProcessor(object):
    channels_count = 24

    def __init__(self):
        dummy_coords_input = lasagne.layers.InputLayer(shape=(1, None, None))
        dummy_vdwradii_input = lasagne.layers.InputLayer(shape=(1, None))
        dummy_natoms_input = lasagne.layers.InputLayer(shape=(1,))
        self.processor = MoleculeMapLayer(incomings=[dummy_coords_input,
                                                     dummy_vdwradii_input,
                                                     dummy_natoms_input],
                                          rotate=False,
                                          minibatch_size=1)

    def process(self, prot_dir):
        try:
            memmaps_sufix = ['_backbone.memmap', '_heavy.memmap', '_hydro.memmap',
                             '_ALA.memmap', '_ARG.memmap', '_ASN.memmap', '_ASP.memmap',
                             '_CYS.memmap', '_GLN.memmap', '_GLU.memmap', '_GLY.memmap',
                             '_HIS.memmap', '_ILE.memmap', '_LEU.memmap', '_LYS.memmap',
                             '_MET.memmap', '_PHE.memmap', '_PRO.memmap', '_SER.memmap',
                             '_THR.memmap', '_TRP.memmap', '_TYR.memmap', '_VAL.memmap']
            coords = [np.memmap(os.path.join(prot_dir, 'coords.memmap'), mode='r', dtype=floatX).reshape((1, -1, 3))]
            vdwradii = [np.memmap(os.path.join(prot_dir, 'vdwradii.memmap'), mode='r', dtype=floatX).reshape((1, -1))]
            n_atoms = [np.array([vdwradii[0].size], dtype=intX)]
            for suffix in memmaps_sufix[:self.channels_count - 1]:
                next_coords = np.zeros_like(coords[0])
                next_vdwradii = np.zeros_like(vdwradii[0])

                try:
                    masked_coords = np.memmap(os.path.join(prot_dir, 'coords' + suffix), mode='r',
                                              dtype=floatX).reshape((1, -1, 3))
                    masked_vdwradii = np.memmap(os.path.join(prot_dir, 'vdwradii' + suffix), mode='r',
                                                dtype=floatX).reshape((1, -1))
                    masked_natoms = np.array([masked_vdwradii.size], dtype=intX)
                    next_coords[:, :masked_natoms[0]] = masked_coords
                    next_vdwradii[:, :masked_natoms[0]] = masked_vdwradii
                except ValueError:  # this channel has no atoms
                    masked_natoms = np.array([1], dtype=intX)

                coords.append(next_coords)
                vdwradii.append(next_vdwradii)
                n_atoms.append(masked_natoms)
        except IOError:
            return None
        result = []
        for c, v, na in zip(coords, vdwradii, n_atoms):
            mol_info = [theano.shared(x) for x in
                        [c, v, na]]
            result.append(self.processor.get_output_for(mol_info).eval())
        return np.transpose(np.concatenate(result), (1, 0, 2, 3, 4))
