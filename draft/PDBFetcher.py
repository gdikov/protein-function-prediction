import rdkit.Chem as Chem
from os import listdir, path


class PDBFetcher(object):
    """
    PDBFetcher can download PDB files from the PDB and
    also convert the files to rdkit molecules.
    """

    def __init__(self, dir_path, count=None):
        self.dir_path = dir_path
        self.count = count

    def download_pdb(self):
        from Bio.PDB import PDBList
        pl = PDBList(pdb=self.dir_path)
        pl.flat_tree = 1
        pl.download_entire_pdb()

    def get_molecules(self):
        files = [path.join(self.dir_path, f) for f in listdir(self.dir_path)
                 if f.endswith(".ent") or f.endswith(".pdb")]
        if self.count is None:
            self.count = len(files)
        return [Chem.MolFromPDBFile(molFileName=f) for f in files[:self.count]]


if __name__ == "__main__":
    fetcher = PDBFetcher(dir_path=path.join(path.dirname(path.realpath(__file__)), "../data/pdb"))
