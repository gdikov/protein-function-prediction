import os
import colorlog as log
import logging

log.basicConfig(level=logging.DEBUG)


class EnzymeFetcher(object):
    """
    EnzymeFetcher queries PDB ids for the enzymes EC2PDB data set, extracting them
    from the EC2PDB website based on desired EC categories.
    """
    def __init__(self, categories, excluded_categories=list(), enzyme_dir=None):
        """
        :param categories: which enzyme categories to download
        :param excluded_categories: which enzyme categories to exclude
        :param enzyme_dir: where to download the enzymes
        """
        self.enzyme_dir = enzyme_dir
        self.excluded_categories = excluded_categories
        self.leaf_categories = list()

        log.info("Evaluating the total categorical hierarchy...")
        for cat in set(categories) - set(excluded_categories):
            self._find_leaf_categories(cat)

        self.fetched_prot_codes = dict()

    def _find_leaf_categories(self, cat):
        """
        Internal method, finds all categories in the leafs of the EC tree that are
        children of the specified category.

        The below code parses the HTML of the website for EC2PDB.
        """
        import requests
        from bs4 import BeautifulSoup

        hierarchy_level = cat.count('.') + 1
        if hierarchy_level == 4:
            print("adding: %s" % cat)
            self.leaf_categories.append(cat)
            return

        url = "https://www.ebi.ac.uk/thornton-srv/databases/cgi-bin/enzymes/GetPage.pl?ec_number=" + cat
        page = BeautifulSoup(requests.get(url).text, "html.parser")

        # children table is located after 2 header tables + 2*hierarchy
        # tables for parent categories
        first_child_index = hierarchy_level * 2 + 2

        try:
            children_table = \
            page.find('body').find_all('table', recursive=False)[2]
            children = children_table.find('tr').find('td').find_all('table',
                                                                     recursive=False)[
                       first_child_index:]
        except (AttributeError, IndexError):
            log.warning(
                "No subcategory table found for parent category {0}".format(
                    cat))
            return
        for child in children:
            try:
                child_cat = child.find('a', {'class': 'menuClass'},
                                       href=True).text
            except (AttributeError, IndexError):
                log.warning("Wno link to child category")
                continue
            # remove trailing .- and the "EC " in front
            child_cat = child_cat.rstrip('.-')[3:]
            self._find_leaf_categories(child_cat)

    def fetch_enzymes(self):
        """
        Fetches the protein codes for all enzymes that were selected by the user in the constructor
        of this class.
        :return: a dictionary with keys EC categories (e.g. '3.4.21.4') and values list of protein
        codes for that category (e.g. ['1A0H', '1A0Z']). Only leaf categories (i.e. depth 4) are
        put as keys into the dict.
        """
        if self.leaf_categories is not None:
            log.info(
                "Processing html pages for each enzyme classes ({0} in total). "
                "This may take a while...".format(len(self.leaf_categories)))
            self.fetched_prot_codes = self._ecs2pdbs()
        return self.fetched_prot_codes

    def _ecs2pdbs(self):
        import requests
        pdbs = dict()
        for category in self.leaf_categories:
            url = "https://www.ebi.ac.uk/thornton-srv/databases/cgi-bin/enzymes/GetPage.pl?ec_number=" + category
            response = requests.get(url)
            prots = self._extract_pdbs_from_html(response.text, category)
            if prots is not None:
                pdbs[category] = prots
        return pdbs

    @staticmethod
    def _extract_pdbs_from_html(html_page, cat):
        from bs4 import BeautifulSoup

        parsed_html = BeautifulSoup(html_page, "html.parser")

        try:
            pdb_table = parsed_html.find('body').find_all('p')[2].find('table')
        except (AttributeError, IndexError):
            log.warning("Something went wrong while parsing " + str(
                cat) + " Probably no PDB table present.")
            return None

        if pdb_table is None:
            log.warning(
                "A pdbs-containing table was not found while parsing " + str(
                    cat))
            return None

        pdbs = []

        # skip the first three rows as they don't contain any pdbs and iterate
        # over all others
        pdb_rows = pdb_table.find_all('tr')[3:]
        for row in pdb_rows:
            # get the first data entry and the href argument
            try:
                pdb_code = row.find('td').find('a', href=True).text
            except AttributeError:
                continue
            if len(pdb_code) == 4:
                pdbs.append(str(pdb_code).upper())

        return pdbs


def download_pdbs(base_dir, protein_codes):
    """
    Downloads the PDB database (or a part of it) as PDB files. Every protein is stored in it's own
    directory (with name the PDB code) under base_dir.

    :param base_dir: where to download all the proteins.
    :param protein_codes: the PDB codes of the proteins that should be downloaded.
    """
    prot_codes = []
    if isinstance(protein_codes, dict):
        for key in protein_codes.keys():
            prot_codes += protein_codes[key]
    else:
        prot_codes = protein_codes
    prot_codes = list(set(prot_codes))
    from Bio.PDB import PDBList
    failed = 0
    attempted = len(prot_codes)
    for code in prot_codes:
        try:
            pl = PDBList(pdb=os.path.join(base_dir, code.upper()))
            pl.flat_tree = 1
            pl.retrieve_pdb_file(pdb_code=code)
        except IOError:
            log.warning("Failed to download protein {}".format(code))
            failed += 1
            continue
    log.info(
        "Downloaded {0}/{1} molecules".format(attempted - failed, attempted))


if __name__ == "__main__":
    ep = EnzymeFetcher(categories=['3.4.21', '3.4.24'])
    ep.fetch_enzymes()
    pdbs21 = []
    pdbs24 = []
    for key, value in ep.pdb_files.items():
        if value is not None:
            if key.startswith('3.4.21'):
                pdbs21 += value
            elif key.startswith('3.4.24'):
                pdbs24 += value
    log.info(pdbs21)
    log.info(pdbs24)
