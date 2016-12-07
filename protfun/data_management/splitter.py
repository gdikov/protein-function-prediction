import os
import colorlog as log
import numpy as np
import cPickle

class _DataSplitter():
    def __init__(self,
                 data_dir=None,
                 percentage_test=10,
                 percentage_val=20):
        self.trainval_dir = data_dir

        self.percentage_test = percentage_test
        self.percentage_val = percentage_val

    def store_test_data(self):
        test_dir = os.path.join(self.trainval_dir['data_proc'], 'FORBIDDEN_FOLDER')
        self.test_dir = {'data': test_dir,
                         'pdb': os.path.join(test_dir, "pdb"),
                         'go': os.path.join(test_dir, "go"),
                         'enzymes': os.path.join(test_dir, "enzymes"),
                         'moldata': os.path.join(test_dir, "moldata")}

        if not os.path.exists(self.test_dir['data']):
            os.makedirs(self.test_dir['data'])
        if not os.path.exists(self.test_dir['pdb']):
            os.makedirs(self.test_dir['pdb'])
        if not os.path.exists(self.test_dir['go']):
            os.makedirs(self.test_dir['go'])
        if not os.path.exists(self.test_dir['enzymes']):
            os.makedirs(self.test_dir['enzymes'])
        if not os.path.exists(self.test_dir['moldata']):
            os.makedirs(self.test_dir['moldata'])

        if len(os.listdir(self.test_dir['enzymes'])) > 0:
            log.warning("CAUTION: This method is meant to be executed only once. "
                        "It look like a test set has already been created. "
                        "Reinitializing it may bias the actual performance of the network and will reduce "
                        "the size of the train set as merging back is morally not allowed. "
                        "Are you sure you want to reinitialize it again? yesimsure/[n]")
            resp = raw_input()
            if resp.startswith('yesimsure'):
                test_enzymes = self._split_test()
        else:
            test_enzymes = self._split_test()

        return self.test_dir, test_enzymes


    def split_trainval(self):
        return self._split_trainval()


    def _split_test(self):
        log.info("Splitting data into test and trainval")
        test_dict = dict()
        # take self.percentage_test data points from each hierarchical leaf class
        leaf_classes = [x for x in os.listdir(self.trainval_dir['enzymes_proc']) if x.endswith('.proteins')]
        for cls in leaf_classes:
            path_to_cls = os.path.join(self.trainval_dir['enzymes_proc'], cls)
            with open(path_to_cls, 'r') as f:
                prot_codes_in_cls = [pc.strip() for pc in f.readlines()]

            num_prots_in_cls = len(prot_codes_in_cls)
            if num_prots_in_cls < 1:
                test_indices = np.array([])
            else:
                test_indices = np.random.choice(np.arange(num_prots_in_cls),
                                                replace=False,
                                                size=int(num_prots_in_cls
                                                         * (float(self.percentage_test) / 100.0)))
            test_prot_codes_in_cls = [prot_codes_in_cls[i] for i in test_indices]
            test_dict[cls.replace('.proteins', '')] = test_prot_codes_in_cls
            trainval_indices = np.setdiff1d(np.arange(num_prots_in_cls), test_indices)
            trainval_prot_codes_in_cls = [prot_codes_in_cls[i] for i in trainval_indices]
            # remove entry from the trainval set
            with open(path_to_cls, 'w') as f:
                for pc in trainval_prot_codes_in_cls:
                    f.write(pc + '\n')
            # write to test dir
            path_to_test_cls = os.path.join(self.test_dir['enzymes'], cls)
            with open(path_to_test_cls, 'w') as f:
                for pc in test_prot_codes_in_cls:
                    f.write(pc + '\n')

        with open(os.path.join(self.test_dir['enzymes'], 'test_data.pickle'), 'wb') as f:
            cPickle.dump(test_dict, f)

        return test_dict


    def _split_trainval(self):
        log.info("Splitting trainval data into train and validation sets")
        # take self.percentage_val data points from each hierarchical leaf class
        leaf_classes = [x for x in os.listdir(self.trainval_dir['enzymes_proc']) if x.endswith('.proteins')]
        train_enzymes = dict()
        val_enzymes = dict()
        for cls in leaf_classes:
            path_to_cls = os.path.join(self.trainval_dir['enzymes_proc'], cls)
            with open(path_to_cls, 'r') as f:
                prot_codes_in_cls = [pc.strip() for pc in f.readlines()]
            num_prots_in_cls = len(prot_codes_in_cls)
            if num_prots_in_cls < 1:
                val_indices = np.array([])
            else:
                val_indices = np.random.choice(np.arange(num_prots_in_cls),
                                               replace=False,
                                               size=int(num_prots_in_cls
                                                        * (float(self.percentage_val) / 100.0)))
            val_prot_codes_in_cls = [prot_codes_in_cls[i] for i in val_indices]
            val_enzymes[cls.replace('.proteins', '')] = val_prot_codes_in_cls

            train_indices = np.setdiff1d(np.arange(len(prot_codes_in_cls)), val_indices)
            train_prot_codes_in_cls = [prot_codes_in_cls[i] for i in train_indices]
            train_enzymes[cls.replace('.proteins', '')] = train_prot_codes_in_cls

        with open(os.path.join(self.trainval_dir['enzymes_proc'], 'train_data.pickle'), 'wb') as f:
            cPickle.dump(train_enzymes, f)
        with open(os.path.join(self.trainval_dir['enzymes_proc'], 'val_data.pickle'), 'wb') as f:
            cPickle.dump(val_enzymes, f)

        return train_enzymes, val_enzymes
