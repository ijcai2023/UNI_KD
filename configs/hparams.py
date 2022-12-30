
## The cuurent hyper-parameters values are not necessarily the best ones for a specific risk.
def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class HAR():
    def __init__(self):
        super(HAR, self).__init__()
        self.train_params = {
                'num_epochs': 10,
                'batch_size': 32,
                'weight_decay': 1e-4,
                'learning_rate': 1e-2
        }
        self.alg_hparams = {
            'UDA_KD': {'learning_rate': 1e-2, 'temperature': 4, 'src_cls_loss_wt': 1, 'domain_loss_wt': 1, 'errG': 1},
            'MobileDA':     {'learning_rate': 1e-2, 'temperature': 2},
            'JointUKD':     {'learning_rate': 1e-2, 'temperature': 20},
            'AAD':          {'learning_rate': 1e-2, 'temperature': 4, 'src_cls_loss_wt': 1, 'soft_loss_wt': 1, 'errG': 0.1},
            'DANN':         {'learning_rate': 1e-2, 'src_cls_loss_wt': 1, 'domain_loss_wt': 1},
            'DDC':          {'learning_rate': 5e-3, 'src_cls_loss_wt': 6.24, 'domain_loss_wt': 6.36},
            'HoMM':         {'learning_rate': 1e-3, 'src_cls_loss_wt': 2.15, 'domain_loss_wt': 9.13},
            'CoDATS':       {'learning_rate': 1e-3, 'src_cls_loss_wt': 6.21, 'domain_loss_wt': 1.72},
            'MMDA':         {'learning_rate': 1e-3, 'src_cls_loss_wt': 6.13, 'mmd_wt': 2.37, 'coral_wt': 8.63, 'cond_ent_wt': 7.16},
            'CDAN':         {'learning_rate': 1e-2, 'src_cls_loss_wt': 5.19, 'domain_loss_wt': 2.91, 'cond_ent_wt': 1.73},
            'DIRT':         {'learning_rate': 5e-4, 'src_cls_loss_wt': 7.00, 'domain_loss_wt': 4.51, 'cond_ent_wt': 0.79, 'vat_loss_wt': 9.31}
        }


class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        self.train_params = {
                'num_epochs': 10,
                'batch_size': 128,
                'weight_decay': 1e-4,
                'learning_rate': 1e-2

        }
        self.alg_hparams = {
            'UDA_KD': {'learning_rate': 1e-2, 'temperature': 4, 'src_cls_loss_wt': 1, 'domain_loss_wt': 0.01,'errG': 0.1},
            'MobileDA':     {'learning_rate': 1e-2,     'temperature': 2},
            'JointUKD':     {'learning_rate': 1e-2,     'temperature': 20},
            'AAD':          {'learning_rate': 1e-2, 'temperature': 4, 'src_cls_loss_wt': 1, 'soft_loss_wt': 1, 'errG': 0.1},
            'DANN':         {'learning_rate': 0.0005,   'src_cls_loss_wt': 8,       'domain_loss_wt': 0.1, },
            'DDC':          {'learning_rate': 0.0005,   'src_cls_loss_wt': 2.951,   'domain_loss_wt': 8.923, },
            'HoMM':         {'learning_rate': 0.0005,   'src_cls_loss_wt': 0.197,   'domain_loss_wt': 1.102, },
            'CoDATS':       {'learning_rate': 0.01,     'src_cls_loss_wt': 9.239,   'domain_loss_wt': 1.342, },
            'MMDA':         {'learning_rate': 0.0005,   'src_cls_loss_wt': 4.48,    'mmd_wt': 5.951, 'coral_wt': 3.36, 'cond_ent_wt': 6.13, },
            'CDAN':         {'learning_rate': 0.001,    'src_cls_loss_wt': 6.803,   'domain_loss_wt': 4.726, 'cond_ent_wt': 1.307, },
            'DIRT':         {'learning_rate': 0.005,    'src_cls_loss_wt': 9.183,   'domain_loss_wt': 7.411, 'cond_ent_wt': 2.564, 'vat_loss_wt': 3.583, },
        }


class HHAR_SA():
    def __init__(self):
        super(HHAR_SA, self).__init__()
        self.train_params = {
                'num_epochs': 10,
                'batch_size': 32,
                'weight_decay': 1e-4,
                'learning_rate': 1e-2
        }
        self.alg_hparams = {
            'UDA_KD': {'learning_rate': 1e-2, 'temperature': 4, 'src_cls_loss_wt': 1, 'domain_loss_wt': 1, 'errG': 1},
            'MobileDA':     {'learning_rate': 1e-2,     'temperature': 2},
            'JointUKD':     {'learning_rate': 1e-2,     'temperature': 20},
            'AAD':          {'learning_rate': 1e-2, 'temperature': 4, 'src_cls_loss_wt': 1, 'soft_loss_wt': 1, 'errG': 0.1},
            'DANN':         {'learning_rate': 0.0005,   'src_cls_loss_wt': 1.0,     'domain_loss_wt': 1.0},
            'DDC':          {'learning_rate': 0.01,     'src_cls_loss_wt':  0.1593, 'domain_loss_wt': 0.2048},
            'HoMM':         {'learning_rate':0.001,     'src_cls_loss_wt': 0.2429,  'domain_loss_wt': 0.9824},
            'CoDATS':       {'learning_rate': 0.0005,   'src_cls_loss_wt': 0.5416,  'domain_loss_wt': 0.5582},
            'MMDA':         {'learning_rate': 0.001,    'src_cls_loss_wt': 0.9505,  'mmd_wt': 0.5476,           'cond_ent_wt': 0.5167,  'coral_wt': 0.5838, },
            'CDAN':         {'learning_rate': 0.001,    'src_cls_loss_wt': 0.6636,  'domain_loss_wt': 0.1954,   'cond_ent_wt':0.0124},
            'DIRT':         {'learning_rate': 0.001,    'src_cls_loss_wt': 0.9752,  'domain_loss_wt': 0.3892,   'cond_ent_wt': 0.09228,  'vat_loss_wt': 0.1947}
        }


class FD():
    def __init__(self):
        super(FD, self).__init__()
        self.train_params = {
                'num_epochs': 10,
                'batch_size': 32,
                'weight_decay': 1e-4,
                'learning_rate': 1e-2
        }
        self.alg_hparams = {
            'UDA_KD':   {'learning_rate': 1e-2, 'temperature': 4, 'src_cls_loss_wt': 1, 'domain_loss_wt': 1, 'errG': 1},
            'MobileDA': {'learning_rate': 1e-2, 'temperature': 2},
            'JointUKD': {'learning_rate': 1e-2, 'temperature': 20},
            'AAD':      {'learning_rate': 1e-2, 'temperature': 4, 'src_cls_loss_wt': 1, 'soft_loss_wt': 1, 'errG': 0.1},
            'DANN':         {'learning_rate': 0.0005,   'src_cls_loss_wt': 0.9603,  'domain_loss_wt':0.9238},
            'DDC':          {'learning_rate': 0.01,     'src_cls_loss_wt':  0.1593, 'domain_loss_wt': 0.2048},
            'HoMM':         {'learning_rate':0.001,     'src_cls_loss_wt': 0.2429,  'domain_loss_wt': 0.9824},
            'CoDATS':       {'learning_rate': 0.0005,   'src_cls_loss_wt': 0.5416,  'domain_loss_wt': 0.5582},
            'MMDA':         {'learning_rate': 0.001,    'src_cls_loss_wt': 0.9505,  'mmd_wt': 0.5476,           'cond_ent_wt': 0.5167,  'coral_wt': 0.5838, },
            'CDAN':         {'learning_rate': 0.001,    'src_cls_loss_wt': 0.5,  'domain_loss_wt': 0.1,   'cond_ent_wt':0.1},
            'DIRT':         {'learning_rate': 0.001,    'src_cls_loss_wt': 1.0,  'domain_loss_wt': 0.5,   'cond_ent_wt': 0.1,  'vat_loss_wt': 0.1}
        }