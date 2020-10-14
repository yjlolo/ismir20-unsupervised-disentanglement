import os
import copy
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json


class ConfigParser:
    def __init__(self, config, testing=False, resume=None, modification=None, run_id=None):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['trainer']['save_dir'])

        if self.config['arch']['type'] != 'Classifier':
            self.config['name'] = str(Path(self.config['name']) / self._naming())

        exper_name = self.config['name']
        # if run_id is None: # use timestamp as default run-id
        #     run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        # self._save_dir = save_dir / 'models' / exper_name / run_id
        # self._log_dir = save_dir / 'log' / exper_name / run_id
        self._save_dir = save_dir / 'models' / exper_name
        self._log_dir = save_dir / 'log' / exper_name

        # make directory for saving checkpoints and log.
        if not testing:
            exist_ok = run_id == ''
            self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

            # save updated config file to the checkpoint dir
            print('**********************************************************************************************************')
            write_json(self.config, self.save_dir / 'config.json')

            # configure logging module
            setup_logging(self.log_dir)
            self.log_levels = {
                0: logging.WARNING,
                1: logging.INFO,
                2: logging.DEBUG
            }

    def _naming(self):
        if self.config['arch']['type'] == 'HarmonicVAE':
            arch_args = copy.deepcopy(self.config['arch']['args'])
            for k, v in arch_args.items():
                if isinstance(v, bool):
                    arch_args[k] = 't' if v else 'f'

            seed = 'ms=' + str(self.config['trainer']['seed'])
            split = 'ds=' + str(self.config['data_loader']['args']['split'])

            pitch_shift = 'ps=' + str(self.config['trainer']['pitch_shift'])
            n_latent = 'l=' + str(arch_args['latent_dim'])
            decoding = arch_args['decoding']
            mfcc = 'mfcc=' + arch_args['encode_mfcc']
            pitch_emb = arch_args['pitch_embedding'] + '_' + arch_args['learn_pitch_emb']
            pretrain_step = 'pre=' + str(self.config['loss']['args']['pretrain_step'])
            # anneal_step = 'an=' + str(self.config['loss']['args']['anneal_step'])
            gumbel = 'gb=' + arch_args['gumbel']
            # hard = 'gh=' + arch_args['hard_gumbel']

            # use_hp = 'hp=' + arch_args['use_hp']
            # hp_share = 'hp_s=' + arch_args['hp_share']
            # hp = '-'.join([use_hp, hp_share])

            # bn_act = arch_args['bn'] + '_' + arch_args['act']
            # decoder_arch = arch_args['decoder_arch']
            bs = str(self.config['data_loader']['args']['batch_size'])
            w_recon = str(self.config['loss']['args']['w_recon'])
            w_kl = str(self.config['loss']['args']['w_kl'])
            w_lmse = str(self.config['loss']['args']['w_lmse'])
            w_contrast = str(self.config['loss']['args']['w_contrast'])
            w_cycle = str(self.config['loss']['args']['w_cycle'])
            w_pseudo = str(self.config['loss']['args']['w_pseudo'])
           
            labeled = 'su=' + str(self.config['trainer']['labeled'])
            pseudo_train = 'pse=' + str(self.config['trainer']['pseudo_train'])
            back_freeze = 'bf=' + str(self.config['trainer']['freeze_encoder'])            


            if labeled.split('=')[-1] != '0.0':
                assert w_pseudo == '1'
                if labeled.split('=')[-1] == '1.0':
                    print('Fully supervised training')
                    w_pseudo = '0'

            weights = '-'.join([w_recon, w_kl, w_lmse, w_contrast, w_cycle, w_pseudo])
            # jfname = '-'.join([split, seed, bs, decoding, pitch_emb, hp, n_latent, mfcc, labeled, pitch_shift, pretrain_step, \
            # j                  gumbel, pseudo_train, back_freeze, weights])
            # fname = '-'.join([split, seed, bs, decoding, pitch_emb, n_latent, mfcc, labeled, pitch_shift, pretrain_step, \
            #                   gumbel, pseudo_train, back_freeze, weights])
            fname = '-'.join([seed, pitch_shift, labeled, n_latent, weights, back_freeze])
            return fname

        elif self.config['arch']['type'] == 'ConvNet':
            arch_args = copy.deepcopy(self.config['arch']['args'])
            target = 't=' + str(arch_args['target'])
            fname = target
            return fname

    @classmethod
    def from_args(cls, args, options='', testing=False):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)
        
        config = read_json(cfg_fname)
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        # parse custom cli options into dictionary
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume=resume, modification=modification, testing=testing)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config

def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
