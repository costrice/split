from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """
    This class includes training options.
    It also includes shared options defined in BaseOptions.
    """
    
    def __init__(self):
        super(TrainOptions, self).__init__()
        # network saving, loading and logging parameters
        self.parser.add_argument("--save-epoch-freq", type=int, default=5,
                                 help="frequency of saving checkpoints at the end of epochs")
        self.parser.add_argument("--log-batch-freq", type=int, default=10,
                                 help="frequency of logging running metrics during training on batches")
        self.parser.add_argument("--epoch", type=int, default=100, help="# of total epochs to be trained")
        self.parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of Adam")
        self.parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate for Adam")
        self.parser.add_argument("--lambda_adv", type=float, default=0.002, help="weight for adv loss")
        self.parser.add_argument("--gan_early_epoch", type=int, default=20, help="just train gan for the early epochs")
        self.parser.add_argument("--vgg_warmup", type=int, default=-1, help="close vgg for the early epochs")
        self.parser.add_argument("--val-rate", type=float, default=0.1, help="proportions of training data used as "
                                                                             "validation set")
        self.parser.add_argument("--envmap_size", type=int, default=128, help="proportions of training data used as "
                                                                             "validation set")
        self.parser.add_argument("--sphere", default=False, action='store_true', help="proportions of training data used as validation set")
        self.parser.add_argument("--autoencode", default=False, action='store_true', help="proportions of training data used as validation set")
        self.parser.add_argument("--indoor", type=int, default=0, help="0 for face training, 1 for indoor, 2 for outdoor")

        self.parser.add_argument("--net_type", type=str, default='autoencoder', help="training_type. linet: pipeline 0;autoencoder: autoencoder in pipeine 1; predictor: predictor in pipeline 1")

        self.parser.add_argument("--ibr_loss", type=int, default=1, help="whether to use ibr loss in training, 1 for using, 0 for not using")
        self.parser.add_argument("--use_aug", type=int, default=1, help="whether to use data augmentation in training, 1 for using, 0 for not using")

        self.parser.add_argument("--env_datasets", nargs='*', default=None, help="datasets for envmap training")
        self.parser.add_argument("--use_gan", type=int, default=0, help="whether to use gan in training, 1 for using, 0 for not using")
        self.parser.add_argument("--ldr_train", type=int, default=0, help="whether to clip envmap in training, 1 for using, 0 for not using")
        self.parser.add_argument("--load_ae_dir", type=str, default='', help="AE loading dir. Only useful for predictor training")
        self.parser.add_argument("--predictor_type", type=str, default='cyanet', help="predictor_type. only useful for predictor training. options: cya_net and hrnet")
        self.parser.add_argument("--read_from_sphere", type=int, default=0, help="whether to directly read sphere data")
        self.parser.add_argument("--load_pred_dir", type=str, default='', help="AE loading dir. Only useful for predictor training")
        self.parser.add_argument("--ab_setting", nargs='*', default=None, help="datasets for envmap training")
        self.parser.add_argument("--ibr_setting", type=str, default='ours', help="ibr setting. ours or google")
        self.parser.add_argument("--compo_ab", type=int, default=0, help="compo ablation option. o for ssn 1 for no shading, 2 for no specular, 3 for no normal")
        # self.parser.add_argument("--scheduler", nargs='*', default=None, help="datasets for envmap training")
        self.isTrain = True
