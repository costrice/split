from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def __init__(self):
        super(TestOptions, self).__init__()
        # parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results-dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--expname', type=str, default='./results/', help='experiment name.')
        self.parser.add_argument('--aspect-ratio', type=float, default=1.0, help='aspect ratio of result images')
        # Dropout and Batchnorm has different behavioir during training and test.
        self.parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        self.parser.add_argument('--num-test', type=int, default=500, help='how many test images to run')
        self.parser.add_argument('--num-epoch', type=int, default=20, help='epoch of .pth')
        # rewrite devalue values

        self.parser.add_argument("--net_type", type=str, default='autoencoder', help="training_type. linet: pipeline 0;autoencoder: autoencoder in pipeine 1; predictor: predictor in pipeline 1")

        self.parser.add_argument("--ibr_loss", type=int, default=1, help="whether to use ibr loss in training, 1 for using, 0 for not using")
        self.parser.add_argument("--sphere", default=False, action='store_true', help="proportions of training data used as validation set")
        self.parser.add_argument("--autoencode", default=False, action='store_true', help="proportions of training data used as validation set")
        self.parser.add_argument("--indoor", type=int, default=0, help="0 for face training, 1 for indoor, 2 for outdoor")
        self.parser.add_argument("--use_aug", type=int, default=1, help="whether to use data augmentation in training, 1 for using, 0 for not using")

        self.parser.add_argument("--real_data", type=int, default=0, help="whether to use real data testing in training, 1 for using, 0 for not using")
        self.parser.add_argument("--load_ae_dir", type=str, default='', help="AE loading dir. Only useful for predictor training")
        self.parser.add_argument("--predictor_type", type=str, default='hrnet', help="predictor_type. only usefule for predictor training. options: and hrnet")
        self.parser.add_argument("--vis_size", type=int, default=128, help="size used in visualizing")
        self.parser.add_argument("--real_test_mode", type=int, default=0, help="test mode for real data, 0 for indoor+outdoor, 1 for indoor, 2 for outdoor")
        self.parser.add_argument("--ab_setting", nargs='*', default=None, help="datasets for envmap training")
        self.parser.add_argument("--saving_prefix", type=str, default='', help="")
        self.parser.add_argument("--compo_ab", type=int, default=0, help="compo ablation option. o for ssn 1 for no shading, 2 for no specular")
        
        self.parser.add_argument("--no_visualize_test", default=False, action='store_true', help="whether to visualize test results")
        self.parser.add_argument("--gt_test", default=False, action='store_true', help="whether to use gt componets for testing")
        self.parser.add_argument("--in_the_wild", default=False, action='store_true', help="whether to use gt componets for testing")
        self.parser.add_argument("--new_outdoor", default=False, action='store_true', help="whether to use gt componets for testing")
        self.parser.add_argument("--syn_real", default=False, action='store_true', help="whether to use gt componets for testing")
        self.parser.add_argument("--syn_real_v2", default=False, action='store_true', help="whether to use gt componets for testing")
        self.parser.add_argument("--no_si", default=False, action='store_true', help="whether to use gt componets for testing")
        
        self.parser.add_argument("--load_est_dir", type=str, default='', help="AE loading dir. Only useful for predictor training")
        self.parser.add_argument("--load_pred_dir", type=str, default='', help="PRED loading dir. Only useful for predictor training")
        self.parser.add_argument("--use_tex", default=False, action='store_true', help="whether to use gt componets for testing")
        self.parser.add_argument("--output_fid", default=False, action='store_true', help="whether to use gt componets for testing")
        self.parser.add_argument("--rand_tex", default=False, action='store_true', help="whether to use gt componets for random texture")
        self.parser.add_argument("--filter_cloudy", default=False, action='store_true', help="whether to filter cloudy envmaps")
        self.parser.add_argument("--verbose_rand_tex", default=False, action='store_true', help="whether to verbose random texture")
        self.parser.add_argument("--rerender_loss", default=False, action='store_true', help="whether to verbose random texture")

        self.parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        # parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
