import argparse
import os

import torch


class BaseOptions():
    """
    This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and
    model class.
    """
    
    def __init__(self):
        """
        Define the common options that are used in both training and test.
        """
        self.isTrain = None
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # basic parameters
        self.parser.add_argument("--data-path", type=str, default="/data4/chengyean/skin_face/albedo/",
                                 help="path to images (should have subfolders train, test). if not set, use default "
                                      "BASE_OUTPUT_PATH in config.py")
        self.parser.add_argument("--name", type=str, default="default-name",
                                 help="name of the experiment. It decides where to store samples and models")
        self.parser.add_argument("--gpu-ids", type=str, default="0",
                                 help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU")
        self.parser.add_argument("--checkpoints-dir", type=str, default="/userhome/chengyean/face_lighting/lief_face_lighting/codes/models/checkpoints/",
                                 help="models are saved here")
        self.parser.add_argument("--phase", type=str, default="train",
                                 help="train, val, test")
        
        # model parameters
        self.parser.add_argument("--input-hw", type=int, default=256,
                                 help="the width and height of input images")
        self.parser.add_argument("--latent-dim", type=int, default=1024,
                                 help="the size of latent vectors")
        # self.parser.add_argument("--res_pretrain", action="store_true")
        
        # dataset parameters
        self.parser.add_argument("--batch-size", type=int, default=32, help="input batch size.")
        self.parser.add_argument("--max-dataset-size", type=int, default=None,
                                 help="Maximum number of samples allowed per dataset. If the dataset directory "
                                      "contains more than max_dataset_size, only a subset is loaded.")
        # self.parser.add_argument("--no_flip", action="store_true",
        #                     help="if specified, do not flip the images for data augmentation")
        
        # additional parameters
        # self.parser.add_argument("--load_iter", type=int, default="0",
        #                     help="which iteration to load? if load_iter > 0, the codes will load models by iter_["
        #                          "load_iter];  otherwise, the codes will load models by [epoch]")
        # self.parser.add_argument("--epoch", type=str, default="latest",
        #                     help="which epoch to load? set to latest to use latest cached model")
        self.parser.add_argument("--verbose", action="store_true",
                                 help="if specified, print more debugging information")
    
    def print_options(self, opt):
        """
        Print and save options.
        It will print both current options and default values (if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ""
        message += "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = f"\t[default: {default}]"
            message += f"{str(k):>25}: {str(v):<30}{comment}\n"
        message += "----------------- End -------------------"
        print(message)
        
        # save to the disk
        # breakpoint()
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, f"{opt.phase}_opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")
    
    def parse(self):
        """
        Parse our options, create checkpoints directory suffix, and set up gpu device.
        """
        opt, _ = self.parser.parse_known_args()
        # self.print_options(opt)
        
        opt.isTrain = self.isTrain  # train or test
        
        # set gpu ids
        if opt.gpu_ids == "-1":
            opt.device = "cpu"
        else:
            opt.device = "cuda:" + opt.gpu_ids
        
        self.opt = opt
        return self.opt
