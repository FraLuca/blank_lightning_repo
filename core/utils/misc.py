import argparse
import os
import errno
from core.configs import cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Optimization Learning with Diffusion Process")
    parser.add_argument("-cfg", "--config-file", default="", metavar="FILE", help="path to config file", type=str)
    parser.add_argument(
        "--proctitle",
        type=str,
        default="param_text_align",
        help="allow a process to change its title",
    )
    parser.add_argument(
        "opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER
    )

    args = parser.parse_args()

    if args.opts is not None and args.opts != []:
        args.opts[-1] = args.opts[-1].strip("\r\n")

    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.SAVE_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.NAME)
    print("Saving to {}".format(cfg.SAVE_DIR))
    cfg.freeze()

    return args



def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise