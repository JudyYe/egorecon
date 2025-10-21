import os.path as osp
from jutils import web_utils

def show_all_figs(root_dir,):
    pred_files = glob(osp.join(root_dir, "test_post_*_pred_0000000.mp4"))