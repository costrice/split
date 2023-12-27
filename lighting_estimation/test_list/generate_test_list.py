import csv
import os
import pandas as pd

from pdb import set_trace as st
from tqdm import tqdm


import pickle5

def generate_test_list(in_dir: str, out_name: str) -> None:
    # only useful for one level of subfolders but much faster
    in_list = os.listdir(in_dir)
    # sort the list
    in_list = sorted(in_list)
    out_list = [os.path.join(in_dir, i) for i in tqdm(in_list)]
    df = pd.DataFrame(out_list)
    df.to_csv(out_name, index=False)
    print(f'Generated {out_name} with {len(out_list)} images')

def generate_test_list_traverse(in_dir: str, out_name: str) -> None:
    # extremely slow
    out_list = []
    for path, _, _ in tqdm(os.walk(in_dir)):
        if 'albedo.hdr' in os.listdir(path):
            out_list.append(path)
    df = pd.DataFrame(out_list)
    df.to_csv(out_name, index=False)
    print(f'Generated {out_name} with {len(out_list)} images')

def save_list_to_pandas(out_list, out_name):
    df = pd.DataFrame(out_list)
    df.to_csv(out_name, index=False)
    print(f'Generated {out_name} with {len(out_list)} images')
    
def change_ls(in_list)
    
# synthetic test GT sp
# in_dir = '/userhome/feifan/dataset/syntest_distributed/indoor'
# out_name = 'indoor_syntest_gt_list.csv'
# generate_test_list(in_dir, out_name)

# in_dir = '/userhome/feifan/dataset/syntest_distributed/outdoor'
# out_name = 'outdoor_syntest_gt_list.csv'
# generate_test_list(in_dir, out_name)

# synthetic test GT face
# pkl_dir = '/userhome/feifan/dataset/Face-Light-v2/test/dataset-v2-info.pkl'
# data_ls = pickle5.load(open(pkl_dir, 'rb'))

# out_list = data_ls['indoor']
# out_name = 'indoor_syntest_gt_face_list.csv'
# save_list_to_pandas(out_list, out_name)

# out_list = data_ls['outdoor'] + data_ls['sky']
# out_name = 'outdoor_syntest_gt_face_list.csv'
# save_list_to_pandas(out_list, out_name)

# # real in the wild images (do both indoor and outdoor)
# in_dir = '/userhome/feifan/dataset/FFHQ/decomposed_by_I2AN2DS_Syn+Real'
# out_name = 'FFHQ_in_the_wild_syn+real.csv'
# generate_test_list(in_dir, out_name)

# in_dir = '/userhome/feifan/dataset/FFHQ/decomposed_by_I2AN2DS_Syn'
# out_name = 'FFHQ_in_the_wild_syn.csv'
# generate_test_list(in_dir, out_name)


# synthetic test est sp [PAMI]
# in_dir = '/userhome/feifan/dataset/SynValid_EstBySynonly_Sph/indoor'
# out_name = 'pami_indoor_synval_synonly_est_list.csv'
# generate_test_list(in_dir, out_name)

# in_dir = '/userhome/feifan/dataset/SynValid_EstBySynonly_Sph/outdoor'
# out_name = 'pami_outdoor_synval_synonly_est_list.csv'
# generate_test_list(in_dir, out_name)


# synthetic test GT sp [PAMI]
# in_dir = '/userhome/feifan/dataset/SynValid_GT_Sph/indoor'
# out_name = 'pami_indoor_synval_gt_list.csv'
# generate_test_list(in_dir, out_name)

# in_dir = '/userhome/feifan/dataset/SynValid_GT_Sph/outdoor'
# out_name = 'pami_outdoor_synval_gt_list.csv'
# generate_test_list(in_dir, out_name)


# synthetic test GT face [PAMI]
# in_dir = '/userhome/feifan/dataset/SynValid_GT_Face/indoor'
# out_name = 'pami_indoor_synval_gt_face_list.csv'
# generate_test_list(in_dir, out_name)

# in_dir = '/userhome/feifan/dataset/SynValid_GT_Face/outdoor'
# out_name = 'pami_outdoor_synval_gt_face_list.csv'
# generate_test_list(in_dir, out_name)

# synthetic test est face [PAMI]
# in_dir = '/userhome/feifan/dataset/SynValid_EstBySynonly_Face/indoor'
# out_name = 'pami_indoor_synval_est_face_list.csv'
# generate_test_list(in_dir, out_name)

# in_dir = '/userhome/feifan/dataset/SynValid_EstBySynonly_Face/outdoor'
# out_name = 'pami_outdoor_synval_est_face_list.csv'
# generate_test_list(in_dir, out_name)

