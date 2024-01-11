import csv
import os
import pandas as pd

from pdb import set_trace as st
from tqdm import tqdm


import pickle5

def save_list_to_pandas(out_list, out_name):
    df = pd.DataFrame(out_list)
    df.to_csv(out_name, index=False)
    print(f'Generated {out_name} with {len(out_list)} images')
    
    

def parse_to_v2(in_csv, out_csv):
    df = pd.read_csv(in_csv)
    in_list = df['0'].tolist()

    old_middle = 'decom_by_I2AN2DS_Syn+Real'
    new_middle = 'decom_by_I2AN2DS_Syn+Realv2'
    
    out_list = [i.replace(old_middle, new_middle) for i in in_list]
    save_list_to_pandas(out_list, out_csv)

def parse_to_v2_FFHQ(in_csv, out_csv):
    df = pd.read_csv(in_csv)
    in_list = df['0'].tolist()

    old_middle = 'decomposed_by_I2AN2DS_Syn+Real'
    new_middle = 'decomposed_by_I2AN2DS_Syn+Realv2'
    
    out_list = [i.replace(old_middle, new_middle) for i in in_list]
    save_list_to_pandas(out_list, out_csv)

# real in the wild images (do both indoor and outdoor)
# in_csv = '/userhome/chengyean/face_lighting/lief_face_lighting/test_list/full_real_testset_list_new_Syn+Real.csv'
# out_csv = in_csv.replace('.csv', '_v2.csv')
# parse_to_v2(in_csv, out_csv)

# in_csv = '/userhome/chengyean/face_lighting/lief_face_lighting/test_list/full_real_testset_list_old_Syn+Real.csv'
# out_csv = in_csv.replace('.csv', '_v2.csv')
# parse_to_v2(in_csv, out_csv)


in_csv = '/userhome/chengyean/face_lighting/lief_face_lighting/test_list/FFHQ_in_the_wild_syn+real.csv'
out_csv = in_csv.replace('.csv', '_v2.csv')
parse_to_v2_FFHQ(in_csv, out_csv)


# in_csv = '/userhome/chengyean/face_lighting/lief_face_lighting/test_list/laval_cvpr_Syn+Real.csv'
# out_csv = in_csv.replace('.csv', '_v2.csv')
# parse_to_v2(in_csv, out_csv)