import os 
from tqdm import tqdm
import pandas as pd
import cv2

        
def linrgb2srgb(color_linrgb):
    """
    Transform a image in [0, 1] from linear RGB to sRGB space.
    """
    big = color_linrgb > 0.0031308
    color_srgb = big * (1.055 * (color_linrgb ** (1 / 2.4)) - 0.055) + \
                 (~big) * color_linrgb * 12.92
    # color_srgb = color_linrgb ** (1 / 2.2)
    return color_srgb

in_csv = '/userhome/chengyean/face_lighting/lief_face_lighting/test_list/cvpr_final_outdoor_set.csv'
df = pd.read_csv(in_csv)
image_ls = df['path'].tolist()

key_word = 'laval-chosen-ours'

old_prefix = '/data4/chengyean/data/laval-chosen-ours'
new_prefix = '/userhome/feifan/dataset/LavalFaceLighting/face_cropped'

old_middle = '-face'
new_middle = ''

out_dir = '/userhome/chengyean/face_lighting/lief_face_lighting/test_list/verbose'
os.makedirs(out_dir, exist_ok=True)

# Visualize the list
for image_path in tqdm(image_ls):
    if key_word in image_path:
        image_path = image_path.replace(old_prefix, new_prefix)
        image_path = image_path.replace(old_middle, new_middle)
        print(image_path + '.png')
        image_name = image_path.split('/')[-1]
        
        image = cv2.imread(image_path + '.png')
        cv2.imwrite(os.path.join(out_dir, image_name + '.png'), image)
        
        envmap_path = image_path.replace('face_cropped', 'envmap_proc')
        envmap = cv2.imread(envmap_path + '.hdr', flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        cv2.imwrite(os.path.join(out_dir, image_name + '_envmap.png'), envmap)
        