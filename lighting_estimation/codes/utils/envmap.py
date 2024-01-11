# -*- coding: utf-8 -*-
import os
import random

import cv2
import numpy as np
from tqdm import tqdm

import config
from utils.general import read_image, write_image


def collect_all_envmaps(train):
    """
    Traverse the face envmap folder, find all envmaps and return their information as a list.
    Args:
        train: whether to read envmap data for training (False for testing).
    Returns:
        envmaps: list of tuples of
            (name of the envmap (without file extension),
             full path to the envmap)
    """
    phase = "train" if train else "test"
    envmaps = {}
    for dataset_name in config.envmap_dataset_names:
        envmaps[dataset_name] = []
    print(f"\nCollecting environment lighting maps for {phase}ing...")
    # check every directory indicated in config.py
    for dict_path_envmap, dataset_name in zip(config.dir_envmap_datasets, config.envmap_dataset_names):
        data_path = dict_path_envmap[phase]
        filtered_files = tqdm(sorted(filter(lambda entry: entry.is_file() and entry.name.endswith("-PS.hdr"),
                                            os.scandir(data_path)),
                                     key=lambda entry: entry.name[:-4]))
        for envmap in filtered_files:
            filtered_files.set_description(f"{os.path.split(os.path.split(data_path)[0])[1]}")
            envmaps[dataset_name].append((envmap.name, os.path.abspath(envmap.path)))
    return envmaps


def pre_process_envmaps():
    """
    Down-sampling the environment map datasets and spinning the brightest colume to center, while dividing them into
    train and test set.
    Returns:

    """
    for idx, data_path in enumerate(config.base_dirs_envmap_datasets):
        # collect all envmaps in data_path
        if os.path.basename(data_path) in ["IndoorHDRDataset2018", "outdoorPanosExr"]:
            envmaps = filter(lambda entry: entry.is_file() and entry.name.endswith((".hdr", ".exr")),
                             os.scandir(data_path))
            envmaps = sorted(envmaps, key=lambda file: str(file.name[:-4]))
            envmaps = list(map(lambda file: file.path, envmaps))
            n_envmaps = len(envmaps)
        else:
            dates = filter(lambda entry: entry.is_dir() and entry.name.isdigit(),
                           os.scandir(data_path))
            dates = [entry.path for entry in dates]
            # no earlier than 06:00, no later than 19:00
            times = [[entry.path for entry in
                      filter(lambda entry: entry.is_dir()
                                           and entry.name.isdigit()
                                           and 60000 < int(entry.name) < 190000,
                             os.scandir(date))]
                     for date in dates]
            times = [item for sublist in times for item in sublist]  # strange flatten
            envmaps = list(map(lambda time: os.path.join(time, "envmap.exr"), times))
            n_envmaps = 1000
        
        if not envmaps:  # check emptiness
            print(f"Envmaps in {data_path} has already been processed or does not exist. Skip.")
            continue
        
        # random dividing into train and test
        
        train_amount = n_envmaps // 6 * 5
        test_amount = n_envmaps - train_amount
        random.seed(2333)
        random.shuffle(envmaps)
        envmaps = {
            "train": envmaps[:train_amount],
            "test": envmaps[train_amount:n_envmaps]
        }
        
        # ==================== BEGIN: modifying and dividing envmaps ====================
        # for finding brightest spinning angle
        height, width = 1024, 2048
        cosine = np.cos((np.arange(height) + 0.5 - height / 2) / height * np.pi)
        
        for phase in ["train", "test"]:
            # move destination
            dst_dir = config.dir_envmap_datasets[idx][phase]
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
                
            print(f"\nPre-processing envmaps in {data_path} for {phase}ing set...")
            for envmap_path in tqdm(envmaps[phase]):
                if os.path.basename(data_path) in ["IndoorHDRDataset2018", "outdoorPanosExr"]:
                    dst_path = os.path.join(dst_dir, os.path.basename(envmap_path)[:-4] + "-P.hdr")
                else:
                    time_path = os.path.dirname(envmap_path)
                    date_path = os.path.dirname(time_path)
                    dst_path = os.path.join(dst_dir,
                                            f"{os.path.basename(date_path)}-{os.path.basename(time_path)}-P.hdr")
                    
                if os.path.exists(dst_path):
                    continue
                
                try:
                    envmap_full = read_image(envmap_path)
                except Exception:
                    print(f"Encounter exception when reading {envmap_path}. Skip.")
                    continue
                # down-sample to speed up
                envmap = cv2.resize(envmap_full, dsize=(width, height), interpolation=cv2.INTER_AREA)
                # compute brightness for each azimuth
                brightness = 0.2126 * envmap[:, :, 0] + 0.7152 * envmap[:, :, 1] + 0.0722 * envmap[:, :, 2]
                brightness = brightness * cosine[:, None]
                brightest_column = np.argmax(np.sum(brightness, axis=0))
                # azimuth_brightest = (np.argmax(brightness) - width / 2) / width * 2 * np.pi
                # spin_brightest_on_front = np.pi / 2 - azimuth_brightest
                
                # spin the brightest column to center
                if brightest_column < width // 2:
                    envmap = np.concatenate((envmap[:, brightest_column + width // 2:, :],
                                             envmap[:, :brightest_column, :],
                                             envmap[:, brightest_column: brightest_column + width // 2, :]),
                                            axis=1)
                else:
                    envmap = np.concatenate((envmap[:, brightest_column - width // 2: brightest_column, :],
                                             envmap[:, brightest_column:, :],
                                             envmap[:, :brightest_column - width // 2, :]),
                                            axis=1)
                    
                # modify mean brightness to 0.1
                envmap = envmap * (0.3 / np.mean(brightness))
                
                write_image(dst_path, envmap)

        # ==================== END: modifying and dividing envmaps ====================
        
        print(f"\nPre-processing of envmaps in {data_path} completed."
              f"\n{train_amount} into train, {test_amount} into test.")


def make_envmap():
    """
    Make a blue and red lighting map.
    """
    dataset_path = os.path.join(config.dir_datasets, "AtypicalPanos")
    width, height = 2048, 1024
    cosine = np.cos((np.arange(height) + 0.5 - height / 2) / height * np.pi)
    envmap = np.zeros(shape=(height, width, 3), dtype=np.float32)
    
    envmap[:, int(width*0.22):int(width*0.28)] = np.array((1, 0.2, 0.2), dtype=np.float32)
    # envmap[:, int(width*0.45):int(width*0.55)] = np.array((0.1, 1, 0.1), dtype=np.float32)
    # envmap[:, int(width*0.7):int(width*0.8)] = np.array((0.1, 0.1, 1), dtype=np.float32)
    envmap[:, int(width*0.72):int(width*0.78)] = np.array((0.2, 1, 0.2), dtype=np.float32)
    # envmap = envmap / np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)

    brightness = 0.2126 * envmap[:, :, 0] + 0.7152 * envmap[:, :, 1] + 0.0722 * envmap[:, :, 2]
    brightness = brightness * cosine[:, None]
    
    envmap = envmap * (0.1 / np.mean(brightness))
    
    write_image(os.path.join(dataset_path, "RG_HHF.hdr"), envmap)
    

if __name__ == '__main__':
    # pre_process_envmaps()
    envmap_train = collect_all_envmaps(train=True)
    envmap_test = collect_all_envmaps(train=False)
    for dataset_name in config.envmap_dataset_names:
        print(f"Dataset {dataset_name}: collected {len(envmap_train[dataset_name])} envmap for training,"
              f" {len(envmap_test[dataset_name])} envmaps for testing.")
    print(f"Envmap size: {read_image(envmap_train['indoor'][0][1]).shape}")
    # make_envmap()