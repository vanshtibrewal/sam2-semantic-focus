import os
import shutil
from tqdm import tqdm
import numpy as np

video_data_path = "/workspace/group12"

for vid_name in os.listdir(video_data_path):
    print(vid_name)
    if vid_name.endswith('.mp4'):
        continue
    vid_path = os.path.join(video_data_path, vid_name)
    video_base_name = os.path.splitext(vid_name)[0]
    output_dir = os.path.join(video_data_path, video_base_name)
    top_dir = os.path.join(output_dir, 'top')
    bottom_dir = os.path.join(output_dir, 'bottom')

    img_names = sorted(os.listdir(top_dir), key=lambda x: int(x.split('.')[0]))
    curr_split = []
    split_no = 0
    if img_names:
        curr_split.append(os.path.join(top_dir, img_names[0]))
    
    for i in tqdm(range(1, len(img_names))):
        img_name = img_names[i]
        prev_name = img_names[i-1]
        img_path = os.path.join(top_dir, img_name)
        if int(img_name.split('.')[0]) - int(prev_name.split('.')[0]) > 10:
            split_dir = os.path.join(top_dir, f'split_{split_no}')
            os.makedirs(split_dir, exist_ok=True)
            for img in curr_split:
                shutil.move(img, os.path.join(split_dir, os.path.basename(img)))
            split_no += 1
            curr_split = [img_path]
        else:
            curr_split.append(img_path)
    if curr_split:
        split_dir = os.path.join(top_dir, f'split_{split_no}')
        os.makedirs(split_dir, exist_ok=True)
        for img in curr_split:
            shutil.move(img, os.path.join(split_dir, os.path.basename(img)))

    img_names = sorted(os.listdir(bottom_dir), key=lambda x: int(x.split('.')[0]))
    curr_split = []
    split_no = 0
    if img_names:
        curr_split.append(os.path.join(bottom_dir, img_names[0]))

    for i in tqdm(range(1, len(img_names))):
        img_name = img_names[i]
        prev_name = img_names[i-1]
        img_path = os.path.join(bottom_dir, img_name)
        if int(img_name.split('.')[0]) - int(prev_name.split('.')[0]) > 10:
            split_dir = os.path.join(bottom_dir, f'split_{split_no}')
            os.makedirs(split_dir, exist_ok=True)
            for img in curr_split:
                shutil.move(img, os.path.join(split_dir, os.path.basename(img)))
            split_no += 1
            curr_split = [img_path]
        else:
            curr_split.append(img_path)
    if curr_split:
        split_dir = os.path.join(bottom_dir, f'split_{split_no}')
        os.makedirs(split_dir, exist_ok=True)
        for img in curr_split:
            shutil.move(img, os.path.join(split_dir, os.path.basename(img)))
