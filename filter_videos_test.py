import os
import cv2
from tqdm import tqdm
import numpy as np
import time

video_data_path = "/home/vansh/Desktop/labelled_data"

for vid_name in os.listdir(video_data_path):
    print(vid_name)
    vid_path = os.path.join(video_data_path, vid_name)
    video_base_name = os.path.splitext(vid_name)[0]
    output_dir = os.path.join(video_data_path, video_base_name)
    top_dir = os.path.join(output_dir, 'top')
    bottom_dir = os.path.join(output_dir, 'bottom')
    min_b, min_g, min_r = 1000, 1000, 1000
    max_b, max_g, max_r = 0, 0, 0
    
    print("Top")
    for img_name in tqdm(sorted(os.listdir(top_dir), key=lambda x: int(x.split('.')[0]))):
        img_path = os.path.join(top_dir, img_name)
        img = cv2.imread(img_path)
        blue_channel, green_channel, red_channel = cv2.split(img)
        max_val_blue, max_val_green, max_val_red = np.max(blue_channel), np.max(green_channel), np.max(red_channel)
        min_val_blue, min_val_green, min_val_red = np.min(blue_channel), np.min(green_channel), np.min(red_channel)
        if min_val_blue > 50 and min_val_green > 50 and min_val_red > 50:
            print(img_name, "min", min_val_blue, min_val_green, min_val_red)
        else:
            max_b, max_g, max_r = max(max_b, min_val_blue), max(max_g, min_val_green), max(max_r, min_val_red)
        if max_val_blue < 200 and max_val_green < 200 and max_val_red < 200:
            print(img_name, "max", max_val_blue, max_val_green, max_val_red)
        else:
            min_b, min_g, min_r = min(min_b, max_val_blue), min(min_g, max_val_green), min(min_r, max_val_red)
    print(min_b, min_g, min_r)
    print(max_b, max_g, max_r)

    min_b, min_g, min_r = 1000, 1000, 1000
    max_b, max_g, max_r = 0, 0, 0
    print("Bottom")
    for img_name in tqdm(sorted(os.listdir(bottom_dir), key=lambda x: int(x.split('.')[0]))):
        img_path = os.path.join(bottom_dir, img_name)
        img = cv2.imread(img_path)
        blue_channel, green_channel, red_channel = cv2.split(img)
        max_val_blue, max_val_green, max_val_red = np.max(blue_channel), np.max(green_channel), np.max(red_channel)
        min_val_blue, min_val_green, min_val_red = np.min(blue_channel), np.min(green_channel), np.min(red_channel)
        if min_val_blue > 50 and min_val_green > 50 and min_val_red > 50:
            print(img_name, "min", min_val_blue, min_val_green, min_val_red)
        else:
            max_b, max_g, max_r = max(max_b, min_val_blue), max(max_g, min_val_green), max(max_r, min_val_red)
        if max_val_blue < 200 and max_val_green < 200 and max_val_red < 200:
            print(img_name, "max", max_val_blue, max_val_green, max_val_red)
        else:
            min_b, min_g, min_r = min(min_b, max_val_blue), min(min_g, max_val_green), min(min_r, max_val_red)
    print(min_b, min_g, min_r)
    print(max_b, max_g, max_r)
