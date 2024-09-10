import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import pandas as pd
from tqdm import tqdm

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def df_set(df, image_name, key, value):
    df.at[df.index[df['image_name'] == image_name][0], key] = value

def df_get(df, image_name, key):
    return df.at[df.index[df['image_name'] == image_name][0], key]

root_dir = "/workspace/group12"

video_paths = []

for vid_name in os.listdir(root_dir):
    vid_path = os.path.join(root_dir, vid_name)
    top_path = os.path.join(vid_path, "top")
    bottom_path = os.path.join(vid_path, "bottom")

    for split_name in os.listdir(top_path):
        split_path = os.path.join(top_path, split_name)
        video_paths.append(split_path)
    
    for split_name in os.listdir(bottom_path):
        split_path = os.path.join(bottom_path, split_name)
        video_paths.append(split_path)

print(f"Found {len(video_paths)} videos (splits)")

# from sam2.build_sam import build_sam2_video_predictor

# sam2_checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
# model_cfg = "sam2_hiera_t.yaml"

# predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

# video_dir = "./notebooks/videos/to_process"
# batch_size = 500
# reach = 5

for i in range(len(video_paths)):
    print(video_paths[i])
    root_path = video_paths[i]
    frame_names = [
        p for p in os.listdir(root_path)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    df = pd.read_csv(os.path.join(root_path, "data.csv"))
    init_click_x = df_get(df, frame_names[0], 'click_x')
    if init_click_x != -1.0:
        print(f"yay {root_path} has click_x {i}")
    else:
        printf(f"boo no click_x {root_path} {i}")
    # for i in range(1, 5):
    #     if f'rand_x_{i}' not in df.columns:
    #         df[f'rand_x_{i}'] = -1.0
    #     if f'rand_y_{i}' not in df.columns:
    #         df[f'rand_y_{i}'] = -1.0

    # if 'min_x' not in df.columns:
    #     df['min_x'] = -1.0
    # if 'min_y' not in df.columns:
    #     df['min_y'] = -1.0
    # if 'max_x' not in df.columns:
    #     df['max_x'] = -1.0
    # if 'max_y' not in df.columns:
    #     df['max_y'] = -1.0

