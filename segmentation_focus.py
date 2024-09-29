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

root_dir = "/path/to/root/directory"

video_paths = []

for vid_name in os.listdir(root_dir):
    vid_path = os.path.join(root_dir, vid_name)
    frame_path = os.path.join(vid_path, "frames")

    for split_name in os.listdir(frame_path):
        split_path = os.path.join(frame_path, split_name)
        video_paths.append(split_path)

print(f"Found {len(video_paths)} videos (splits)")

from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

video_dir = "./notebooks/videos/to_process"
batch_size = 500
reach = 5

for i in tqdm(range(len(video_paths))):
    print(video_paths[i])
    root_path = video_paths[i]
    frame_names = [
        p for p in os.listdir(root_path)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    df = pd.read_csv(os.path.join(root_path, "data.csv"))

    for i in range(1, 5):
        if f'rand_x_{i}' not in df.columns:
            df[f'rand_x_{i}'] = -1.0
        if f'rand_y_{i}' not in df.columns:
            df[f'rand_y_{i}'] = -1.0

    if 'min_x' not in df.columns:
        df['min_x'] = -1.0
    if 'min_y' not in df.columns:
        df['min_y'] = -1.0
    if 'max_x' not in df.columns:
        df['max_x'] = -1.0
    if 'max_y' not in df.columns:
        df['max_y'] = -1.0

    for frame_idx in range(0, len(frame_names), batch_size):
        if os.path.exists(video_dir):
            shutil.rmtree(video_dir)
        os.makedirs(video_dir)

        curr_frame_names = frame_names[frame_idx:frame_idx + batch_size + reach]

        if df_get(df, curr_frame_names[-1], 'min_x') != -1:
            continue

        for frame_name in curr_frame_names:
            frame_path = os.path.join(root_path, frame_name)
            shutil.copy(frame_path, video_dir)
        
        torch.cuda.empty_cache()
        
        inference_state = predictor.init_state(video_path=video_dir)
        predictor.reset_state(inference_state)

        if frame_idx == 0:
            init_click_x = df_get(df, curr_frame_names[0], 'click_x')
            init_click_y = df_get(df, curr_frame_names[0], 'click_y')
            assert init_click_x != -1.0
            assert init_click_y != -1.0

            points = np.array([[init_click_x, init_click_y]], dtype=np.float32)
            labels = np.array([1], np.int32)

            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=points,
                labels=labels,
            )
        else:
            for sub_frame_no in range(reach):
                points = []
                labels = []
                for i in range(1, 5):
                    rand_x = df_get(df, curr_frame_names[sub_frame_no], f'rand_x_{i}')
                    rand_y = df_get(df, curr_frame_names[sub_frame_no], f'rand_y_{i}')
                    if rand_x != -1.0 and rand_y != -1.0:
                        points.append([rand_x, rand_y])
                        labels.append(1)
                    else:
                        print(f"WARN - {root_path} {curr_frame_names[sub_frame_no]} does not have random points in batch", frame_idx)
                points = np.array(points, dtype=np.float32)
                labels = np.array(labels, np.int32)
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=sub_frame_no,
                    obj_id=1,
                    points=points,
                    labels=labels,
                )
                x_min = df_get(df, curr_frame_names[sub_frame_no], 'min_x')
                y_min = df_get(df, curr_frame_names[sub_frame_no], 'min_y')
                x_max = df_get(df, curr_frame_names[sub_frame_no], 'max_x')
                y_max = df_get(df, curr_frame_names[sub_frame_no], 'max_y')
                if x_min == -1.0 or y_min == -1.0 or x_max == -1.0 or y_max == -1.0:
                    print(f"WARN - {root_path} {curr_frame_names[sub_frame_no]} does not have crop box in batch", frame_idx)
                    continue
                x_len = (x_max - x_min) // 10
                y_len = (y_max - y_min) // 10
                x_min = max(x_min - x_len, 0)
                y_min = max(y_min - y_len, 0)
                x_max = min(x_max + x_len, 960)
                y_max = min(y_max + y_len, 540)
                box = [x_min, y_min, x_max, y_max]
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=sub_frame_no,
                    obj_id=1,
                    box=box,
                )
                        
        video_segments = {} 
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        out_obj_id = 1
        for save_frame_idx in range(0, len(curr_frame_names)):
            frame_name = curr_frame_names[save_frame_idx]
            out_mask = video_segments[save_frame_idx][out_obj_id][0]

            y, x = np.nonzero(out_mask)
            if len(y) == 0:
                print("WARN - no points in mask for frame", root_path, frame_name, "in batch", frame_idx)
                continue
            y_min, x_min = np.min(y), np.min(x)
            y_max, x_max = np.max(y), np.max(x)
            df_set(df, frame_name, 'min_x', x_min)
            df_set(df, frame_name, 'min_y', y_min)
            df_set(df, frame_name, 'max_x', x_max)
            df_set(df, frame_name, 'max_y', y_max)
            if save_frame_idx >= len(curr_frame_names) - reach:
                for i in range(1, 5):
                    rand = np.random.randint(len(y))
                    df_set(df, frame_name, f'rand_x_{i}', x[rand])
                    df_set(df, frame_name, f'rand_y_{i}', y[rand])

        df.to_csv(os.path.join(root_path, "data.csv"), index=False)
