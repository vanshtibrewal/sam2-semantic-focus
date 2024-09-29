import os
from PIL import Image
import pandas as pd
from tqdm import tqdm

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

for i in tqdm(range(len(video_paths))):
    root_path = video_paths[i]
    new_path = root_path.replace('old_root', 'new_root') # replace with your directory names
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    frame_names = [
        p for p in os.listdir(root_path)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    if len(frame_names) == len(os.listdir(new_path)):
        continue
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    df = pd.read_csv(os.path.join(root_path, "data.csv"))

    for frame_name in tqdm(frame_names):
        new_frame_path = os.path.join(new_path, frame_name)
        if os.path.exists(new_frame_path):
            continue
        frame = Image.open(os.path.join(root_path, frame_name))

        x_min = df_get(df, frame_name, 'min_x')
        y_min = df_get(df, frame_name, 'min_y')
        x_max = df_get(df, frame_name, 'max_x')
        y_max = df_get(df, frame_name, 'max_y')

        if x_min == -1.0 or y_min == -1.0 or x_max == -1.0 or y_max == -1.0:
            print(f"WARN - {frame_name} in split {root_path} does not have crop box")
            continue

        x_len = (x_max - x_min) // 10
        y_len = (y_max - y_min) // 10

        x_min = max(x_min - x_len, 0)
        y_min = max(y_min - y_len, 0)
        x_max = min(x_max + x_len, 960) # replace with your horizontal resolution
        y_max = min(y_max + y_len, 540) # replace with your vertical resolution

        cropped_frame = frame.crop((x_min, y_min, x_max, y_max))
        cropped_frame.save(new_frame_path, quality=100)
