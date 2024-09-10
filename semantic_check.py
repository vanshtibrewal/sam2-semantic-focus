import os
from PIL import Image
import pandas as pd
from tqdm import tqdm

def df_set(df, image_name, key, value):
    df.at[df.index[df['image_name'] == image_name][0], key] = value

def df_get(df, image_name, key):
    return df.at[df.index[df['image_name'] == image_name][0], key]

root_dir = "/workspace/data/labelled_data"

root_path = "/workspace/good_bad/S013-20220917-110000-150000/top/split_1" #"/workspace/full_data/root_path/S013-20220920-000000-120000/top/split_0"
 
frame_names = [
    p for p in os.listdir(root_path)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]

frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
df = pd.read_csv(os.path.join(root_path, "data.csv"))

for frame_name in tqdm(frame_names):
    if frame_name != "189375.jpg": # "749980.jpg":
        continue
    frame = Image.open(os.path.join(root_path, frame_name))
    frame.save("test0.jpg", quality=100)
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
    x_max = min(x_max + x_len, 960)
    y_max = min(y_max + y_len, 540)

    cropped_frame = frame.crop((x_min, y_min, x_max, y_max))
    cropped_frame.save("test.jpg", quality=100)
