import os
import cv2
from tqdm import tqdm

video_data_path = "/workspace/group12"

for vid_name in os.listdir(video_data_path):
    if vid_name.endswith('.mp4') is False:
        continue
    print(vid_name)
    vid_path = os.path.join(video_data_path, vid_name)
    video_base_name = os.path.splitext(vid_name)[0]
    output_dir = os.path.join(video_data_path, video_base_name)
    top_dir = os.path.join(output_dir, 'top')
    bottom_dir = os.path.join(output_dir, 'bottom')

    cap = cv2.VideoCapture(vid_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frames)
    print("top", len(os.listdir(top_dir)))
    print("bottom", len(os.listdir(bottom_dir)))
    
    cap.release()
