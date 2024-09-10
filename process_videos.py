import os
import cv2
from tqdm import tqdm

video_data_path = "/workspace/group12"

top_box = (0, 0, 960, 540)
bottom_box = (0, 540, 960, 1080)
vid_names = ['S18-20221212-120000-150000.mp4', 'S19-20221212-120000-150000.mp4', 'S20-20230118-120000-150000.mp4', 'S21-20230118-120000-150000.mp4', 'S22-20230129-120000-150000.mp4', 'S23-20230129-120000-150000.mp4', 'S24-20230215-110000-140000.mp4', 'S25-20230215-110000-140000.mp4', 'S26-20230301-120000-150000.mp4', 'S27-20230301-120000-150000.mp4']
i = 7
vid_names = vid_names[i:i+3]
# 0 | 1, 2, 3 | 4, 5, 6 | 7, 8, 9
for vid_name in vid_names:
    vid_path = os.path.join(video_data_path, vid_name)
    video_base_name = os.path.splitext(vid_name)[0]
    output_dir = os.path.join(video_data_path, video_base_name)
    top_dir = os.path.join(output_dir, 'top')
    bottom_dir = os.path.join(output_dir, 'bottom')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(top_dir, exist_ok=True)
    os.makedirs(bottom_dir, exist_ok=True)

    cap = cv2.VideoCapture(vid_path)
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with tqdm(total=total_frames, desc=f'Processing {vid_name}', unit='frame', leave=False) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
        
            frame_filename = f'{frame_count:06d}.jpg'
            if frame_count % 10 != 0:
                frame_count += 1
                pbar.update(1)
                continue
            top_frame = frame[top_box[1]:top_box[3], top_box[0]:top_box[2]]
            top_frame_path = os.path.join(top_dir, frame_filename)
            cv2.imwrite(top_frame_path, top_frame)
            
            bottom_frame = frame[bottom_box[1]:bottom_box[3], bottom_box[0]:bottom_box[2]]
            bottom_frame_path = os.path.join(bottom_dir, frame_filename)
            cv2.imwrite(bottom_frame_path, bottom_frame)

            frame_count += 1
            pbar.update(1)
    
    cap.release()
