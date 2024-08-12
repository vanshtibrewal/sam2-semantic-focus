import os
import cv2
from tqdm import tqdm

video_data_path = "/home/vansh/Desktop/labelled_video_unlocked_vids"

top_box = (0, 0, 960, 540)
bottom_box = (0, 540, 960, 1080)

for vid_name in os.listdir(video_data_path):
    print(vid_name)
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
    
    with tqdm(total=total_frames, desc=f'Processing {vid_name}', unit='frame') as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_count % 10 == 0:
                frame_filename = f'{frame_count:06d}.jpg'
                
                top_frame = frame[top_box[1]:top_box[3], top_box[0]:top_box[2]]
                top_frame_path = os.path.join(top_dir, frame_filename)
                cv2.imwrite(top_frame_path, top_frame)
                
                bottom_frame = frame[bottom_box[1]:bottom_box[3], bottom_box[0]:bottom_box[2]]
                bottom_frame_path = os.path.join(bottom_dir, frame_filename)
                cv2.imwrite(bottom_frame_path, bottom_frame)

            frame_count += 1
            pbar.update(1)
    
    cap.release()
