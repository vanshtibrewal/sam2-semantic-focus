import os
import cv2
from tqdm import tqdm

video_data_path = "/path/to/root/directory"

for vid_name in os.listdir(video_data_path):
    print(vid_name)
    vid_path = os.path.join(video_data_path, vid_name)
    video_base_name = os.path.splitext(vid_name)[0]
    output_dir = os.path.join(video_data_path, video_base_name)
    top_dir = os.path.join(output_dir, 'frames')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(top_dir, exist_ok=True)

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
                
                top_frame_path = os.path.join(top_dir, frame_filename)
                cv2.imwrite(top_frame_path, frame)

            frame_count += 1
            pbar.update(1)
    
    cap.release()
