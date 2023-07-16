import cv2
import sys
import tqdm 
import os
import glob

if not os.path.exists('Data'):
    os.mkdir("Data")

path = sys.argv[1]

for video in tqdm.tqdm(glob.glob(path + '*')):
    vid_name = video.split('/')[-1].split('.')[0]
    vidcap = cv2.VideoCapture(video)
    success,image = vidcap.read()
    count = 0
    if not os.path.exists(f'Data/{vid_name}'):
        os.mkdir(f'Data/{vid_name}')
    else:
        continue
    while success:
        cv2.imwrite(f"Data/{vid_name}/frame%d.jpg" % count, image)        
        success,image = vidcap.read()
        count += 1
        