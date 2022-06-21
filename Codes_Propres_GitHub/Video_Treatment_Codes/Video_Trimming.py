# =============================================================================
# Code used to trim video at the correct length
# =============================================================================

# Loading of the different libraries
import cv2
import numpy as np # Computer vision library

# Loading of the video
cap = cv2.VideoCapture('filename')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Frame rate of the video
fps = 150

# Creation of the new trimmed video
writer = cv2.VideoWriter('filename_trimmed_video',cv2.VideoWriter_fourcc(*'DIVX'), fps, (height,width))

# Start of the trimmed video in seconds
t_start = 0

# End of the trimmed video in seconds
t_end = 60

# Trimming loop
i = 0
while True and i < fps*t_end:
    i += 1
    
    # Access the ith frame
    ret,frame = cap.read()
    
    # Break the loop if the video is finished
    if frame is None:
        break
    
    # Write in the new video file
    if i > fps*t_start:
        writer.write(np.flip(np.transpose(frame,axes=[1,0,2]),axis=0))

# Release both videos
writer.release()
cap.release()