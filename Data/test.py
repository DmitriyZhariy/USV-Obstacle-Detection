import cv2
import numpy as np

mask = cv2.imread(r"E:\Education\4 course 2 semester\Practice\panoptic_project\Data\lars_converted\semantic\train\masks\davimar_seq_01_00017.png", 0)
print(np.unique(mask))