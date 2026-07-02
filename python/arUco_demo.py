import cv2
import numpy as np
import matplotlib.pyplot as plt
from skvideo.io import vread
import moviepy.editor as mpy
from tqdm import tqdm
from mpl_toolkits.mplot3d import axes3d, Axes3D

def npy_to_gif(npy, filename):
    clip = mpy.ImageSequenceClip(list(npy), fps=10)
    clip.write_gif(filename)

vid = vread("src/shisa.mp4")
print(vid.shape)