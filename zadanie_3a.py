import os
import cv2
from numpy import *
from pandas import *
from skimage.feature import graycomatrix, graycoprops 
class Processor:
    def __init__(self, input: str, output: str, crop_size: float):
        self._input: str = input
        self._output: str = output,
        self._crops: float = crop_size
    def crop(self):
        v_features = []
        files = [file for file in os.listdir(self._input) if any(file.endswith(f"*.{ext}") for ext in "jpg png".split())]
        img_size = float("inf")
        for file in files:
            image = cv2.imread(os.path.join(self._input, file))
            img_size = min(img_size, min(image.shape[:2]))
        xy = (
            (img_size // self._crops),
            (img_size // self._crops)
        )
