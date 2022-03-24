from multiprocessing.dummy import Array
import cv2
from skimage.feature.texture import graycomatrix
import numpy as np

class GLCMExtractor:

	def __init__(self, angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], distances=[1], levels=256) -> None:
		self.angles = angles
		self.distances = distances
		self.levels = levels


	def construct_rgb_glcm(self, full_path_filename) -> Array:
		image = cv2.imread(full_path_filename)
		r = np.array(image[:, :, 2])
		g = np.array(image[:, :, 1])
		b = np.array(image[:, :, 0])

		glcm_r = graycomatrix(r, self.distances, self.angles, self.levels)
		glcm_g = graycomatrix(g, self.distances, self.angles, self.levels)
		glcm_b = graycomatrix(b, self.distances, self.angles, self.levels)

		glcm_r = np.sum(glcm_r, axis=3)[:, :, 0]
		glcm_g = np.sum(glcm_g, axis=3)[:, :, 0]
		glcm_b = np.sum(glcm_b, axis=3)[:, :, 0]

		return np.asarray([glcm_r, glcm_g, glcm_b], dtype=np.float32)
