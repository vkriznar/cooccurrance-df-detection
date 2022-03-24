from GLCMExtractor import GLCMExtractor


if __name__ == "__main__":
	test_image_path = "./data/test/test-image.jpg"
	glcm_extractor = GLCMExtractor(test_image_path)
	glcm_extractor.construct_rgb_glcms()