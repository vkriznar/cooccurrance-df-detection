import numpy as np
from glob import glob
from GLCMExtractor import GLCMExtractor

if __name__ == "__main__":
	glcm_extractor = GLCMExtractor()
	files = glob("UADFV/30-frames-faces/*/*/*.png")

	for file_name in files:
		tensor = glcm_extractor.construct_rgb_glcm(file_name)
		filename_split = file_name.split("\\")
		tensor_filename = "\\".join(filename_split[:-1]) + "\\" + filename_split[-1].replace(".png", ".npy")
		with open(tensor_filename, 'wb') as f:
			np.save(f, tensor)
		f.close()

	""" fake_files = glob("UADFV/30-frames-faces/fake/*/*.png")
	real_files = glob("UADFV/30-frames-faces/real/*/*.png")

	for file_name in fake_files[:int(9*len(fake_files) / 10)]:
		csv_writer.writerow([glcm_extractor.construct_rgb_glcm(file_name), 1])

	for file_name in real_files[:int(9*len(fake_files) / 10)]:
		csv_writer.writerow([glcm_extractor.construct_rgb_glcm(file_name), 0])

	f.close()

	csv_file = "UADFV/cooccurrance-0-degree-validate.csv"
	f = open(csv_file, "w")
	csv_writer = csv.writer(f)

	for file_name in fake_files[int(9*len(fake_files) / 10):]:
		csv_writer.writerow([glcm_extractor.construct_rgb_glcm(file_name), 1])

	for file_name in real_files[int(9*len(fake_files) / 10):]:
		csv_writer.writerow([glcm_extractor.construct_rgb_glcm(file_name), 0])

	f.close()

	frame_reader = open(csv_file, 'r')
	csv_reader = csv.reader(frame_reader)

	for f in csv_reader:
		print(f[0]) """



