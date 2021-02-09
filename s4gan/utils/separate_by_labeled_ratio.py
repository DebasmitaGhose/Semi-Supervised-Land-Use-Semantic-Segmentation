import numpy as np
import os

sampling_techniques = ['uncertainty', 'entropy', 'margin']
labeled_ratio_dict = {34:"0.02", 85:"0.05", 211:"0.125", 338:"0.2", 557:"0.33", 845:"0.5", 1267:"0.75"}


for sampling in sampling_techniques:

	names_file = '../'+sampling+'/'+sampling+'_names.npy'

	names = np.load(names_file)

	print(np.shape(names))

	save_dir = '../'+sampling

	if not os.path.exists(save_dir):
    		os.makedirs(save_dir)

	for item in labeled_ratio_dict.items():
    		num_images = item[0]
    		lab_rat = item[1]
    		names_lab_rat = names[:num_images]
    		print(np.shape(names_lab_rat))  
    		text_file_name = os.path.join(save_dir, sampling+ '_' + lab_rat + '.txt')  
    		np.savetxt(text_file_name, names_lab_rat,  newline="\n", fmt="%s")
