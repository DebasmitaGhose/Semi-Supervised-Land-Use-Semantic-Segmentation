import os
import shutil

dest_images = 'Satellite_Images/UCMerced_Images/'
source_images = 'UCMerced_LandUse/Images/'

for root, dirs, files in os.walk(source_images):
	for file in files:
		shutil.move(root+'/'+file, dest_images)

dest_labels = 'Satellite_Images/UCMerced_Labels/'
source_labels = 'DLRSD/Images/'

for root, dirs, files in os.walk(source_labels):
        for file in files:
                shutil.move(root+'/'+file, dest_labels)

