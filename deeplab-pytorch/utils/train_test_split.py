import numpy as np
from sklearn.model_selection import train_test_split

text_file = 'Satellite_Images/ImageSets/all_images.txt'
train_text_file = 'Satellite_Images/ImageSets/train.txt'
test_text_file = 'Satellite_Images/ImageSets/test.txt'

train_file = open(train_text_file, 'r+')
test_file = open(test_text_file, 'r+')

f = open(text_file, 'r')
images = []
for x in f:
	images.append(x)

images = np.array(images)
train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

#print(train_images)
#print(test_images)

for i in train_images:
	train_image = i.split('\n')[0]
	train_image = train_image + '\n'
	train_file.write(train_image)

for i in test_images:
        test_image = i.split('\n')[0]
        test_image = test_image + '\n'
        test_file.write(test_image)


