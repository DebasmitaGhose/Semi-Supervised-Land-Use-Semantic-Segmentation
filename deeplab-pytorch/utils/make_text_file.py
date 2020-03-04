import os

file1 = open("images.txt","r+")  


for root, dirs, files in os.walk('DLRSD/'):
	for file in files:
		with open(os.path.join(root, file), "r") as auto:
			data = file.split('.')[0] + '\n'
			file1.write(data)
