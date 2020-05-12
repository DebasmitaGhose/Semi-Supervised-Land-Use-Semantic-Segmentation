import numpy as np  
import matplotlib.pyplot as plt

lab_02 = np.array([45.70, 42.84, 40.33, 43.23, 49.07, 47.86, 39.40, 43.74, 38.11])
lab_05 = np.array([48.34, 44.66, 43.07, 49.07, 53.55, 46.26, 52.65, 51.04, 53.23])
lab_125 = np.array([51.45, 51.58, 51.07, 53.70, 51.40, 50.74, 51.88, 51.39, 56.58])
lab_2 = np.array([53.56, 60.26, 55.42, 61.02, 59.99, 56.83, 62.74, 55.17, 61.26])
lab_33 = np.array([63.03, 62.19, 62.93, 58.65, 60.31, 62.40, 60.27, 59.69, 59.47])
lab_5 = np.array([60.98, 59.69, 60.79, 63.52, 61.42, 61.21, 62.72, 62.37, 62.31])
lab_75 = np.array([61.02, 61.98, 62.28, 62.66, 63.86, 63.43, 62.04, 61.68, 62.95])
lab_95 = np.array([62.83, 62.62, 64.85, 63.42, 61.42, 61.78, 61.54, 60.89, 63.16])


lab_02_mean = np.mean(lab_02)
lab_05_mean = np.mean(lab_05)
lab_125_mean = np.mean(lab_125)
lab_2_mean = np.mean(lab_2)
lab_33_mean = np.mean(lab_33)
lab_5_mean = np.mean(lab_5)
lab_75_mean = np.mean(lab_75)
lab_95_mean = np.mean(lab_95)

lab_02_std = np.std(lab_02)
lab_05_std = np.std(lab_05)
lab_125_std = np.std(lab_125)
lab_2_std = np.std(lab_2)
lab_33_std = np.std(lab_33)
lab_5_std = np.std(lab_5)
lab_75_std = np.std(lab_75)
lab_95_std = np.std(lab_95)

means = [lab_02_mean, lab_05_mean, lab_125_mean, lab_2_mean, lab_33_mean, lab_5_mean, lab_75_mean, lab_95_mean]
print(means)
errors = [lab_02_std, lab_05_std, lab_125_std, lab_2_std, lab_33_std, lab_5_std, lab_75_std, lab_95_std]
#print(errors)
#print(np.subtract(means, errors))

labels = ['0.02', '0.05', '0.125', '0.2', '0.33', '0.5', '0.75', '0.95']
x_pos = np.arange(len(labels))

# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, means,
       yerr=errors,
       align='center',
       alpha=0.5,
       ecolor='black',
       capsize=10)
ax.set_ylabel('mean IoU')
ax.set_xlabel('Labeled Ratios')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_title('Mean IoU values for Different Labeled Ratios and their variation with different Self-Training Thresholds')
ax.set_ylim(0, 100)
ax.yaxis.grid(True)

plt.show()

pixel_acc = np.array([67.08, 70.54, 72.40, 77.06, 77.80, 78.03, 78.69, 78.71, 82.45])
mean_acc = np.array([66.12, 68.24, 72.75, 79.97, 76.21, 77.15, 78.28, 79.09, 80.03])
miou = np.array([49.07, 53.55, 56.48, 62.74, 63.03, 63.52, 63.86, 64.85, 68.84])

labels_line = ['0.02', '0.05', '0.125', '0.2', '0.33', '0.5', '0.75', '0.95', '1.0(baseline)']

x_axis = np.arange(0,9)
plt.ylim(0,100)
plt.xticks(x_axis, labels_line)  
plt.plot(x_axis, pixel_acc, marker='o', label='Pixel Accuracy')
plt.plot(x_axis, mean_acc, marker='o', label='Mean Accuracy')
plt.plot(x_axis, miou, marker='o', label='Mean IoU')
plt.legend(loc='lower right')
plt.xlabel('Labeled Ratios')
plt.ylabel('% (Metrics)')
plt.title('mIoU, Pixel Accuracy and Mean Accuracy for different Labeled Ratios')
plt.grid(True)
plt.show()