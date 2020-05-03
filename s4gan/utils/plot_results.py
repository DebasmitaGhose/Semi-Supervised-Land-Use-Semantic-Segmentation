import numpy as np  
import matplotlib.pyplot as plt

lab_02 = np.array([45.70, 42.84, 40.33, 43.23, 39.40, 43.74, 38.11])
lab_05 = np.array([48.34, 44.66, 43.07, 46.26, 52.65, 51.04, 53.23])
lab_125 = np.array([51.45, 51.58, 51.07])
lab_2 = np.array([53.70, 55.17])
lab_33 = np.array([63.03, 62.19, 62.93, 58.65, 60.31, 60.27, 59.69, 59.47])
lab_5 = np.array([60.98, 59.69, 60.79, 63.52, 61.42, 61.21, 62.72, 62.37, 62.31])
lab_75 = np.array([61.02, 61.98, 62.28, 62.66, 63.43, 62.04, 61.68, 62.95])
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
#print(means)
errors = [lab_02_std, lab_05_std, lab_125_std, lab_2_std, lab_33_std, lab_5_std, lab_75_std, lab_95_std]
#print(errors)

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
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_title('mean IoU for Different Labeled Ratios')
ax.set_ylim(0, 100)
ax.yaxis.grid(True)

plt.show()