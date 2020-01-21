import numpy as np
import matplotlib.pyplot as plt
import json
metrics_dicts = {}
with open("/home/marco//HyperOptBachelor/cnn2d_100epochs_testmodel/metrics_test.json", "r") as fp:
    metrics_dicts['cnn2d'] = json.load(fp)
with open("/home/marco//HyperOptBachelor/cnn3d_100epochs_testmodel_3dreshape/metrics_test.json", "r") as fp:
    metrics_dicts['cnn3d']  = json.load(fp)
with open("/home/marco//HyperOptBachelor/vgg19_100epochs_testmodel_lowerlearningrate/metrics_test.json", "r") as fp:
    metrics_dicts['vgg19']  = json.load(fp)
with open("/home/marco//HyperOptBachelor/inception_100epochs_testmodel_lowerlearningrate/metrics_test.json", "r") as fp:
    metrics_dicts['inceptionresnet']  = json.load(fp)

keytoplot = '1'
baskets = ['cnn2d', 'cnn3d', 'vgg19', 'inceptionresnet']
x_pos = np.arange(len(baskets))
values = []
for model in baskets:
    categories = list(metrics_dicts[model]['category'].keys())
    counts = list(metrics_dicts[model]['count'].keys())
    if keytoplot in categories:
        values.append(float(metrics_dicts[model]['category'][keytoplot]))
    elif keytoplot in counts:
        values.append(float(metrics_dicts[model]['count'][keytoplot]))
    elif keytoplot == 'overall':
        values.append(float(metrics_dicts[model]['overall']))
#Create a figure instance
fig = plt.figure(1, figsize=(9, 6))
#Create an axes instance
ax = fig.add_subplot(111)
ax.set_title(keytoplot)
ax.set_ylabel('Motion Metric Accuracy')
ax.set_xlabel('Model')
##BARPLOT##
#Build the plot
ax.bar(x_pos, values, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_xticks(x_pos)
ax.set_xticklabels(baskets)
ax.yaxis.grid(True)
# Save the figure and show
plt.tight_layout()
#plt.savefig('testbarplotresults.png')
plt.show()