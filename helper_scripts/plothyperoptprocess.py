import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib as mpl
## agg backend is used to create plot as a .png file
#mpl.use('agg')
import json
parameter = 'pretrained'
plot = 'barplot'
architecture = 'inception'
if architecture == 'inception':
    with open("../helper_data/inceptionhyperoptparameters.json", "r") as fp:
        hyperopt_dict = json.load(fp)
elif architecture == 'vgg19':
    with open("../helper_data/vgg19hyperoptparameters.json", "r") as fp:
        hyperopt_dict = json.load(fp)
x_values = []
y_values = []

for n, value in enumerate(hyperopt_dict):
    y_values.append(hyperopt_dict[n]['validation_score'])
    x_values.append(hyperopt_dict[n][parameter])

baskets = sorted(set(x_values))
means = []
stds = []
data_baskets = []
x_values, y_values = zip(*sorted(zip(x_values, y_values)))
x_pos = np.arange(len(baskets))
for basket in baskets:
    temp_list = []
    for n, value in enumerate(x_values):
        if basket == value:
            temp_list.append(y_values[n])
    means.append(np.mean(np.array(temp_list)))
    stds.append(np.std(np.array(temp_list)))
    data_baskets.append(temp_list)
#Create a figure instance
fig = plt.figure(1, figsize=(9, 6))
#Create an axes instance
ax = fig.add_subplot(111)
ax.set_title('Hyperparameter Optimization - Pretrained')
ax.set_ylabel('Motion Metric Accuracy')
ax.set_xlabel('Pretrained')
if plot == 'boxplot':
    ##BOXPLOT##
    # Create the boxplot
    bp = ax.boxplot(data_baskets)
    ax.set_xticklabels(baskets)
    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)
    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)
    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)
    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)
    # Save the figure
    fig.savefig('boxplot_hyperopt_'+architecture+'_'+parameter+'.png', bbox_inches='tight')

elif plot == 'barplot':
    ##BARPLOT##
    #Build the plot
    ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(baskets)
    ax.yaxis.grid(True)
    # Save the figure and show
    plt.tight_layout()
    plt.savefig('barplot_hyperopt_'+architecture+'_'+parameter+'.png')

elif plot == 'scatterplot':
    ##SCATTERPLOT##
    #Build the plot#
    # Save the figure
    ax.scatter(x_values,y_values)
    fig.savefig('scatterplot_hyperopt_'+architecture+'_' + parameter + '.png', bbox_inches='tight')

plt.show()