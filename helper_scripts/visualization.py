from keras.models import load_model
from vis.utils import utils
from matplotlib import pyplot as plt
from vis.visualization import visualize_saliency, overlay


model = load_model('/home/marco/HyperOptBachelor/final_inception_lowerlearningrate/weights.hdf5')
layer_idx = utils.find_layer_idx(model, -5)
img = utils.load_img('/home/marco/HyperOptBachelor/final_inception_lowerlearningrate/out/test-96-teacher.png')
f, ax = plt.sublot(1,1)
grads = visualize_saliency(model,layer_idx,filter_indices=10,seed_input=img)
ax[1].imshow(grads, cmap='jet')