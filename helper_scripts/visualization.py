from keras.models import load_model
from vis.utils import utils
from matplotlib import pyplot as plt
from vis.visualization import visualize_saliency, overlay
from kinectgestures.metrics import motion_metric


model = load_model('/home/marco/HyperOptBachelor/vgg19_final100epochs/weights.hdf5', custom_objects={'motion_metric': motion_metric})
print(model.summary())
print([layer.name for layer in model.get_layer('vgg19').layers])
layer_idx = utils.find_layer_idx(model, 'block5_conv4')
img = utils.load_img('/home/marco/HyperOptBachelor/vgg19_final100epochs/out/test-96-teacher.png')
f, ax = plt.subplots(1,1)
grads = visualize_saliency(model,-5,filter_indices=10,seed_input=img)
ax[1].imshow(grads, cmap='jet')