import os
from time import time
import datetime
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from kinectgestures.metrics import motion_metric
import kinectgestures.metrics as metrics
from kinectgestures.preprocessing import default_evaluation_preprocessing, default_training_preprocessing
from kinectgestures.data import GestureDataset


from keras.applications.vgg19 import VGG19

from keras.layers import Dense, Flatten, Reshape, Input, Lambda, Dropout, BatchNormalization
from keras.models import Model
import keras.backend as K
from keras.utils import plot_model

#pathname to dataset
dirname = os.path.dirname(__file__)
dataset_dir = os.path.abspath(os.path.join(dirname, "../../datasets/kinect-gestures-v1-240x320"))

#dummy configuration
#pretrained= True
#num_features = 384
#batch_size = 16
#optimizer = 'adam'
#epochs = 1
config_dict= {"in_shape": [120,160,1],"preprocessing_scale": [144,192],"preprocessing_scale_teacher": [60,80]}

def output_of_stack_channels(input_shape):
    return input_shape[0], input_shape[1], input_shape[2], 3

def stack_channels(x):
    stacked = K.stack([x, x, x], axis=3)
    squeezed = K.squeeze(stacked, axis=-1)
    return squeezed

def create_model_vgg19(params, in_shape=(120, 160, 1)):
    out_height, out_width = (60,80)

    # increase channels from 1 -> 3
    input_layer = Input(in_shape)
    input_stacked = Lambda(stack_channels, output_shape=output_of_stack_channels)(input_layer)

    # learn a normalization, roughly map distributions Kinect data -> RGB
    input_normalized = BatchNormalization()(input_stacked)

    # pre-trained VGG19 feature extraction
    weights = 'imagenet'
    base_model = VGG19(weights=weights, include_top=False)
    plot_model(base_model, to_file='model_vgg19.png', show_shapes=True, show_layer_names=True)
    # , input_tensor=input_normalized)
    # x = base_model.output
    base_model.summary()
    features = Model(inputs=input_layer, outputs=base_model(input_normalized))
    x = features.output

    # add MLP on top
    x = Flatten()(x)
    num_features = int(params['num_features'])
    x = Dense(num_features)(x)
    x = Dropout(0,5)(x)
    x = Dense(out_height * out_width)(x)
    x = Reshape((out_height, out_width))(x)

    # combine into single model objecthttps://keras.io/
    model = Model(inputs=input_layer, outputs=x)

    # freeze VGG19 layers
    if params['pretrained']:
        for layer in base_model.layers:
            layer.trainable = False

    return model

def train(params):
    #####################
    #Dataset

    dataset_train = GestureDataset(dataset_dir,
                                   which_split='train',
                                   last_frame_only=True,
                                   batch_size=params['batch_size'])
    dataset_validation = GestureDataset(dataset_dir,
                                        which_split='validation',
                                        last_frame_only=True,
                                        batch_size=params['batch_size'])

    #####################
    ## Model
    model = create_model_vgg19(params)


    model.summary()

    #####################
    ## Data augmentation
    dataset_train_augmented = default_training_preprocessing(config_dict, dataset_train)
    dataset_validation_prepared = default_evaluation_preprocessing(config_dict, dataset_validation)

    #####################
    ## Training setup
    metrics.BATCH_SIZE = params['batch_size']
    model.compile(optimizer=params['optimizer'], loss='mse', metrics=[motion_metric])

    #####################
    # Callbacks
    # filepath = get_checkpoint_filepath(config, pattern='weights.hdf5')
    # checkpoint_saver = ModelCheckpoint(filepath,
    #                                    monitor='val_motion_metric',
    #                                    save_best_only=True,  # only overwrite if model is better
    #                                    mode='max'  # higher is better for this metric
    #                                    )

    #####################
    ## Go!

    # print("shape of train dataste augm:", dataset_train_augmented.shape)
    history = model.fit_generator(generator=dataset_train_augmented,
                                  validation_data=dataset_validation_prepared,
                                  #callbacks=[checkpoint_saver],
                                  epochs=params['epochs'],
                                  verbose=2  # 0 = silent, 1 = progress bar, 2 = one line per epoch.
                                  )

    print(model.metrics_names)
    x_evaluate = []
    y_evaluate = []
    for i in range(len(dataset_validation_prepared)):
        batch, teachers = dataset_validation_prepared[i]
        x_evaluate.append(batch)
        y_evaluate.append(teachers)
    print("meine samples für evaluate als liste")
    print(len(x_evaluate))
    x_eval = np.asarray(x_evaluate)
    print(x_eval.shape)
    x_eval = x_eval.reshape(-1, *x_eval.shape[-3:])
    print("meine samples als np array")
    print(x_eval.shape)
    print("meine teachers für evaluate als liste")
    print(len(y_evaluate))
    y_eval = np.asarray(y_evaluate)
    print(y_eval.shape)
    y_eval = y_eval.reshape(-1, *y_eval.shape[-2:])
    print("meine teachers als np array")
    print(y_eval.shape)
    loss, motion_score = model.evaluate(x_eval, y_eval, batch_size=params['batch_size'])
    print("mein score")
    print(motion_score)

    return model, history, motion_score

def objective(params):
    print('Params testing: ', params)
    model, hist, score = train(params)
    history_dict = hist.history
    # test data and write results
    print("Meine score für diese Einstellungen: \n")
    print(score)
    return {'loss': score*(-1), 'status': STATUS_OK}

space = {
    'batch_size': hp.choice('batch_size', [2,4,8,12,16,24,32]),
    'epochs': hp.choice('epochs', [10, 20, 30, 40, 50, 60, 70 ,80, 90, 100]),
    'optimizer': hp.choice('optimizer', ['sgd', 'adam', 'rmsprop']),
    'num_features': hp.qloguniform('num_features', 4, 8, 10),
    'pretrained': hp.choice('pretrained', [True, False])
}
trials = Trials()
print("Startzeit:")
print(str(datetime.datetime.now()))
best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=100)
print(best)
print("Endzeit:")
print(str(datetime.datetime.now()))
print(trials.best_trial)