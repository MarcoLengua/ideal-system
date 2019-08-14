import os

import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "1";
import keras
from keras.callbacks import ModelCheckpoint

from kinectgestures.preprocessing import default_evaluation_preprocessing, default_training_preprocessing
from kinectgestures.data import GestureDataset

from kinectgestures.models import create_model_2d, create_model_3d
from kinectgestures.test import test
from kinectgestures.transfer import create_model_vgg, create_model_vgg19, create_model_inception
from kinectgestures.util import save_history, save_config, get_checkpoint_filepath
from kinectgestures.visuals import plot_history
from kinectgestures.util import checkpoint_dir_exists, dataset_dir_exists, make_or_get_checkpoint_dir, get_dataset_dir, \
    ask_yes_no_question
from kinectgestures.metrics import motion_metric
import kinectgestures.metrics as metrics


def test_dummy_prediction(model):
    # specify batch_dimension
    batch_shape = (1,) + model.input_shape[1:]
    print(batch_shape)

    # random sample
    sample = np.random.rand(*batch_shape)
    print("Testing forward pass...")
    result = model.predict(sample, batch_size=1)
    print(result.shape)
    print("[DONE] Tested forward pass.")


def train(config):
    #####################
    ## Dataset
    is_2d_model = config['model'] in ("cnn2d", "vgg16", "vgg19", "inception")

    dataset_train = GestureDataset(get_dataset_dir(config),
                                   which_split='train',
                                   last_frame_only=is_2d_model,
                                   batch_size=config["batch_size"])
    dataset_validation = GestureDataset(get_dataset_dir(config),
                                        which_split='validation',
                                        last_frame_only=is_2d_model,
                                        batch_size=config["batch_size"])

    #####################
    ## Model
    kwargs = dict(out_shape=config["out_shape"],
                  in_shape=config["in_shape"],
                  config=config)
    if config['model'] == "cnn2d":
        model = create_model_2d(**kwargs)
    elif config['model'] == "cnn3d":
        model = create_model_3d(**kwargs)
    elif config['model'] == "vgg16":
        model = create_model_vgg(**kwargs)
    elif config['model'] == "vgg19":
        model = create_model_vgg19(**kwargs)
    elif config['model'] == "inception":
        model = create_model_inception(**kwargs)
    else:
        raise ValueError("Unknown model {}".format(config["model"]))

    model.summary()

    #####################
    ## Data augmentation
    dataset_train_augmented = default_training_preprocessing(config, dataset_train)
    dataset_validation_prepared = default_evaluation_preprocessing(config, dataset_validation)

    #####################
    ## Training setup
    metrics.BATCH_SIZE = config["batch_size"]
    model.compile(optimizer='sgd', loss='mse', metrics=[motion_metric])

    #####################
    # Callbacks
    filepath = get_checkpoint_filepath(config, pattern='weights.hdf5')
    checkpoint_saver = ModelCheckpoint(filepath,
                                       monitor='val_motion_metric',
                                       save_best_only=True,  # only overwrite if model is better
                                       mode='max'  # higher is better for this metric
                                       )

    #####################
    ## Go!

    # print("shape of train dataste augm:", dataset_train_augmented.shape)
    history = model.fit_generator(generator=dataset_train_augmented,
                                  validation_data=dataset_validation_prepared,
                                  callbacks=[checkpoint_saver],
                                  epochs=config["epochs"],
                                  verbose=2  # 0 = silent, 1 = progress bar, 2 = one line per epoch.
                                  )

    return model, history

def trainforhyperopt(config):
    #####################
    ## Dataset
    is_2d_model = config['model'] in ("cnn2d", "vgg16", "vgg19", "inception")
    print(config['batch_size'])
    print(get_dataset_dir(config))
    dataset_train = GestureDataset(get_dataset_dir(config),
                                   which_split='train',
                                   last_frame_only=is_2d_model,
                                   batch_size=config['batch_size'])
    dataset_validation = GestureDataset(get_dataset_dir(config),
                                        which_split='validation',
                                        last_frame_only=is_2d_model,
                                        batch_size=config['batch_size'])

    #####################
    ## Model
    kwargs = dict(out_shape=config["out_shape"],
                  in_shape=config["in_shape"],
                  config=config)
    if config['model'] == "cnn2d":
        model = create_model_2d(**kwargs)
    elif config['model'] == "cnn3d":
        model = create_model_3d(**kwargs)
    elif config['model'] == "vgg16":
        model = create_model_vgg(**kwargs)
    elif config['model'] == "vgg19":
        model = create_model_vgg19(**kwargs)
    elif config['model'] == "inception":
        model = create_model_inception(**kwargs)
    else:
        raise ValueError("Unknown model {}".format(config["model"]))

    model.summary()

    #####################
    ## Data augmentation
    dataset_train_augmented = default_training_preprocessing(config, dataset_train)
    dataset_validation_prepared = default_evaluation_preprocessing(config, dataset_validation)

    #####################
    ## Training setup
    metrics.BATCH_SIZE = config["batch_size"]
    model.compile(optimizer=config["optimizer"], loss='mse', metrics=[motion_metric])

    #####################
    # Callbacks
    filepath = get_checkpoint_filepath(config, pattern='weights.hdf5')
    checkpoint_saver = ModelCheckpoint(filepath,
                                       monitor='val_motion_metric',
                                       save_best_only=True,  # only overwrite if model is better
                                       mode='max'  # higher is better for this metric
                                       )

    #####################
    ## Go!

    # print("shape of train dataste augm:", dataset_train_augmented.shape)
    history = model.fit_generator(generator=dataset_train_augmented,
                                  validation_data=dataset_validation_prepared,
                                  callbacks=[checkpoint_saver],
                                  epochs=config["epochs"],
                                  verbose=2  # 0 = silent, 1 = progress bar, 2 = one line per epoch.
                                  )

    print(model.metrics_names)
    x_evaluate = []
    y_evaluate = []
    for i in range(len(dataset_validation_prepared)):
        batch, teachers = dataset_validation_prepared[i]
        x_evaluate.append(batch)
        y_evaluate.append(teachers)
    #print("meine samples evaluate als liste")
    #print(len(x_evaluate))
    x_eval = np.asarray(x_evaluate)
    #print(x_eval.shape)
    x_eval = x_eval.reshape(-1, *x_eval.shape[-3:])
    #print("meine samples als np array")
    #print(x_eval.shape)
    #print("meine teachers evaluate als liste")
    #print(len(y_evaluate))
    y_eval = np.asarray(y_evaluate)
    #print(y_eval.shape)
    y_eval = y_eval.reshape(-1, *y_eval.shape[-2:])
    #print("meine teachers als np array")
    #print(y_eval.shape)
    loss, motion_score = model.evaluate(x_eval, y_eval, batch_size=config['batch_size'])
    #print("mein score")
    #print(motion_score)

    return model, history, motion_score

def run_experiment(config):
    if not dataset_dir_exists(config):
        raise FileNotFoundError("Dataset not found at {}".format(config["dataset_path"]))

    if checkpoint_dir_exists(config):
        should_overwrite = ask_yes_no_question(
            "[PROMPT] Overwrite existing checkpoint? {}".format(config["checkpoint_path"]))
        if should_overwrite:
            make_or_get_checkpoint_dir(config)
        else:
            print("Skipping experiment...")
            return

    print("===================================")
    print("Starting experiment for model {}".format(config["model"]))
    print("===================================")

    # store visuals and files
    model, hist = train(config)
    history_dict = hist.history
    plot_history(config, history_dict)
    save_history(config, history_dict)
    save_config(config)

    # test data and write results
    test(model, config, store_output=True, evaluate_splits=True)


def main():
    pass


if __name__ == "__main__":
    main()
