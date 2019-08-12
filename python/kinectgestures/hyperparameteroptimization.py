import datetime
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from kinectgestures.train import trainforhyperopt
from kinectgestures.util import save_history, save_config, get_checkpoint_filepath
from kinectgestures.visuals import plot_history



def objective(params):
    print('Params testing: ', params)
    outpath = "../../checkpoints/vgg19/"+str(params['pretrained'])+"opt_"+params['optimizer']+"epochs_"+params['epochs']+"batch_"+params['batch_size']+"bottleneck_"+params['num_features']
    config = {
    "dataset_path": "../../datasets/kinect-gestures-v1-240x320",
    "checkpoint_path": outpath,
    "out_shape": [60, 80],  # exactly half the input size, original: [120, 160]
    "in_shape": [120, 160],
    "preprocessing_scale": [180, 220],
    "preprocessing_scale_teacher": [60, 80],
    "batch_size": params['batch_size'],
    "epochs": params['epochs'],
    "model": "vgg19",
    "dropout_rate": 0.5,
    "num_features": params['num_features'],
    "pretrained": params['pretrained'],
    "optimizer": params['optimizer']
    }
    model, hist, score = trainforhyperopt(config)
    history_dict = hist.history
    history_dict['validation_score']=score
    plot_history(config, history_dict)
    save_history(config, history_dict)
    save_config(config)
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
starttime = datetime.datetime.now()
print(str(starttime))
best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=100)
print(best)
print("Endzeit:")
endtime = datetime.datetime.now()
print(str(endtime))
totaltime = endtime- starttime
print("Zeitinsgesamt:")
print(str(totaltime))
print(trials.best_trial)