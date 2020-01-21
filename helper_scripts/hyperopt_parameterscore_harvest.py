import json
import os
import glob
architecture = 'vgg19'
if architecture == 'inception':
    # Mac
    # checkpoint_glob = "/Users/djikorange/hyperopt_2082019_cpu/*"
    # Home Linux#
    #checkpoint_glob = "/home/marco/HyperOptBachelor/vgg19_net_hyperopt_final/*"
    checkpoint_glob = "/home/marco/HyperOptBachelor/inception_net_hyperopt_final/*"
    # Uni linux
elif architecture == 'vgg19':
    # Mac
    # checkpoint_glob = "/Users/djikorange/hyperopt_2082019_cpu/*"
    # Home Linux#
    checkpoint_glob = "/home/marco/HyperOptBachelor/vgg19_net_hyperopt_final/*"
    # checkpoint_glob = "/home/marco/HyperOptBachelor/inception_net_hyperopt_final/*"
    # Uni linux
checkpoints = glob.glob(checkpoint_glob)
results = []
trial_dict = {}

for checkpoint in checkpoints:
    checkpoint_dir = checkpoint.rsplit('/', 1)[-1]
    temp_list = checkpoint_dir.split('opt_', 1)
    pretrained = temp_list[0]
    temp_list = temp_list[1].split('batch_', 1)
    optimizer = temp_list[0]
    temp_list = temp_list[1].split('bottleneck_', 1)
    batch_size = int(temp_list[0])
    bottleneck = int(float(temp_list[1]))
    trial_dict = {'pretrained': pretrained, 'optimizer': optimizer, 'batch_size': batch_size, 'bottleneck': bottleneck}
    history_path = os.path.join(checkpoint, "history.json")
    with open(history_path, "r") as fp:
        history = json.load(fp)
        trial_dict['checkpoint_path'] = checkpoint
        trial_dict['validation_score'] = history['validation_score']
        trial_dict['loss'] = history['loss'][-1]
        trial_dict['val_loss'] = history['val_loss'][-1]
        trial_dict['motion_metric'] = history['motion_metric'][-1]
        trial_dict['val_motion_metric'] = history['val_motion_metric'][-1]
        results.append(trial_dict)

with open('../helper_data/'+architecture+'hyperoptparameters.json', 'w') as fp:
    json.dump(results, fp)