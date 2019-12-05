import json
import os
import glob

checkpoint_glob = "/Users/djikorange/hyperopt_2082019_cpu/*"
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
    batch_size = temp_list[0]
    bottleneck = temp_list[1]
    trial_dict = {'pretrained': pretrained, 'optimizer': optimizer, 'batch_size': batch_size, 'bottleneck': bottleneck}
    history_path = os.path.join(checkpoint, "history.json")
    with open(history_path, "r") as fp:
        history = json.load(fp)
        val_score = history['validation_score']
        last_motion_metric = history['motion_metric'][-1]
        last_val_motion_metric = history['val_motion_metric'][-1]
        last_loss = history['loss'][-1]
        last_val_loss = history['val_loss'][-1]
        trial_dict = {'checkpoint_path':checkpoint, 'validation_score':val_score, 'loss': last_loss, 'val_loss': last_val_loss,'motion_metric':last_motion_metric,'val_motion_metric':last_val_motion_metric}
        results.append(trial_dict)

print(results)