import os
import json
import glob

checkpoint_glob = "/informatik2/students/home/1lengua/hyperopt/hyperopt_2082019_cpu/*"
#checkpoint_glob = "/informatik2/students/home/1lengua/hyperopt_inception/*"
#checkpoint_glob = "/informatik2/students/home/1lengua/hyperopt_inception_2/*"

checkpoints = glob.glob(checkpoint_glob)
results = []

for checkpoint in checkpoints:
    history_path = os.path.join(checkpoint, "history.json")
    with open(history_path, "r") as fp:
        history = json.load(fp)
        max_val_score = history['validation_score']
        max_val_motion_metric = max(history['val_motion_metric'])
        max_value_dict = {'checkpoint_path':checkpoint, 'max_validation_score':max_val_score, 'max_val_motion_metric':max_val_motion_metric}
        results.append(max_value_dict)
maxValidationScore = max(results, key=lambda x:x['max_validation_score'])
maxValMotion = max(results, key=lambda x:x['max_val_motion_metric'])
print("Max Validation Score for Path")
print(str(maxValidationScore))
print("Max Val Motion Metric during Training for Path")
print(str(maxValMotion))