import datetime
import argparse
from kinectgestures.train import trainfforvalidationandtest
from kinectgestures.util import save_history, save_config
from kinectgestures.visuals import plot_history
from kinectgestures.util import load_json
from kinectgestures.test import test

def train(config_path):
    config = load_json(config_path)
    run_train_validation_test_final(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run training for a given config file")
    parser.add_argument("path", help="Path to config json")
    args = parser.parse_args()
    train(args.path)

def run_train_validation_test_final(config)

 # store visuals and files
    print("Startzeit:")
    starttime = datetime.datetime.now()
    print(str(starttime))
    model, hist, score = trainfforvalidationandtest(config)
    history_dict = hist.history
    history_dict['validation_score'] = score
    plot_history(config, history_dict)
    save_history(config, history_dict)
    save_config(config)
    print("Meine final score für Validation set: \n")
    print(score)
    # test data and write results
    test(model, config, store_output=True, evaluate_splits=True)
    print("Endzeit:")
    endtime = datetime.datetime.now()
    print(str(endtime))
    totaltime = endtime- starttime
    print("Zeitinsgesamt:")
    print(str(totaltime))
