import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os

def clear_plot():
    plt.gcf().clear()

def plot_history(checkpoint_directory):

    # prepare paths
    dirname = "/Users/djikorange/checkpoints/inception_sgd/"
    checkpoint_dir = os.path.join(dirname, checkpoint_directory)
    path_fig_loss_train = os.path.join(checkpoint_dir, "plot_loss_train.png")
    path_fig_loss_validation = os.path.join(checkpoint_dir, "plot_loss_validation.png")
    path_fig_motion_train = os.path.join(checkpoint_dir, "plot_motion_train.png")
    path_fig_motion_validation = os.path.join(checkpoint_dir, "plot_motion_validation.png")

    #load json to dict
    with open(os.path.join(checkpoint_dir, "history.json"), "r") as fp:
        metrics = json.load(fp)

    # Plot training accuracy values
    plt.plot(metrics['motion_metric'])
    plt.title('Motion metric train')
    plt.ylabel('Motion')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.savefig(path_fig_motion_train)
    clear_plot()

    # Plot validation accuracy values
    plt.plot(metrics['val_motion_metric'])
    plt.title('Motion metric validation')
    plt.ylabel('Motion')
    plt.xlabel('Epoch')
    plt.legend(['Validation'], loc='upper left')
    plt.savefig(path_fig_motion_validation)
    clear_plot()

    # Plot training loss values
    plt.plot(metrics['loss'])
    plt.title('Loss metric train')
    plt.ylabel('Motion')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.savefig(path_fig_loss_train)
    clear_plot()

    # Plot validation loss values
    plt.plot(metrics['val_loss'])
    plt.title('Loss metric validation')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Validation'], loc='upper left')
    plt.savefig(path_fig_loss_validation)
    clear_plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run plots for a given checkpoint directory")
    parser.add_argument("path", help="name of the checkpoiont directory")
    args = parser.parse_args()
    plot_history(args.path)
