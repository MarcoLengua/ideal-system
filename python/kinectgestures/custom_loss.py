import tensorflow as tf
from keras import backend as K

# set this from outside before evaluating the metric, to allow for other batch sizes. it's a bit hacky, I know
BATCH_SIZE = 16

def is_motion_at_right_location(y_true, y_pred, threshold):
    # flatten frame representation for easier indexing: (height, width) -> (height*width)
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    number_of_locations = 2000
    # where is the "active" actor located? in the network output?
    values, active_locations = tf.nn.top_k(y_pred_flat, number_of_locations)

    # is this also "active" in the teacher?
    actual_values = tf.gather(y_true_flat, active_locations)
    results = tf.map_fn(lambda x: tf.cast(tf.greater(x, tf.constant(threshold)), dtype=tf.float32), actual_values)
    metric_mean = -tf.reduce_mean(results)

    return metric_mean


def contains_motion(frame, threshold):
    return tf.greater(tf.count_nonzero(frame > threshold), 0)


def motion_metric_single_element_motion_in_teacher(y_true, y_pred, threshold):
    """
    Apply the motion metric to a single element, NOT a full batch.
    This method assumes that the teacher frame HAS motion in it.
    """

    return is_motion_at_right_location(y_true, y_pred, threshold)


def motion_metric_single_element_no_motion_in_teacher(y_true, y_pred, threshold):
    """
    Apply the motion metric to a single element, NOT a full batch.
    This method assumes that the teacher frame HAS NO motion in it.
    """

    # no motion in the teacher: make sure result also contains none
    cond_no_motion_in_teacher = tf.equal(tf.count_nonzero(y_true > threshold), 0)
    cond_no_motion_in_prediction = tf.equal(tf.count_nonzero(y_pred > threshold), 0)

    return tf.cond(tf.logical_and(cond_no_motion_in_teacher, cond_no_motion_in_prediction),
                   lambda: -1.0,
                   lambda: 0.0)


def motion_loss(y_true, y_pred, threshold=-0.3):
    motion_in_teacher = contains_motion(y_true, threshold)

    return tf.cond(motion_in_teacher,
                   lambda: motion_metric_single_element_motion_in_teacher(y_true, y_pred, threshold),
                   lambda: motion_metric_single_element_no_motion_in_teacher(y_true, y_pred, threshold))
