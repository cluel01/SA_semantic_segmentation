#    Edited by Sizhuo Li
#    Author: Ankit Kariryaa, University of Bremen


import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf


def get_loss(loss_fn, tversky_alpha_beta=None):
    """Wrapper function to allow only storing loss function name in config"""
    if loss_fn == "tversky":
        tversky_function = tversky
        # Update default arguments of tversky() with configured alpha beta.
        if tversky_alpha_beta:
            tversky_function.__defaults__ = tversky_alpha_beta
        return tversky_function
    elif loss_fn == "dynamic_tversky":
        tversky_function = dynamic_tversky
        # Update default arguments of tversky() with configured alpha beta.
        if tversky_alpha_beta:
            tversky_function.__defaults__ = tversky_alpha_beta
        return tversky_function
    elif loss_fn.lower() == "weighted_bce":
        return weighted_BCE
    elif loss_fn == "dice":
        return dice_loss
    elif loss_fn == "continuous_tversky":
        return continuous_tversky
    elif loss_fn == "BCE":
        return BCE
    else:
        # Used when passing string names of built-in tensorflow optimizers
        return loss_fn


def BCE(y_true, y_pred):
    # Extract ground truth
    true = y_true[..., 0]
    true = true[..., np.newaxis]

    # calculate the binary cross entropy
    return K.binary_crossentropy(true, y_pred)


def tversky(y_true, y_pred, alpha=0.50, beta=0.50, epsilon=1e-6):
    """
    Function to calculate the Tversky loss for imbalanced data
    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param alpha: weight of false positives
    :param beta: weight of false negatives
    :param weight_map:
    :return: the loss
    """

    y_t = y_true[..., 0]
    y_t = y_t[..., np.newaxis]
    # weights
    y_weights = np.abs(y_true[..., 1])
    y_weights = y_weights[..., np.newaxis]

    ones = 1
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_t
    g1 = ones - y_t

    tp = tf.reduce_sum(y_weights * p0 * g0)
    fp = alpha * tf.reduce_sum(y_weights * p0 * g1)
    fn = beta * tf.reduce_sum(y_weights * p1 * g0)

    numerator = tp + epsilon
    denominator = tp + fp + fn + epsilon
    score = numerator / denominator
    loss = 1.0 - tf.reduce_mean(score)

    return loss


def dynamic_tversky(y_true, y_pred, alpha=0.40, beta=0.60):
    """
    Function to calculate the Tversky loss for imbalanced data
    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param alpha: weight of false positives
    :param beta: weight of false negatives
    :param weight_map:
    :return: the loss
    """
    
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]

    # weights
    y_weights = y_true[...,1]
    y_weights = y_weights[...,np.newaxis]

    # cover
    y_cover = y_true[...,2]
    # y_cover = y_cover[..., np.newaxis]
    gamma = 1 + (tf.reduce_mean(y_cover)/200)
    # print(gamma)


    ones = 1 
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_t
    g1 = ones - y_t
    tp = tf.reduce_sum(y_weights * p0 * g0)
    fp = alpha * tf.reduce_sum(y_weights * p0 * g1)
    fn = beta * tf.reduce_sum(y_weights * p1 * g0)

    EPSILON = 0.000001
    numerator = tp + EPSILON
    denominator = tp + fp + fn + EPSILON
    score = numerator / denominator
    tversky_loss = (1.0 - tf.reduce_mean(score))**(1/gamma)
    if tversky_loss is None:
        print('Nan loss:')
        print(gamma, numerator, denominator)
    # print(tversky_loss)
    # assert False
    return tversky_loss


def continuous_tversky(y_true, y_pred, alpha=0.5, beta=0.5, gamma=0.2, epsilon=1e2):
    loss = tversky(y_true, y_pred, alpha, beta, epsilon) + tversky(1-y_true, 1-y_pred, 1-alpha, 1-beta, epsilon) * gamma
    # loss = (loss / (1 + gamma))

    return loss

def weighted_BCE(y_true, y_pred, weight_zero=0.25, weight_one=1):
    """
    Calculates weighted binary cross entropy. The weights are fixed.

    This can be useful for unbalanced catagories.

    Adjust the weights here depending on what is required.

    For example if there are 10x as many positive classes as negative classes,
        if you adjust weight_zero = 1.0, weight_one = 0.1, then false positives
        will be penalize 10 times as much as false negatives.
    """

    # Extract ground truth
    true = y_true[..., 0]
    true = true[..., np.newaxis]

    # calculate the binary cross entropy
    bin_crossentropy = K.binary_crossentropy(true, y_pred)

    # get weights from cover band
    y_cover = y_true[...,2]
    # y_cover = y_cover[..., np.newaxis]
    weight_one = 1 - (tf.reduce_mean(y_cover)/100*0.9)
    weight_zero = 0.1 + (tf.reduce_mean(y_cover)/100*0.9)

    # apply the weights
    weights = true * weight_one + (1. - true) * weight_zero
    weighted_bin_crossentropy = weights * bin_crossentropy

    return K.mean(weighted_bin_crossentropy)


def accuracy(y_true, y_pred):
    """compute accuracy"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.equal(K.round(y_t), K.round(y_pred))

def dice_coef(y_true, y_pred, smooth=0.0000001):
    """compute dice coef"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    intersection = K.sum(K.abs(y_t * y_pred), axis=-1)
    union = K.sum(y_t, axis=-1) + K.sum(y_pred, axis=-1)
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=-1)

def dice_loss(y_true, y_pred):
    """compute dice loss"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return 1 - dice_coef(y_t, y_pred)

def true_positives(y_true, y_pred):
    """compute true positive"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.round(y_t * y_pred)

def false_positives(y_true, y_pred):
    """compute false positive"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.round((1 - y_t) * y_pred)

def true_negatives(y_true, y_pred):
    """compute true negative"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.round((1 - y_t) * (1 - y_pred))

def false_negatives(y_true, y_pred):
    """compute false negative"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.round((y_t) * (1 - y_pred))

def sensitivity(y_true, y_pred):
    """compute sensitivity (recall)"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    tp = true_positives(y_t, y_pred)
    fn = false_negatives(y_t, y_pred)
    return K.sum(tp) / (K.sum(tp) + K.sum(fn))

def specificity(y_true, y_pred):
    """compute specificity (precision)"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    tn = true_negatives(y_t, y_pred)
    fp = false_positives(y_t, y_pred)
    return K.sum(tn) / (K.sum(tn) + K.sum(fp))
