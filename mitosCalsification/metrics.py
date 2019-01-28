import keras.backend as K

def mitos_fscore(y_true, y_pred):
    """
    Calculate the f1 score based on the class 0 (mitosis). It uses the
    functions for tensors provided by Tensorflow/Theano. And can be used as
    metric for keras model
    :param y_true: True labels. Theano/TensorFlow tensor.
    :param y_pred: False labels. Theano/TensorFlow tensor.
    :return: f1 score as a tensor
    """
    # true_output = K.argmax(y_true)
    # pred_output = K.argmax(y_pred)
    true_output = y_true
    pred_output = K.round(y_pred)
    index_true_class = K.equal(true_output, 0)
    floatx = K.floatx()
    if K.backend() == 'theano':
        # compares index_true_class == false
        inverted_true_class = K.equal(index_true_class, 0)
        index_pred_class = K.equal(pred_output, 0)
    else:
        inverted_true_class =  K.equal(index_true_class, False)
        inverted_true_class = K.cast(inverted_true_class, floatx)
        index_pred_class = K.equal(pred_output, False)
        index_pred_class = K.cast(index_pred_class, floatx)
        index_true_class = K.cast(index_true_class, floatx)

    false_positive = K.sum(inverted_true_class * index_pred_class)
    true_positives = K.sum(K.cast(K.equal(true_output, pred_output), floatx)* index_true_class)
    false_negatives = K.sum(index_true_class) - true_positives

    precision = true_positives / (true_positives + false_positive + K.epsilon())
    recall = true_positives / (true_positives + false_negatives+K.epsilon())
    return 2 * (precision * recall)/(precision + recall+K.epsilon())
    #print(K.eval(fscore))

def fscore(y_true, y_pred, not_extracted = 0):
    from sklearn.metrics import confusion_matrix

    epsilon = 1e-07
    conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    true_positives = conf_mat[0,0]
    false_negatives = conf_mat[0,1] + not_extracted
    false_positives = conf_mat[1,0]

    precision = true_positives / (true_positives + false_positives + epsilon)
    recall = true_positives / (true_positives + false_negatives + epsilon)
    return 2 * (precision * recall)/(precision + recall + epsilon)

def print_conf_matrix(y_true, y_pred, not_extracted=0):
    from sklearn.metrics import confusion_matrix

    conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    true_positives = conf_mat[0, 0]
    false_negatives = conf_mat[0, 1] + not_extracted
    false_positives = conf_mat[1, 0]
    true_negatives = conf_mat[1,1]

    print('\t\tMitosis\tNo-mitosis')
    print('Mitosis\t\t{}\t{}'.format(true_positives, false_negatives))
    print('No-mitosis\t{}\t{}'.format(false_positives, true_negatives))

if __name__ == '__main__':
    m1 = [[1,0],[1,0],[0,1], [1,0], [0,1], [0,1], [1,0],[0,1],[1,0],[1,0],[1,0]]
    m2 = [[1,0], [1,0], [1,0], [0,1], [0,1], [0,1], [1,0], [1,0],[1,0],[0 ,1],[0,1]]

    import numpy as np
    pred = K.variable(m1)
    true = K.variable(m2)
    res = K.eval(mitos_fscore(true, pred))
    print(np.argmax(m2, 1))
    print(res)
    i=0
