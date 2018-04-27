# original code from: https://www.kaggle.com/davidthaler/gap-metric
# modified to adapt chainer
import numpy as np
import pandas as pd
import chainer


def GAP_vector(x, t, return_df=False):
    '''
    Compute Global Average Precision (aka micro AP), the metric for the
    Google Landmark Recognition competition.
    This function takes predictions, labels and confidence scores as vectors.
    In both predictions and ground-truth, use None/np.nan for "no label".

    Args:
        pred: vector of integer-coded predictions
        conf: vector of probability or confidence scores for pred
        true: vector of integer-coded labels for ground truth
        return_x: also return the data frame used in the calculation

    Returns:
        GAP score
    '''
    x_cpu = chainer.backends.to_cpu(x).array
    pred = np.argmax(x_cpu, axis=1)
    conf = x_cpu[np.arange(len(x_cpu)), pred]
    df = pd.DataFrame({'pred': pred, 'conf': conf, 'true': t})
    df.sort_values('conf', ascending=False, inplace=True, na_position='last')
    df['correct'] = (df.true == df.pred).astype(int)
    df['prec_k'] = df.correct.cumsum() / (np.arange(len(df)) + 1)
    df['term'] = df.prec_k * df.correct
    gap = df.term.sum() / df.true.count()
    if return_df:
        return gap, df
    else:
        return gap
