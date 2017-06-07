import numpy as np

def create_history(target):
    # create history
    history = np.zeros(target.shape)
    # print(target.shape)
    history[:, 1:] = target[:, :-1]
    return history