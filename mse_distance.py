import numpy as np

def compute_mse_from_str(max_val, array1, array2):
    array1 = np.array(array1)
    array2 = np.array(array2)
    
    if len(array1) > len(array2):
        array1 = array1[:len(array2)]
    elif len(array2) > len(array1):
        array1 = np.concatenate([array1, np.zeros(len(array2) - len(array1))])


    array1 = (array1 * max_val)/100.0
    array2 = (array2 * max_val)/100.0

    squared_diffs = (array1 - array2) ** 2

    mse = np.mean(squared_diffs)

    return mse


def compute_mae(max_val, array1, array2):
    array1 = np.array(array1)
    array2 = np.array(array2)
    
    if len(array1) > len(array2):
        array1 = array1[:len(array2)]
    elif len(array2) > len(array1):
        array1 = np.concatenate([array1, np.zeros(len(array2) - len(array1))])

    array1 = (array1 * max_val) / 100.0
    array2 = (array2 * max_val) / 100.0

    absolute_diffs = abs(array1 - array2)

    mae = sum(absolute_diffs) / len(absolute_diffs)

    return mae
