import numpy as np

def compute_mse_from_str(type, string1, string2):
    array1 = string1.split(', ')
    array1 = np.array([float(a) for a in array1 if a.isnumeric()])

    array2 = string2.split(', ')
    array2 = np.array([float(a) for a in array2 if a.isnumeric()])

    if len(array1) > len(array2):
        array1 = array1[:len(array2)]
    elif len(array2) > len(array1):
        array1 = np.concatenate([array1, np.zeros(len(array2) - len(array1))])

    if type == 'ecg':
        array1 /= 1000
        array2 /= 1000 
        return ((array1 - array2)**2).mean()
    else:
        raise NotImplementedError("Other modalities haven't been implemeted")
