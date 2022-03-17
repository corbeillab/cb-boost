import numpy as np

def change_label_to_minus(y):
    """
    Change the label 0 to minus one

    Parameters
    ----------
    y :

    Returns
    -------
    label y with -1 instead of 0

    """
    minus_y = np.copy(y)
    minus_y[np.where(y == 0)] = -1
    return minus_y

def _as_matrix(element):
    """ Utility function to convert "anything" to a Numpy matrix.
    """
    # If a scalar, return a 1x1 matrix.
    if len(np.shape(element)) == 0:
        return np.matrix([[element]], dtype=float)

    # If a nd-array vector, return a column matrix.
    elif len(np.shape(element)) == 1:
        matrix = np.matrix(element, dtype=float)
        if np.shape(matrix)[1] != 1:
            matrix = matrix.T
        return matrix

    return np.matrix(element, dtype=float)


def _as_column_matrix(array_like):
    """ Utility function to convert any array to a column Numpy matrix.
    """
    matrix = _as_matrix(array_like)
    if 1 not in np.shape(matrix):
        raise ValueError("_as_column_vector: input must be a vector")

    if np.shape(matrix)[0] == 1:
        matrix = matrix.T

    return matrix


def _as_line_matrix(array_like):
    """ Utility function to convert any array to a line Numpy matrix.
    """
    matrix = _as_matrix(array_like)
    if 1 not in np.shape(matrix):
        raise ValueError("_as_column_vector: input must be a vector")

    if np.shape(matrix)[1] == 1:
        matrix = matrix.T

    return matrix


def sign(array):
    """Computes the elementwise sign of all elements of an array. The sign function returns -1 if x <=0 and 1 if x > 0.
    Note that numpy's sign function can return 0, which is not desirable in most cases in Machine Learning algorithms.

    Parameters
    ----------
    array : array-like
        Input values.

    Returns
    -------
    ndarray
        An array with the signs of input elements.

    """
    signs = np.sign(array)

    signs[array == 0] = -1
    return signs