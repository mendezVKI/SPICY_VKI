"""
Utilities functions to check the input of various SPICY class methods

Authors: Manuel Ratz
"""

import numpy as np


def check_data(data, name, dimension=None):
    """
    Function to check that the arrays in data are numpy arrays of the same length and match the
    given dimension

    Parameters
    ----------

    data : list of 1D numpy.ndarray
        Typically these are scattered data points (such as the coordinates, constraints, or an evaluation data).

    name : str
        Variable name in the code that calls this function.

    dimension : str
        Dimension of the problem. Must be either '1D', '2D' or '3D'.

    """

    em_list_len = 'Input \'' + name + '\' has to be a list with 1D np.ndarrays of equal length'

    # Check that the inputs have the correct dimension
    if not isinstance(data, list):
        raise ValueError(em_list_len)

    # Check that data contains only 1d numpy.arrays
    for i in range(len(data)):
        if not isinstance(data[i], np.ndarray):
            raise ValueError(em_list_len)
        if len(data[i].shape) != 1:
            raise ValueError(em_list_len)

    # Check that data contains only 1d numpy.arrays of the same length
    for i in range(len(data)):
        for j in range(len(data)):
            if len(data[i]) != len(data[j]):
                raise ValueError(em_list_len)

    if dimension is not None:
        if dimension not in ['1D', '2D', '3D']:
            raise ValueError('Input \'dimension\' must be 1D, 2D or 3D')
        # Check that the length of data is correct
        if len(data) != int(dimension[0]) and len(data) != 0:
            if dimension == '1D':
                raise ValueError('When \'dimension\' is 1D, input \'' + name + '\' must be [X]')
            elif dimension == '2D':
                raise ValueError('When \'dimension\' is 2D, input \'' + name + '\' must be [X, Y]')
            elif dimension == '3D':
                raise ValueError('When \'dimension\' is 3D, input \'' + name + '\' must be [X, Y, Z]')

    return


def check_bounds(bounds, dimension, dtype):
    """
    Function to check that the bounds are correct (x_min < x_max) and match the dimension,
    i.e. [x_min, x_max, y_min, y_max] for a 2D problem

    Parameters
    ----------

    bounds : list of int or float
        Bounds which define a hyper rectangle.

    dimension : str
        Dimensionality of the problem. Must be one of '1D', '2D', or '3D'.

    dtype : <class:int> or <class:float>
        Dtype which is inside the bounds list.

    """
    
    if bounds is None:
        return

    if dtype not in [float, int]:
        raise ValueError('Input \'dtype\' must be either \'float\' or \'int\'')

    em_list_type = 'Input \'bounds\' has to be a list of \'' + str(dtype)[8:-2] + '\''

    if not isinstance(bounds, list):
        raise ValueError(em_list_type)

    for value in bounds:
        if not isinstance(value, float) and not isinstance(value, int):
            raise ValueError(em_list_type)

    if len(bounds) != int(dimension[0])*2:
        raise ValueError('When \'dimension\' is ' + dimension + ', \'bounds\' must be a list with length {0}'.
                         format(int(dimension[0])*2))

    if dimension == '1D':
        error_msg_1D = 'Input \'bounds\' must be [x_min, x_max] with x_max>x_min'
        if bounds[1] <= bounds[0]:
            raise ValueError(error_msg_1D)

    elif dimension == '2D':
        error_msg_2D = 'Input \'bounds\' must be [x_min, x_max, y_min, y_max]' + \
                       ' with x_max>x_min and y_max>y_min'
        if bounds[1] <= bounds[0]:
            raise ValueError(error_msg_2D)
        if bounds[3] <= bounds[2]:
            raise ValueError(error_msg_2D)

    elif dimension == '3D':
        error_msg_3D = 'Input \'bounds\' must be [x_min, x_max, y_min, y_max, z_min, z_max]' + \
                       ' with x_max>x_min, y_max>y_min, and z_max>z_min'
        if bounds[1] <= bounds[0]:
            raise ValueError(error_msg_3D)
        if bounds[3] <= bounds[2]:
            raise ValueError(error_msg_3D)
        if bounds[5] <= bounds[4]:
            raise ValueError(error_msg_3D)

def check_number(param, name, dtype=float, threshold=0, check='geq'):
    """
    Function to check the input of a number and compare it to a threshold. If the check fails, a
    value error is raised.

    Since python cannot easily extract variable names, the API requires to call the function like this:
    check_number(foo, 'foo')
    In this way, the error is correctly raised with a reference to the wrong variable.

    Parameters
    ----------

    param : int or float
        Parameter whose value is checked.

    name : str
        Variable name in the code that calls this function.

    dtype : float or int
        Required dtype of 'param'.

    threshold : int or float
        Value against which 'param' is compared.

    check : str
        Which operation to perform. There are four options which are borrowed from latex syntax

        * 'geq' : >=
        * 'leq' : <=
        * 'g' : >
        * 'l' : <

    """

    # # Check that the dtype is correct
    # if dtype not in [float, int]:
    #     raise ValueError('Input \'dtype\' must be one of \'int\' or \'float\'')

    # Check that the threshold is correct
    if type(threshold) not in [float, int]:
        raise ValueError('Input \'threshold\' must be either int or float')

    # Check that the check is correct
    if check not in ['geq', 'leq', 'g', 'l']:
        raise ValueError('Input \'check\' must be one of \'geq\', \'leq\', \'g\' or \'l\'')

    # Check the instance
    if not isinstance(dtype, list):
        if not isinstance(param, dtype):
            raise ValueError('Input \'' + name + '\' must be ' + str(dtype)[8:-2])
    else:
        for typee in dtype:
            if isinstance(param, typee):
                flag = True
        if not flag:
            raise ValueError('Input \'' + name + '\' must be ' + str(dtype)[8:-2])

    # Check the value if the other checks all passed
    if check == 'geq':
        if not param >= threshold:
            raise ValueError('Input \'' + name + '\' must be >= ' + str(threshold))
    if check == 'leq':
        if not param <= threshold:
            raise ValueError('Input \'' + name + '\' must be <= ' + str(threshold))
    if check == 'g':
        if not param > threshold:
            raise ValueError('Input \'' + name + '\' must be > ' + str(threshold))
    if check == 'l':
        if not param < threshold:
            raise ValueError('Input \'' + name + '\' must be < ' + str(threshold))

    return