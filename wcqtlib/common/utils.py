import numpy as np
import os


def create_directory(dname):
    """Create the output directory recursively if it doesn't already exist.

    Parameters
    ----------
    dname : str
        Directory to create.

    Returns
    -------
    success : bool
        True if the requested directory now exists.
    """
    if not os.path.exists(dname):
        os.makedirs(dname)
    return os.path.exists(dname)


def filebase(fpath):
    """Return the file's basename without an extension, e.g. 'x/y.z' -> 'y'."""
    return os.path.splitext(os.path.basename(fpath))[0]


def fold_array(x_in, length, stride):
    """Fold a 2D-matrix into a 3D tensor by wrapping the last dimension.

    Parameters
    ----------
    x_in : np.ndarray, ndim=2
        Array to fold.

    length : int
        Length of each window.

    stride : int
        Stride to advance overlapping windows.

    Returns
    -------
    x_out : np.ndarray, ndim=3
        Layered output, with channels as the first dimension.
    """
    num_tiles = int((x_in.shape[1] - (length-stride)) / float(stride))
    return np.array([x_in[:, n*stride:n*stride + length]
                     for n in range(num_tiles)])


def map_io(input_file, output_directory):
    """Map the basename of an input file path to an output directory.

    Parameters
    ----------
    input_file : str
        Input file path to parse.

    output_directory : str
        Directory the output will be mapped into.

    Returns
    -------
    output_file : str
        Resulting output file mapped to the given directory.
    """
    create_directory(output_directory)
    return os.path.join(output_directory,
                        "{}.npz".format(filebase(input_file)))
