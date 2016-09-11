import numpy as np
import os


def convert_cifar(input_file, output_file=None):
    """Convert CIFAR data to an NPZ archive.

    Note: Only works with python2

    Parameters
    ----------
    input_file : str
        Path to a CIFAR pickle file.

    output_file : str, default=None
        Optional filepath for writing output. If not provided,
        will append '.npz' to the input filename.

    Returns
    -------
    success : bool
        True if the output file was written successfully.
    """
    with open(input_file, 'r') as fp:
        import cPickle
        data = cPickle.load(fp)
    data['data'] = data['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    if not output_file:
        output_file = "{}.npz".format(input_file)
    np.savez(output_file, **data)
    return os.path.exists(output_file)
