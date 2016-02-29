import numpy as np
import os
import zipfile


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


def unzip_files(file_list):
    """Given a list of file paths, unzip them in place.

    Attempts to skip it if the extracted folder exists.

    Parameters
    ----------
    file_list : list of str

    Returns
    -------
    List of created output folders.
    """
    result_list = []
    for zip_path in file_list:
        working_dir = os.path.dirname(zip_path)
        zip_name = os.path.splitext(os.path.basename(zip_path))[0]
        new_folder_path = os.path.join(working_dir, zip_name)
        if not os.path.exists(new_folder_path):
            with zipfile.ZipFile(zip_path, 'r') as myzip:
                # Create a directory of the same name as the zip.
                os.makedirs(new_folder_path)
                myzip.extractall(path=new_folder_path)
                result_list.append(new_folder_path)

    return result_list
