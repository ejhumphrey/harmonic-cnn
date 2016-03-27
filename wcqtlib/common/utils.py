import datetime
import numpy as np
import os
import re
import zipfile

import colorama

COLOR_MAP = {
    "yellow": colorama.Fore.YELLOW,
    "red": colorama.Fore.RED,
    "green": colorama.Fore.GREEN,
    "blue": colorama.Fore.BLUE,
    "magenta": colorama.Fore.MAGENTA,
    "cyan": colorama.Fore.CYAN,
    "white": colorama.Fore.WHITE
}


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


def slice_ndarray(x_in, idx, length, axis=0):
    """Extract a slice from an ndarray, along a given axis.
    Parameters
    ----------
    x_in : np.ndarray
        Array to slice.
    idx : int, 0 < n < x_in.shape[axis] - length
        Index to start the resulting tile.
    length : int
        Total length for the output tile.
    axis : int, default=0
        Axis over which to slice the ndarray.
    Returns
    -------
    z_out : np.ndarray
        The sliced subtensor.
    """
    slice_idxs = [slice(None, ) for n in range(x_in.ndim)]
    slice_idxs[axis] = slice(idx, idx + length)
    return x_in[tuple(slice_idxs)]


def colored(text, color="yellow"):
    """Color terminal text

    Parameters
    ----------
    text : str
        Text to color.
    color : string
        Name of color to print

    Returns
    ------
    colored_text : str
        String of colored text.
    """

    return "{0}{1}{2}".format(
            COLOR_MAP[color], text,
            colorama.Style.RESET_ALL)


def iter_from_params_filepath(params_filepath):
    """Get the model iteration from the params filepath.

    There are two cases; the iteration number case, and 'final.npz'
    For example '/foo/myexperiment/params/params0500.npz' => '0500'

    Parameters
    ----------
    params_filepath : str

    Returns
    -------
    iter_name : str
    """
    basename = os.path.basename(params_filepath)
    return re.search('\d+|final', basename).group(0)


class TimerHolder(object):
    def __init__(self):
        self.timers = {}

    def start(self, tuple_or_list):
        """
        Note: tuples can be keys.
        Parameters
        ----------
        tuple_or_list : str or list of str
        """
        if isinstance(tuple_or_list, (str, tuple)):
            self.timers[tuple_or_list] = [datetime.datetime.now(), None]
        elif isinstance(tuple_or_list, list):
            for key in tuple_or_list:
                self.timers[key] = [datetime.datetime.now(), None]

    def end(self, tuple_or_list):
        """
        Parameters
        ----------
        tuple_or_list : str or list of str
        """
        if isinstance(tuple_or_list, (str, tuple)):
            self.timers[tuple_or_list][1] = datetime.datetime.now()
            return self.timers[tuple_or_list][1] - \
                self.timers[tuple_or_list][0]
        elif isinstance(tuple_or_list, list):
            results = []
            for key in tuple_or_list:
                self.timers[key][1] = datetime.datetime.now()
                results += [self.timers[key][1] - self.timers[key][0]]
            return results

    def get(self, key):
        if key in self.timers:
            if self.timers[key][1]:
                return self.timers[key][1] - self.timers[key][0]
            else:
                return self.timers[key][0]
        else:
            return None

    def get_start(self, key):
        return self.timers.get(key, None)[0]

    def get_end(self, key):
        return self.timers.get(key, None)[1]

    def mean(self, key_root, start_ind, end_ind):
        keys = [(key_root, x) for x in range(max(start_ind, 0), end_ind)]
        values = [self.get(k) for k in keys if k in self.timers]
        return np.mean(values)
