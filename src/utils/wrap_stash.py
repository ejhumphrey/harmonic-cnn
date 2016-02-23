"""Wrap the CQT arrays of a biggie stash.

Each entity is expected to have at least a cqt field, shaped as
    [channel, time, pitch]

Afterward, this will look like:
    [channel, octave, time, pitch]

This ordering is determined primarily for convenience when convolving in
theano.

"""

import argparse
import biggie
import numpy as np
import os
import time

import common.utils as utils

STATUS = dict(total=-1, done=-1)


def wrap_cqt_for_key(stash_in, key, length, stride, stash_out):
    """Apply CQT wrapping to a single entity given an I/O stash pair.

    Parameters
    ----------
    stash_in : biggie.Stash
        Input stash to pull from.

    key : str
        Key in the input stash to wrap.

    length : int
        Length of the window per wrap.

    stride : int
        Number of bins to hop between wraps.

    stash_out : biggie.Stash
        Stash for collecting outputs.
    """
    entity = stash_in.get(key)
    entity.cqt = np.array([utils.fold_array(channel, length, stride)
                           for channel in entity.cqt])
    stash_out.add(key, entity)
    STATUS['done'] += 1
    print("[{now}] {done:12d} / {total:12d}: {key}"
          "".format(now=time.asctime(), key=key, **STATUS))


def main(args):
    in_stash = biggie.Stash(args.data_file)
    utils.create_directory(os.path.dirname(args.output_file))
    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    out_stash = biggie.Stash(args.output_file)
    STATUS['total'] = len(in_stash)
    for idx, key in enumerate(in_stash.keys()):
        new_entity = wrap_cqt_for_key(in_stash.get(key), args.length,
                                      args.stride)
        out_stash.add(key, new_entity)

    out_stash.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Inputs
    parser.add_argument("data_file",
                        metavar="data_file", type=str,
                        help="Path to an optimus file for validation.")
    parser.add_argument("length",
                        metavar="length", type=int,
                        help="Number of bins per CQT slice.")
    parser.add_argument("stride",
                        metavar="stride", type=int,
                        help="Number of bins between slices, i.e. an octave.")
    # Outputs
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Path for the transformed output.")
    main(parser.parse_args())
