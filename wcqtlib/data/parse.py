"""Utilities for wranling the collections, once localized.

The three datasets are assumed to come down off the wire in the following
representations.

# RWC
If obtained from AIST, the data will live on 12 CDs. Here, they have been
backed into a folder hierarchy like the following:

    {base_dir}/
        RWC_I_01/
            {num}/
                {num}{instrument}{style}{dynamic}.{fext}
        RWC_I_02
            ...
        RWC_I_12

Where...
 * base_dir : The path where the data are collated
 * num : A three-digit number contained in the folder
 * instrument : A two-character instrument code
 * style : A two-character style code
 * dynamic : A one-character loudness value

Here, these composite filenames are *ALL* comprised of 8 characters in length.

# Dataset Index
The columns could look like the following:
 * index / id : a unique identifier for each row
 * audio_file :
 * feature_file :
 * dataset :
 * ...?
"""

import glob
import logging
import os
import pandas

logger = logging.getLogger(__name__)


def rwc_to_dataframe(base_dir):
    """Convert a base directory of RWC files to a pandas dataframe.
    """
    file_list = []
    index = 0
    # Iterate over the 12 "CDs".
    for audio_file_path in glob.glob(os.path.join(base_dir, "*/*/*.flac")):
        audio_file_name = os.path.basename(audio_file_path)
        if len(audio_file_name) != 13:
            logger.warning("Audio file '{}' does not have the "
                           "expected file name format: skipping."
                           .format(audio_file_name))
            continue
        instrument_code = audio_file_name[3:5]
        # ...Do we care?
        # style_code = audio_file_name[5:7]
        # dynamic_code = audio_file_name[7]

        file_list.append(
            dict(index=index,
                 audio_file=audio_file_path,
                 dataset="rwc",
                 # Convert this to actual instrument name?
                 instrument=instrument_code))

    return pandas.DataFrame(file_list)


def uiowa_to_dataframe(base_dir):
    """Convert a base directory of UIowa files to a pandas dataframe.
    """
    return pandas.DataFrame()


def philharmonia_to_dataframe(base_dir):
    """Convert a base directory of Philharmonia files to a pandas dataframe.
    """
    return pandas.DataFrame()
