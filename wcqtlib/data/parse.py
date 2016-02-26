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

import pandas


def rwc_to_dataframe(base_dir):
    """Convert a base directory of RWC files to a pandas dataframe.
    """
    return pandas.DataFrame()


def uiowa_to_dataframe(base_dir):
    """Convert a base directory of UIowa files to a pandas dataframe.
    """
    return pandas.DataFrame()


def philharmonia_to_dataframe(base_dir):
    """Convert a base directory of Philharmonia files to a pandas dataframe.
    """
    return pandas.DataFrame()
