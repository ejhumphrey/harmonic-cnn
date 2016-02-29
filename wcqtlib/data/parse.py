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
import zipfile

logger = logging.getLogger(__name__)


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


def rwc_to_dataframe(base_dir):
    """Convert a base directory of RWC files to a pandas dataframe.
    """
    file_list = []
    index = 0
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
        dynamic_code = audio_file_name[7]

        file_list.append(
            dict(index=index,
                 audio_file=audio_file_path,
                 dataset="rwc",
                 # Convert this to actual instrument name?
                 instrument=instrument_code,
                 dynamic=dynamic_code))
        index += 1

    return pandas.DataFrame(file_list)


def uiowa_to_dataframe(base_dir):
    """Convert a base directory of UIowa files to a pandas dataframe.
    """
    file_list = []
    index = 0
    root_dir = os.path.join(base_dir, "theremin.music.uiowa.edu",
                            "sound files", "MIS")
    for item in os.scandir(root_dir):
        if item.is_dir():
            parent_cagetegory = item.name
            audio_files = glob.glob(os.path.join(item.path, "*/*.aiff"))
            for audio_file_path in audio_files:
                filename = os.path.splitext(
                    os.path.basename(audio_file_path))[0]
                parameters = [x.strip() for x in filename.split('.')]
                instrument = parameters.pop(0)
                notevalue = parameters.pop(-1) if len(parameters) else None
                # You have to do this to get rid of this element if it's there
                articulation = parameters.pop(0) if len(parameters) > 1 \
                    else None
                dynamic = parameters.pop(0) if len(parameters) else None
                # There might be more now but we don't really care.

                file_list.append(
                    dict(index=index,
                         audio_file=audio_file_path,
                         dataset="uiowa",
                         instrument=instrument,
                         dynamic=dynamic,
                         note=notevalue,
                         parent=parent_cagetegory)
                    )
                index += 1

    return pandas.DataFrame(file_list)


def philharmonia_to_dataframe(base_dir):
    """Convert a base directory of Philharmonia files to a pandas dataframe.
    """
    root_dir = os.path.join(base_dir, "www.philharmonia.co.uk",
                            "assets", "audio", "samples")

    # These files come in zips. Extract them as necessary.
    zip_files = glob.glob(os.path.join(root_dir, "*/*.zip"))
    unzip_files(zip_files)

    file_list = []
    index = 0
    for audio_file_path in glob.glob(os.path.join(root_dir, "*/*/*.mp3")):
        audio_file_name = os.path.splitext(
            os.path.basename(audio_file_path))[0]
        instrument, note, duration, dynamic, _ = audio_file_name.split('_')

        file_list.append(
            dict(index=index,
                 audio_file=audio_file_path,
                 dataset="philharmonia",
                 # Convert this to actual instrument name?
                 instrument=instrument,
                 note=note,
                 dynamic=dynamic))
        index += 1

    return pandas.DataFrame(file_list)


def load_dataframes(data_dir):
    """Load all the datasets into a single dataframe.

    Parameters
    ----------
    data_dir : str

    Returns
    -------
    dataframe : pandas.DataFrame()
        Dataframe containing pointers to all the files.
    """
    rwc_df = rwc_to_dataframe(os.path.join(data_dir, "RWC Instruments"))
    uiowa_df = uiowa_to_dataframe(os.path.join(data_dir, "uiowa"))
    philharmonia_df = philharmonia_to_dataframe(
        os.path.join(data_dir, "philharmonia"))

    return pandas.concat([rwc_df, uiowa_df, philharmonia_df])


if __name__ == "__main__":
    dfs = load_dataframes(os.path.expanduser("~/data"))
    import pdb; pdb.set_trace()
