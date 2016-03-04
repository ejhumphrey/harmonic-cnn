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

import argparse
import glob
import hashlib
import json
import logging
import os
import pandas
import re

import wcqtlib.common.utils as utils

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    "data")
RWC_INSTRUMENT_MAP_PATH = os.path.join(DATA_DIR, "rwc_instrument_map.json")
CLASS_MAP = os.path.join(DATA_DIR, "class_map.json")
with open(RWC_INSTRUMENT_MAP_PATH, 'r') as fh:
    RWC_INSTRUMENT_MAP = json.load(fh)


def rwc_instrument_code_to_name(rwc_instrument_code):
    """Use the rwc_instrument_map.json to convert an rwc_instrument_code
    to it's instrument name.

    Parameters
    ----------
    rwc_instrument_code : str
        Two character instrument code

    Returns
    -------
    instrument_name : str
        Full instrument name, if it exists, else the code.
    """
    instrument_name = RWC_INSTRUMENT_MAP.get(
        rwc_instrument_code, rwc_instrument_code)
    return instrument_name if instrument_name else rwc_instrument_code


def parse_rwc_path(rwc_path):
    """Takes an rwc path, and returns the extracted codes from the
    filename.

    Parameters
    ----------
    rwc_path : str
        Full path or basename. If full path, gets the basename.

    Returns
    -------
    instrument_name : str
    style_code : str
    dynamic_code : str
    """
    filebase = utils.filebase(rwc_path)
    instrument_code = filebase[3:5]
    # Get the instrument name from the json file.
    instrument_name = rwc_instrument_code_to_name(instrument_code)
    style_code = filebase[5:7]
    dynamic_code = filebase[7]
    return instrument_name, style_code, dynamic_code


def generate_id(dataset, audio_file_path):
    """Create a unique identifier for this entry.

    Returns
    -------
    id : str
        dataset[0] + md5(audio_file_path)[:8]
    """
    dataset_code = dataset[0]
    audio_file_hash = hashlib.md5(
        utils.filebase(audio_file_path)
        .encode('utf-8')).hexdigest()
    return "{0}{1}".format(dataset_code, audio_file_hash[:8])


def rwc_to_dataframe(base_dir, dataset="rwc"):
    """Convert a base directory of RWC files to a pandas dataframe.

    Parameters
    ----------
    base_dir : str
        Full path to the base RWC directory.

    dataset : str
        Datset string to use in this df.

    Returns
    -------
    pandas.DataFrame
        Indexed by:
            id : [dataset identifier] + [8 char md5 of filename]
        With the following columns:
            audio_file : full path to original audio file.
            dataset : dataset it is from
            instrument : instrument label.
            dynamic : dynamic tag
    """
    indexes = []
    records = []
    for audio_file_path in glob.glob(os.path.join(base_dir, "*/*/*.flac")):
        instrument_name, style_code, dynamic_code = \
            parse_rwc_path(audio_file_path)

        indexes.append(generate_id(dataset, audio_file_path))
        records.append(
            dict(audio_file=audio_file_path,
                 dataset=dataset,
                 instrument=instrument_name,
                 dynamic=dynamic_code))

    return pandas.DataFrame(records, index=indexes)


def parse_uiowa_path(uiowa_path):
    filename = utils.filebase(uiowa_path)
    parameters = [x.strip() for x in filename.split('.')]
    instrument = parameters.pop(0)
    # This regex matches note names with a preceeding and following '.'
    note_match = re.search(r"(?<=\.)[A-Fb#0-6]*(?<!\.)", filename)
    notevalue = filename[note_match.start():note_match.end()] \
                if note_match else None
    # This regex matches dynamic chars with a preceeding and following '.'
    dynamic_match = re.search(r"(?<=\.)[f|p|m]*(?<!\.)", filename)
    dynamic = filename[dynamic_match.start():dynamic_match.end()] \
              if dynamic_match else None
    return instrument, dynamic, notevalue


def uiowa_to_dataframe(base_dir, dataset="uiowa"):
    """Convert a base directory of UIowa files to a pandas dataframe.

    Parameters
    ----------
    base_dir : str
        Full path to the base RWC directory.

    dataset : str
        Datset string to use in this df.

    Returns
    -------
    pandas.DataFrame
        With the following columns:
            id
            audio_file
            dataset
            instrument
            dynamic
            note
            parent : instrument category.
    """
    indexes = []
    records = []
    root_dir = os.path.join(base_dir, "theremin.music.uiowa.edu",
                            "sound files", "MIS")
    for item in os.scandir(root_dir):
        if item.is_dir():
            parent_cagetegory = item.name
            audio_files = glob.glob(os.path.join(item.path, "*/*.aif*"))
            for audio_file_path in audio_files:
                instrument, dynamic, notevalue = \
                    parse_uiowa_path(audio_file_path)

                indexes.append(generate_id(dataset, audio_file_path))
                records.append(
                    dict(audio_file=audio_file_path,
                         dataset=dataset,
                         instrument=instrument,
                         dynamic=dynamic,
                         note=notevalue,
                         parent=parent_cagetegory))

    return pandas.DataFrame(records, index=indexes)


def parse_phil_path(phil_path):
    """Convert phil path to codes/parameters.

    Parameters
    ----------
    phil_path : full path.

    Returns
    -------
    tuple of parameters.
    """
    audio_file_name = utils.filebase(phil_path)
    instrument, note, duration, dynamic, articulation = \
        audio_file_name.split('_')
    return instrument, note, duration, dynamic, articulation


def philharmonia_to_dataframe(base_dir, dataset="philharmonia"):
    """Convert a base directory of Philharmonia files to a pandas dataframe.

    Parameters
    ----------
    base_dir : str
        Full path to the base RWC directory.

    dataset : str
        Datset string to use in this df.

    Returns
    -------
    pandas.DataFrame
        With the following columns:
            id
            audio_file
            dataset
            instrument
            note
            dynamic
    """
    root_dir = os.path.join(base_dir, "www.philharmonia.co.uk",
                            "assets", "audio", "samples")

    # These files come in zips. Extract them as necessary.
    zip_files = glob.glob(os.path.join(root_dir, "*/*.zip"))
    utils.unzip_files(zip_files)

    indexes = []
    records = []
    for audio_file_path in glob.glob(os.path.join(root_dir, "*/*/*.mp3")):
        instrument, note, duration, dynamic, _ = \
            parse_phil_path(audio_file_path)

        indexes.append(generate_id(dataset, audio_file_path))
        records.append(
            dict(audio_file=audio_file_path,
                 dataset=dataset,
                 instrument=instrument,
                 note=note,
                 dynamic=dynamic))

    return pandas.DataFrame(records, index=indexes)


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


class InstrumentClassMap(object):
    """Class for handling map between class names and the
    names they possibly could be from the datasets."""

    def __init__(self, file_path):
        """
        Parameters
        ----------
        file_path : str
        """
        with open(file_path, 'r') as fh:
            self.data = json.load(fh)

        # Create the reverse map so we can efficiently do the
        # reverse lookup
        self.reverse_map = {}
        for classname in self.data:
            for item in self.data[classname]:
                self.reverse_map[item] = classname

    @classmethod
    def read(cls, file_path=CLASS_MAP):
        """Read it in with the default path."""
        return cls(file_path)

    @property
    def allnames(self):
        """Return a complete list of all class names for searching the 
        dataframe."""
        return sorted(self.reverse_map.keys())

    @property
    def classnames(self):
        return sorted(self.data.keys())

    def __getitem__(self, searchkey):
        """Get the actual class name. (Actually the reverse map)."""
        return self.reverse_map[searchkey]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Parse raw data into dataframe')
    parser.add_argument("--data_root", default=os.path.expanduser("~/data/"))
    parser.add_argument("-o", "--output_path",
                        default="ismir2016-wcqt-data/datasets.json")
    args = parser.parse_args()

    # Load the datasets dataframe
    print("Loading dataset...")
    dfs = load_dataframes(args.data_root)
    print("Datasets contain {} audio files.".format(len(dfs)))
    # Save it to a json file
    output_path = os.path.join(args.data_root, args.output_path)
    print("Saving to", output_path)
    dfs.to_json(output_path)
