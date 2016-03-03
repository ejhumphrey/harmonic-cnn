"""Utilities for manipulating audio files."""

import claudio
import claudio.sox
import librosa
import logging
import numpy as np
import os

import wcqtlib.common.utils as utils

logger = logging.getLogger(__name__)


def get_onsets(audio, sr, **kwargs):
    # reshape the damn audio so that librosa likes it.
    reshaped_audio = audio.reshape((audio.shape[0],))
    onset_frames = librosa.onset.onset_detect(
        y=reshaped_audio, sr=sr, **kwargs)
    onset_samples = librosa.frames_to_samples(onset_frames)
    return onset_samples


def split_and_standardize_examples(input_audio_path,
                                   output_dir,
                                   first_onset_start=.05,
                                   final_duration=None):
    """Takes an audio file, and splits it up into multiple
    audio files, using silence as the delimiter.

    Once they are split, the onset, as detected by librosa,
    is then placed at the location specified by first_onset_start.

    Parameters
    ----------
    input_audio_path : str
        Full path to the audio file to use.

    output_dir : str
        Full path to the folder where you want to place the
        result files. Will be created if it does not exist.

    first_onset_start : float
        Value in seconds where the first onset will
        be set, for a sort of normalization of the audio.

    final_duration : float or None
        If not None, trims the final audio file to final_duration
        seconds.

    Returns
    -------
    output_files : List of audio files created in this process.
    """
    original_name = os.path.basename(input_audio_path)
    filebase = utils.filebase(original_name)
    new_output_path = os.path.join(output_dir, original_name)

    ready_files = []

    # Split the audio files using claudio.sox
    if claudio.sox.split_along_silence(
                original_name, new_output_path):

        # Sox generates files of the form:
        # original_name001.xxx
        # original_name001.xxx
        process_files = [x for x in os.listdir(output_dir) if filebase in x]

        # For each file generated
        for file_name in process_files:
            audio_path = os.path.join(output_dir, file_name)
            success = standardize_one(audio_path,
                                      first_onset_start=first_onset_start,
                                      final_duration=final_duration)
            if success:
                ready_files.append(audio_path)

    return ready_files


def standardize_one(input_audio_path,
                    first_onset_start=.05,
                    center_of_mass_alignment=False,
                    final_duration=None):
    """Takes a single audio file, and standardizes it based
    on the parameters provided.

    Heads up! Modifies the file in place...

    Parameters
    ----------
    input_audio_path : str
        Full path to the audio file to work with.

    first_onset_start : float or None
        If not None, uses librosa's onset detection to find
        the first onset in the file, and then pads the beginning
        of the file with zeros such that the first onset
        ocurrs at first_onset_start seconds.

        If no onsets are discovered, assumes this is an
        empty file, and returns False.

    center_of_mass_alignment : boolean
        If True, aligns the center of mass of the file to
        be at the center of the sample.

    final_duration : float or None
        If not None, trims the final audio file to final_duration
        seconds.

    Returns
    -------
    True if all processes passed. False otherwise.
    """
    # Load the audio file
    audio, sr = claudio.read(input_audio_path, channels=1)

    # Find the onsets using librosa
    onset_samples = get_onsets(audio, sr)

    first_onset_start_samples = first_onset_start * sr
    actual_first_onset = onset_samples[0]
    # Pad the beginning with up to onset_start ms of silence
    onset_difference = first_onset_start_samples - actual_first_onset

    # Correct the difference by adding or removing samples from the beginning.
    if onset_difference > 0:
        # In this case, we need to append this many zeros to the start
        audio = np.concatenate([
            np.zeros([onset_difference, audio.shape[-1]]),
            audio])
    elif onset_difference < 0:
        audio = audio[np.abs(onset_difference):]

    # save the file back out again.
    claudio.write(input_audio_path, audio, samplerate=sr)

    return True
