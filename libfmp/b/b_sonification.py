"""
Module: libfmp.b.b_sonification
Author: Meinard Mueller, Tim Zunner
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP).
"""

import numpy as np


def list_to_chromagram(note_list, num_frames, frame_rate):
    """Create a chromagram matrix from a list of note events

    Parameters
    ----------
    note_list : List
        A list of note events (e.g. gathered from a CSV file by libfmp.c1.pianoroll.csv_to_list())

    num_frames : int
        Desired number of frames for the matrix

    frame_rate : float
        Frame rate for C (in Hz)

    Returns
    -------
    C : NumPy Array
        Chromagram matrix
    """
    C = np.zeros((12, num_frames))
    for l in note_list:
        start_frame = max(0, int(l[0] * frame_rate))
        end_frame = min(num_frames, int((l[0] + l[1]) * frame_rate) + 1)
        C[int(l[2] % 12), start_frame:end_frame] = 1
    return C


def generate_shepard_tone(chromaNum, Fs, N, weight=1, Fc=440, sigma=15, phase=0):
    """
    inputs:
        chromaNum: 1=C,...
        Fs: sampling frequency
        N: desired length (in samples)
        weight: scaling factor [0:1]
        Fc: frequency for A4
        sigma: parameter for envelope of Shepard tone
        fading: fading at the beginning and end of the tone (in ms)
    output:
        shepard tone
    """
    tone = np.zeros(N)
    # Envelope function for Shepard tones
    p = 24 + chromaNum
    if(p > 32):
        p = p - 12
    while p < 108:
        scale_factor = 1 / (np.sqrt(2 * np.pi) * sigma)
        A = scale_factor * np.exp(-(p - 60) ** 2 / (2 * sigma ** 2))
        f_axis = np.arange(N) / Fs
        sine = np.sin(2 * np.pi * np.power(2, ((p - 69) / 12)) * Fc * (f_axis + phase))
        tmp = weight * A * sine
        tone = tone + tmp
        p = p + 12
    return tone


def sonify_chromagram(chroma_data, N, frame_rate, Fs, fading_msec=5):
    """Sonify the chroma features from a chromagram

    Parameters
    ----------
    chroma_data : NumPy Array
        A chromagram (e.g. gathered from a list of note events by list_to_chromagram())

    N : int
        Length of the sonification (in samples)

    frame_rate : float
        Frame rate for P (in Hz)

    Fs : float
        Sampling frequency (in Hz)

    fading_msec : float
        The length of the fade in and fade out for sonified tones (in msec)

    Returns
    -------
    chroma_son : NumPy Array
        Sonification of the chromagram
    """

    chroma_son = np.zeros((N,))
    fade_sample = int(fading_msec / 1000 * Fs)

    for i in range(12):
        if np.sum(np.abs(chroma_data[i, :])) > 0:
            shepard_tone = generate_shepard_tone(i, Fs, N)
            weights = np.zeros((N,))
            for j in range(chroma_data.shape[1]):
                if np.abs(chroma_data[i, j]) > 0:
                    start = min(N, max(0, int((j - 0.5) * Fs / frame_rate)))
                    end = min(N, int((j + 0.5) * Fs / frame_rate))
                    fade_start = min(N, max(0, start+fade_sample))
                    fade_end = min(N, end+fade_sample)

                    weights[fade_start:end] += chroma_data[i, j]
                    weights[start:fade_start] += np.linspace(0, chroma_data[i, j], fade_start-start)
                    weights[end:fade_end] += np.linspace(chroma_data[i, j], 0, fade_end-end)

            chroma_son += shepard_tone * weights

    chroma_son = chroma_son / np.max(np.abs(chroma_son))

    return chroma_son


def sonify_chromagram_with_signal(chroma_data, x, frame_rate, Fs, fading_msec=5, stereo=True):
    """Sonify the chroma features from a chromagram together with a corresponding signal

    Parameters
    ----------
    chroma_data : NumPy Array
        A chromagram (e.g. gathered from a list of note events by list_to_chromagram()

    x : NumPy Array
        Original signal

    frame_rate : float
        Frame rate for P (in Hz)

    Fs : float
        Sampling frequency (in Hz)

    fading_msec : float
        The length of the fade in and fade out for sonified tones (in msec)

    stereo : bool
        Decision between stereo and mono sonification

    Returns
    -------
    chroma_son : NumPy Array
        Sonification of the chromagram

    out : NumPy Array
        Sonification combined with the original signal
    """

    N = x.size

    chroma_son = sonify_chromagram(chroma_data, N, frame_rate, Fs, fading_msec=fading_msec)
    chroma_scaled = chroma_son * np.sqrt(np.mean(x**2)) / np.sqrt(np.mean(chroma_son**2))

    if stereo:
        out = np.vstack((x, chroma_scaled))
    else:
        out = x + chroma_scaled
    out = out / np.amax(np.abs(out))

    return chroma_son, out


def list_to_pitch_activations(note_list, num_frames, frame_rate):
    """Create a pitch activation matrix from a list of note events

    Parameters
    ----------
    note_list : List
        A list of note events (e.g. gathered from a CSV file by libfmp.c1.pianoroll.csv_to_list())

    num_frames : int
        Desired number of frames for the matrix

    frame_rate : float
        Frame rate for P (in Hz)

    Returns
    -------
    P : NumPy Array
        Pitch activation matrix
        First axis: Indexed by [0:127], encoding MIDI pitches [1:128]
    F_coef_MIDI: MIDI pitch axis
    """

    P = np.zeros((128, num_frames))
    F_coef_MIDI = np.arange(128) + 1
    for l in note_list:
        start_frame = max(0, int(l[0] * frame_rate))
        end_frame = min(num_frames, int((l[0] + l[1]) * frame_rate) + 1)
        P[int(l[2]-1), start_frame:end_frame] = 1
    return P, F_coef_MIDI


def sonify_pitch_activations(P, N, frame_rate, Fs, min_pitch=1, Fc=440, harmonics_weights=[1], fading_msec=5):
    """Sonify the pitches from a pitch activation matrix

    Parameters
    ----------
    P : NumPy Array
        A pitch activation matrix (e.g. gathered from a list of note events by list_to_pitch_activations())
        First axis: Indexed by [0:127], encoding MIDI pitches [1:128]

    N : int
        Length of the sonification (in samples)

    frame_rate : float
        Frame rate for P (in Hz)

    Fs : float
        Sampling frequency (in Hz)

    min_pitch : int
        Lowest MIDI pitch in P

    Fc : float
        Tuning frequency (in Hz)

    harmonics_weights : list
        A list of weights for the harmonics of the tones to be sonified

    fading_msec : float
        The length of the fade in and fade out for sonified tones (in msec)

    Returns
    -------
    pitch_son : NumPy Array
        Sonification of the pitch activation matrix
    """

    fade_sample = int(fading_msec / 1000 * Fs)
    pitch_son = np.zeros((N,))

    for p in range(P.shape[0]):
        if np.sum(np.abs(P[p, :])) > 0:
            pitch = min_pitch + p
            freq = (2 ** ((pitch - 69) / 12)) * Fc
            sin_tone = np.zeros((N,))
            for i in range(len(harmonics_weights)):
                sin_tone += harmonics_weights[i] * np.sin(2 * np.pi * (i+1) * freq * np.arange(N) / Fs)

            weights = np.zeros((N,))
            for n in range(P.shape[1]):
                if np.abs(P[p, n]) > 0:
                    start = min(N, max(0, int((n - 0.5) * Fs / frame_rate)))
                    end = min(N, int((n + 0.5) * Fs / frame_rate))
                    fade_start = min(N, start+fade_sample)
                    fade_end = min(N, end+fade_sample)

                    weights[fade_start:end] += P[p, n]
                    weights[start:fade_start] += np.linspace(0, P[p, n], fade_start-start)
                    weights[end:fade_end] += np.linspace(P[p, n], 0, fade_end-end)

            pitch_son += weights * sin_tone

    pitch_son = pitch_son / np.max(np.abs(pitch_son))
    return pitch_son


def sonify_pitch_activations_with_signal(P, x, frame_rate, Fs, min_pitch=1, Fc=440, harmonics_weights=[1],
                                         fading_msec=5, stereo=True):
    """Sonify the pitches from a pitch activation matrix together with a corresponding signal

    Parameters
    ----------
    P : NumPy Array
        A pitch activation matrix (e.g. gathered from a list of note events by list_to_pitch_activations())

    x : NumPy Array
        Original signal

    frame_rate : float
        Frame rate for P (in Hz)

    Fs : float
        Sampling frequency (in Hz)

    min_pitch : int
        Lowest MIDI pitch in P

    Fc : float
        Tuning frequency (in Hz)

    harmonics_weights : list
        A list of weights for the harmonics of the tones to be sonified

    fading_msec : float
        The length of the fade in and fade out for sonified tones (in msec)

    stereo : bool
        Decision between stereo and mono sonification

    Returns
    -------
    pitch_son : NumPy Array
        Sonification of the pitch activation matrix

    out : NumPy Array
        Sonification combined with the original signal
    """

    N = x.size

    pitch_son = sonify_pitch_activations(P, N, frame_rate, Fs, min_pitch=min_pitch, Fc=Fc,
                                         harmonics_weights=harmonics_weights, fading_msec=fading_msec)
    pitch_scaled = pitch_son * np.sqrt(np.mean(x**2)) / np.sqrt(np.mean(pitch_son**2))

    if stereo:
        out = np.vstack((x, pitch_scaled))
    else:
        out = x + pitch_scaled

    return pitch_son, out
