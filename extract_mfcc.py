
import json
import os
import math
import librosa

dataset_path = '003.wav'

def save_mfcc(dataset_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):

    SAMPLE_RATE = 22050
    TRACK_DURATION = 2 # measured in seconds
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
    # dictionary to store mapping, labels, and MFCCs
    mfcc_lst = []
    _mapping = {
    1:"Bis'mi",
    2:"Al-lahi",
    3:"Al-rahmaani",
    4:"Al-raheemi",
    5:"Alhamdu",
    6:"lillaahi",
    7:"Rabbil",
    8:"aalameen",
    9:"Ar-Rahmaan",
    10:"Ar-Raheem",
    11:"Maaliki",
    12:"Yumid",
    13:"Diin",
    14:"Iyyaka",
    15:"Na'abudu",
    16:"Iyyaka",
    17:"Nasta'een",
    18:"Ihdinas",
    19:"Siraatal",
    20:"Mustaqeem",
    21:"Siraatal",
    22:"Ladheena",
    23:"An'amta",
    24:"Alaihim",
    25:"Ghayril",
    26:"Maghdubi",
    27:"Alaihim",
    28:"Wala al-dalina"}

    # we divide the track into 5 segments
    # to calculate sample per segment - we need to know the number of samples per track (which is the sample rate * the track duration )
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    
    signal, sample_rate = librosa.load(dataset_path, sr=SAMPLE_RATE)

    # process all segments of audio file
    for d in range(num_segments):

        # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # extract mfcc - for each segment of signal
        mfcc = librosa.feature.mfcc(y=signal[start:finish],sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T

        # store only mfcc feature with expected number of vectors
        if len(mfcc) == num_mfcc_vectors_per_segment:
            mfcc_lst.append(mfcc.tolist())

    # print("MFCC LIST: ", mfcc_lst[0])
    return mfcc_lst[0]

if __name__ == "__main__":
    x = save_mfcc(dataset_path)
    print("Done",x)