import json
import os
import math
import librosa

DATASET_PATH = "./001/008"
JSON_PATH = "data_a8.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 2 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.
        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

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

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            label = dirpath[-2:]
            label = int(label)
            print("label",label)
            data["mapping"].append(_mapping[label])

            # process all audio files in genre sub-dir
            for f in filenames:

		        # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # process all segments of audio file
                for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc - for each segment of signal
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        # data["labels"].append(i-1)
                        data["labels"].append(label-1)
                        print("{}, segment:{}".format(file_path, d+1))

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
        
# if __name__ == "__main__":
    # save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)

''' To combine all the json files of each ayah'''
# # combine all json file into one
# import json
# merge_data= {
#     "mapping": [],
#     "labels": [],
#     "mfcc": [],
# }
# for filename in ["data_a1.json", "data_a2.json", "data_a3.json", "data_a4.json", "data_a5.json", "data_a6.json", "data_a7.json", "data_a8.json"]:
#     with open(filename) as file:
#         data = json.load(file)
#         # Merge the data into the dictionary
#         merge_data["mapping"] += data["mapping"]
#         merge_data["labels"] += data["labels"]
#         merge_data["mfcc"] += data["mfcc"]

# # Write the merged data to a new file
# with open("data_surah_1.json", "w") as file:
#     json.dump(merge_data, file, indent=4)