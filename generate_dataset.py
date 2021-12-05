import os
import json
import python_speech_features as psf
import scipy.io.wavfile as wav


DATASET_PATH = r"Yes_no_dataset"
JSON_PATH = "dataset_google_psf.json"
SAMPLES_TO_CONSIDER = 22050*2



def preprocess_dataset(dataset_path, json_path, num_fft=2048):
    """Ekstraksi MFCCs from music dataset and saves them into a json file.

    :param dataset_path (str): Path to dataset
    :param json_path (str): Path to json file used to save MFCCs
    :param num_mfcc (int): Number of coefficients to extract
    :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
    :param hop_length (int): Sliding window for FFT. Measured in # of samples
    :return:
    """

    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # loop through all sub-dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        if dirpath is not dataset_path:

            label = dirpath.split("/")[-1]
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))

            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # memuat file audio dan mengirisnya untuk memastikan konsistensi panjang di antara file yang berbeda
                sample_rate, signal = wav.read(file_path)
                #signal=np.array(signal)
                if len(signal) >= SAMPLES_TO_CONSIDER:

                    # memastikan konsistensi panjang sinyal yang digunakan
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    # Ekstraksi MFCCs
                    MFCCs = psf.mfcc(signal, sample_rate, nfft=num_fft)

                    # store data for analysed track
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["labels"].append(i-1)
                    data["files"].append(file_path)
                    print("{}: {}".format(file_path, i-1))
                    print(MFCCs.shape)

    # save data in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, JSON_PATH)