# Aperiodic neural activity distinguishes between phasic and tonic REM sleep
# Yevgenia Rosenblum, Tamás Bogdány, Lili Benedikta Nádasy, Ilona Kovács, Ferenc Gombos, Péter Ujma, Róbert Bódizs, Nico Adelhöfer, Péter Simor, Martin Dresler
# Automated REM detection using YASA and MNE Python
# by Lili Benedikta Nádasy - lilibenedikta@gmail.com

# import necessary packages
# pip install mne
import numpy as np
import pandas as pd
import mne
import yasa
import math

# List of all participants
participants = [
    "P01",
    "P02",
    "P03",
    "P04",
    "P05",
    "P06",
    "P07",
    "P08",
    "P09",
    "P10",
    "P11",
    "P12",
    "P13",
    "P14",
    "P15",
    "P16",
    "P17",
    "P18",
    "P19",
    "P20",
]

# Define file paths (artefact, hypnogram and EDFs) at row 156!
# Hypnograms The night is divided into 20-sec windows, and each 20-sec window is assigned a number (nights are scored as 20-sec windows). Number 2 marks REM phase.
# Artefacts The night is divided into 4-sec windows, and each 4-sec window is assigned a number (1: clean, no artefact, 2: artefact, noise)

# Select window size
window = 4

# Initialize large matrices or lists to store results for all participants
all_results_unfilt = []
all_results_filt = []


output_path_unfilt = (
    rf"C:\Users\matrices_unfilt\all_results_unfilt_yasa_amplitude_{window}sec.txt"
)
output_path_filt = (
    rf"C:\Users\matrices_filt\all_results_filt_yasa_amplitude_{window}sec.txt"
)


# Definte the REM detection function that creates the matrix content
def detect_rems(raw_data, roc_channel, loc_channel, hypno, artefacts):
    """Detect REMs using YASA's REM detection function.

    Args:
        raw_data: Raw data object from MNE.
        roc_channel: Name of the ROC channel.
        loc_channel: Name of the LOC channel.

    Returns:
        REM detection results.
    """
    # Get the sampling frequency
    sf = raw_data.info["sfreq"]

    # Given your sampling frequency is 512 Hz, each 4-second window will have 512 * 4 = 2048 samples.
    window_size = window * sf  # 4-second windows
    num_windows = raw_data.n_times // window_size

    # Placeholder for results
    rem_window_results = []

    # Create copies of the raw data for each channel
    raw_copy_roc = raw_data.copy().pick([roc_channel])
    raw_copy_loc = raw_data.copy().pick([loc_channel])

    # Get the data for each channel
    ROC_data = raw_copy_roc.get_data(units="uV")[
        0
    ]  # Select first (and only) channel's data
    LOC_data = raw_copy_loc.get_data(units="uV")[
        0
    ]  # Select first (and only) channel's data

    # Detect REMs using YASA's REM detection function
    rem = yasa.rem_detect(ROC_data, LOC_data, sf=sf, hypno=None)

    # Extract the summary of REM events
    rem_events = rem.summary()

    # Iterate over each 4-second window
    for i in range(int(num_windows)):
        start_sec = i * window
        end_sec = start_sec + window
        hypno_value = hypno[
            i // (20 // window)
        ]  # Get hypnogram value for the current 4-sec window

        rem_in_window = rem_events[
            (rem_events["Peak"] >= start_sec) & (rem_events["Peak"] < end_sec)
        ]
        loc_max_amplitude = max(ROC_data[int(start_sec * sf) : int(end_sec * sf)])
        roc_max_amplitude = max(LOC_data[int(start_sec * sf) : int(end_sec * sf)])
        window_max_amplitude = max(loc_max_amplitude, roc_max_amplitude)

        artifact_value = artefacts[i // window]

        if not rem_in_window.empty:
            max_loc_abs_peak = rem_in_window["LOCAbsValPeak"].max()
            rem_window_results.append(
                (
                    participant,
                    start_sec + start_tsec,
                    end_sec,
                    artifact_value,
                    hypno_value,
                    True,
                    window_max_amplitude,
                    max_loc_abs_peak,
                )
            )
        else:
            max_loc_abs_peak = 0
            rem_window_results.append(
                (
                    participant,
                    start_sec + start_tsec,
                    end_sec,
                    artifact_value,
                    hypno_value,
                    False,
                    window_max_amplitude,
                    max_loc_abs_peak,
                )
            )

    return rem, sf, ROC_data, LOC_data, rem_window_results


# Define the function to save the data
def save_to_txt(file_path, data):
    # Convert list of tuples to a NumPy array for saving
    data_array = np.array(data, dtype=object)

    # Save the array to a text file
    np.savetxt(file_path, data_array, fmt="%s", delimiter=",")


for participant in participants:
    # Define paths
    edf_path = rf"C:\Users\EDFs\{participant}.edf"
    hypnogram_path = rf"C:\Users\hypnograms\{participant}.hyp.txt"
    artefact_path = rf"C:\Users\artefaktok\{participant}_art.txt"
    print(
        "######################Current participant: ",
        participant,
        "########################",
    )
    # Load participant's EDF
    # Read the file metadata without loading the data, but include only the 'EOG+' channel
    raw_data = mne.io.read_raw_edf(edf_path, preload=False, include="EOG+")

    # Get the total duration of the recording in seconds
    max_sec = raw_data.times[-1]

    # Now you can set your start and end times for cropping
    start_tsec = 0  # Start from the beginning
    end_tsec = max_sec  # End at the maximum time

    # Crop the data
    raw_data.crop(tmin=start_tsec, tmax=end_tsec)

    # Load the cropped data
    raw_data.load_data()

    # Load participant's hypnogram
    # Load the hypnogram file (had to rename it because of the "." and the " ")
    hypno = np.loadtxt(hypnogram_path)

    # Calculate the duration of the raw data in seconds
    raw_data_duration_sec = raw_data.times[-1]

    # Calculate the expected number of hypnogram data points
    expected_hypno_length = math.ceil(raw_data_duration_sec / 20)

    # Compare the actual hypnogram length with the expected length
    if len(hypno) == expected_hypno_length:
        print("The hypnogram file matches the length of the raw_data file.")
    else:
        print(
            f"Mismatch in lengths. Expected {expected_hypno_length} data points in the hypnogram, found {len(hypno)}."
        )

    # Load participant's artefact data
    artefacts = np.loadtxt(artefact_path)

    # Calculate the expected number of hypnogram data points
    expected_artefact_length = math.ceil(raw_data_duration_sec / 4)

    # Compare the actual hypnogram length with the expected length
    if len(artefacts) == expected_artefact_length:
        print("The artefact file matches the length of the raw_data file.")
    else:
        print(
            f"Mismatch in lengths. Expected {expected_artefact_length} data points in the artefact file, found {len(artefacts)}."
        )

    # Check if there is a mismatch in lengths
    if len(artefacts) < expected_artefact_length:
        missing_length = expected_artefact_length - len(artefacts)
        print(
            f"Mismatch in lengths. Expected {expected_artefact_length} data points in the artefact file, found {len(artefacts)}. Appending {missing_length} missing rows."
        )

        # Create an array of ones for the missing data points
        missing_data = np.ones(missing_length)

        # Append the missing data to the artefacts array
        artefacts = np.concatenate((artefacts, missing_data))
    else:
        print("The artefact file matches the length of the raw_data file.")

    # Positive and negative EOG channels
    # Specify the sampling frequency of your data
    sf = raw_data.info["sfreq"]

    # Ensure you're working with a copy of the raw data to prevent modifying the original
    raw_copy = raw_data.copy()

    # Pick the EOG+ channel from the copy
    eog_plus_data = raw_copy.pick(["EOG+"]).get_data()

    # Verify the shape and contents of eog_plus_data
    print("EOG+ Data Shape:", eog_plus_data.shape)
    print(
        "Sample values from EOG+:", eog_plus_data[0, :10]
    )  # Print first 10 samples for inspection

    # Create the negative EOG+ channel data
    eog_neg_data = -1 * eog_plus_data

    # Verify the transformation
    print(
        "Sample values from EOG- (negated):", eog_neg_data[0, :10]
    )  # Print first 10 samples for inspection

    # Create an MNE Info structure for the new channel
    ch_info = mne.create_info(ch_names=["EOG-"], sfreq=sf, ch_types=["eog"])

    # Create a new RawArray object for the negative EOG+ data
    eog_neg_raw = mne.io.RawArray(eog_neg_data, ch_info)

    # Append the new EOG-_neg channel to the original raw data
    raw_data.add_channels([eog_neg_raw], force_update_info=True)

    # Now, you can use raw_data with the new EOG-_neg channel for further processing

    # Apply bandpass filtering
    # Create a copy of the raw data
    raw_data_filtered = raw_data.copy()

    # Apply a bandpass filter between 0.3 and 10 Hz to the data (focus just EOG+/-)
    # raw_data_filtered.filter(l_freq=0.3, h_freq=10, picks=['EOG+','ROC-M1','LOC-M2','EOG-'])
    raw_data_filtered.filter(l_freq=0.3, h_freq=10, picks=["EOG+", "EOG-"])

    # Define custom scaling for EOG channels
    scalings = {"eog": 20e-6}  # This is an example scaling factor in Volts

    # Call the detect_rems function for both unfiltered and filtered data
    (
        rem_unfilt_eog,
        sf_unfilt,
        roc_unfilt_eog,
        loc_unfilt_eog,
        rem_window_results_unfilt_eog,
    ) = detect_rems(raw_data, "EOG+", "EOG-", hypno, artefacts)
    rem_filt_eog, sf_filt, roc_filt_eog, loc_filt_eog, rem_window_results_filt_eog = (
        detect_rems(raw_data_filtered, "EOG+", "EOG-", hypno, artefacts)
    )

    # Append the results to the all_results lists
    all_results_unfilt.extend(rem_window_results_unfilt_eog)
    all_results_filt.extend(rem_window_results_filt_eog)

    # Convert lists to NumPy arrays for efficient storage (if necessary)
    all_results_unfilt_array = np.array(all_results_unfilt)
    all_results_filt_array = np.array(all_results_filt)

# Save the data to text files
save_to_txt(output_path_unfilt, all_results_unfilt)
save_to_txt(output_path_filt, all_results_filt)

# File paths
filt_path = rf"C:\Users\matrices_filt\all_results_filt_yasa_amplitude_{window}sec.txt"

# Load the data from the file
# Replace 'unfilt_path' with 'filt_path' as needed
data_matrix = pd.read_csv(
    output_path_filt,
    delimiter=",",
    header=0,
    names=[
        "Participant",
        "Start Time",
        "End Time",
        "Artefact Value",
        "Hypnogram Value",
        "REM Detected",
        "Max Amplitude",
        "YASA Amplitude Peak",
    ],
)

# Convert 'REM Detected' to boolean
data_matrix["REM Detected"] = data_matrix["REM Detected"].astype(bool)

# Filter rows where Hypnogram Value is 2
rem_data = data_matrix[data_matrix["Hypnogram Value"] == 2]

# Split data based on REM Detection
rem_detected = rem_data[rem_data["REM Detected"]]
rem_not_detected = rem_data[~rem_data["REM Detected"]]

# Define output path for filtered file (with bandpass filter)
output_path_filt = (
    rf"C:\Users\matrices_filt\all_results_filt_yasa_amplitude_{window}sec_final.txt"
)

# Select only REM stage, non-artefact windows
data_matrix_filtered = data_matrix[
    (data_matrix["Artefact Value"] == 1) & (data_matrix["Hypnogram Value"] == 2)
]

# Define columns
data_matrix_filtered = data_matrix_filtered[
    [
        "Participant",
        "Start Time",
        "End Time",
        "REM Detected",
        "Max Amplitude",
        "YASA Amplitude Peak",
    ]
]

# Save final file
save_to_txt(output_path_filt, data_matrix_filtered)
