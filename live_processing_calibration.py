# Read multi-channel OD and ADC data from OxySoft via LSL and process it
# © Johann Benerradi
# 1) Setup LSL on OxySoft
# 2) Run this script

import mne
import nirsimple as ns
import numpy as np
import pandas as pd
import re
import warnings
import os
import keyboard

from mne.preprocessing.nirs import temporal_derivative_distribution_repair
from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_stream
from scipy.stats import linregress


GYRO = ['HEADING', 'ROLL', 'PITCH']
mne.set_log_level(verbose=False)


def main(device="octamon"):
    """
    Read multi-channel OD and ADC data from OxySoft via LSL and process it.

    Parameters
    ----------
    device : string
        Device to emulate, can be `'octamon'` (default) or `'brite'`.
    """
    # Data stream
    data_stream = resolve_stream("name", "OxySoft")[0]
    data_inlet = StreamInlet(data_stream)
    info = data_inlet.info()
    name, sfreq, stream_type = info.name(), info.nominal_srate(), info.type()
    print(f"Reading {name} stream of {sfreq} Hz {stream_type}...")
    all_chs = []
    all_ch_wls = []
    ch = info.desc().child("channels").child("channel")
    for _ in range(info.channel_count()):
        label = ch.child_value("label")
        if re.match(r"\[\d*\] .* \[\d*nm\]", label):
            all_chs.append(label.split("] ")[1].split(" [")[0])
            all_ch_wls.append(label.split(" [")[1].split("nm]")[0])
        else:
            all_chs.append(label)
            all_ch_wls.append(None)
        ch = ch.next_sibling()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    keep = pd.read_csv(dir_path + f"/Octamon/info/optodes_{device}.csv")
    ch_rls = keep.columns
    ch_sds = keep.values[0]
    indices_keep = [all_chs.index(ch_keep) for ch_keep in ch_rls]
    ch_names = [ch.split(' ')[0] for ch in ch_sds]
    ch_wls = [all_ch_wls[i] for i in indices_keep]
    ch_dpfs = [6.0 for _ in ch_sds]
    ch_distances = keep.values[1].astype(float)

    indices_gyro = [all_chs.index(ch_gyro) for ch_gyro in GYRO
                    if ch_gyro in all_chs]

    # BCI stream
    bci_info = StreamInfo("BCI Class", "Fear Data", 3, 1, 'float32')
    bci_outlet = StreamOutlet(bci_info)

    od_buffer = np.empty((len(indices_keep), 0))
    gyro_buffer = np.empty((len(indices_gyro), 0))
    
    # Preset all adaptation values
    peak_sample = 0.0
    peak_average = 0.0
    average_peak = 0.0
    sampling = False
    readyToSample = False
    count = 0
    
    while True:
        
        # Check for space pressed on keyboard (inline with input for video start in Unity)
        if keyboard.is_pressed('space') and sampling == False:
            #Allow sampling to start
            sampling = True
            print("Sampling START")

            #Reset values for next samples
            peak_sample = 0.0
            average_peak = 0.0
            peak_average = 0.0
            count = 0
        
        # Get new sample
        sample, timestamp = data_inlet.pull_sample()
        sample_array = np.array(sample)
        od = sample_array[indices_keep]
        od = od[:, np.newaxis]
        gyro = sample_array[indices_gyro]
        gyro = gyro[:, np.newaxis]

        # Append new sample
        od_buffer = np.append(od_buffer, od, axis=1)
        gyro_buffer = np.append(gyro_buffer, gyro, axis=1)

        # Wait until buffer has 10 sec of data
        if od_buffer.shape[1] < (10*sfreq) and readyToSample == False:
            continue
        elif readyToSample == False:
            readyToSample = True
            print("Ready To Sample")

        # Preprocessing
        warnings.simplefilter("ignore")
        dod = ns.od_to_od_changes(od_buffer)
        mbll_data = ns.mbll(dod, ch_names, ch_wls, ch_dpfs, ch_distances, "cm")
        dc, ch_names, ch_types = mbll_data
        warnings.resetwarnings()

        # Loading with MNE
        mne_ch_names = [f"{ch} {ch_types[i]}" for i, ch in enumerate(ch_names)]
        info = mne.create_info(ch_names=mne_ch_names, sfreq=sfreq,
                               ch_types=ch_types)
        raw = mne.io.RawArray(dc, info)

        # TDDR
        raw = temporal_derivative_distribution_repair(raw)

        # Bandpass filtering
        iir_params = dict(order=4, ftype="butter", output="sos")
        raw.filter(0.01, 1.5, method="iir", iir_params=iir_params)

        # Feature extraction
        raw_hbo_mean = raw.get_data(picks="hbo", units="uM").mean(axis=0)
        
        # slope = linregress(raw.times, raw_hbo_mean).slope
        
        if sampling == True:
            current_peak = 0.0
        
            # Grab the average hbo values and the current peak value
            for i in raw_hbo_mean:
                count += 1
                peak_average += i
                if i > peak_sample:
                    peak_sample = i
                if i > current_peak:
                    current_peak = i
            
            average_peak += current_peak
                        
            # Process gyro data
            if indices_gyro:
                gyro_std = gyro_buffer.std()
            else:
                gyro_std = None

            # Print processed data
            print(f"Peak Sample: {peak_sample} | Peak Average: {peak_average / count} | Average Peak: {average_peak / (count / 100)} | Samples: {count}")

            # Create list of peak and averages
            peaks_output = [peak_sample, peak_average / count, average_peak / (count / 100)]

            # Send BCI class
            bci_outlet.push_sample(peaks_output)
            
            #Stop sampling once 32000 samples taken
            #32000 refers to 32 seconds of sampling (extra time for 30 second videos)
            if(count >=32000):
                sampling = False
                print("Sampling END")

        # Pop oldest sample
        od_buffer = np.delete(od_buffer, 0, axis=1)
        gyro_buffer = np.delete(gyro_buffer, 0, axis=1)

if __name__ == "__main__":
    main("octamon")
    # main("brite")
