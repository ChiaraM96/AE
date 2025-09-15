import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import json
import h5py
from scipy.stats import zscore
import pdb

## USE PYTHON 3.10 TO BUILD THE POINTPROCESS LIBRARY!!! 

class PointProcessModule:
    def __init__(self, path_to_lib):
        self.path_to_lib = path_to_lib
        sys.path.append(self.path_to_lib)
        from pointprocess import (compute_full_regression, 
                                  get_ks_coords, 
                                  Distributions)
        self.compute_full_regression = compute_full_regression
        self.get_ks_coords = get_ks_coords
        self.distributions = Distributions

    def computePP(self, rr):
        coords_dict = {}
        d_dict = {}
        for order in range(6, 12):
            order = order * 2
            try:
                d, coords_dict[order] = self.fullRegression(rr, order=order)
                d_dict[order] = d
            except KeyError:
                print('Regression failed for order ', order)
        return d_dict, coords_dict

    def evalPP(self, rr):
        window = 60 * 5
        d_dict, coords_dict = self.computePP(rr[rr < window])
        plt.figure(figsize=[5, 5])
        colors = sns.color_palette("husl", len(d_dict.keys()))
        ks = {}
        for order in d_dict.keys():
            try:
                ks[order] = max(np.sqrt((
                    (np.array(coords_dict[order].z) - np.array(coords_dict[order].inner)) ** 2
                    + (np.array(range(len(np.array(coords_dict[order].z))))
                        / len(np.array(coords_dict[order].z))
                        - np.array(coords_dict[order].inner)) ** 2)))
                plt.plot(coords_dict[order].z, coords_dict[order].inner, "k",
                         linewidth=0.8, label=f'{str(order)}, ks = {ks[order]:.2e}',
                         color=colors[int((order - min(d_dict.keys())) / 2)])
            except Exception as e:
                print(f'Error processing order {order}: {e}')
                pass
        order = list(d_dict.keys())[0]
        plt.plot(coords_dict[order].lower, coords_dict[order].inner, "b", linewidth=0.5)
        plt.plot(coords_dict[order].upper, coords_dict[order].inner, "b", linewidth=0.5)
        plt.plot(coords_dict[order].inner, coords_dict[order].inner, "r", linewidth=0.5)
        plt.legend()
        plt.show()
        best_order = min(ks, key=ks.get)
        del d_dict, coords_dict
        return best_order

    def plotPP(self, rr, d, id_temp):
        fig, axs = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        fig.suptitle('drive' + id_temp)
        axs[0].plot(rr[1:], np.diff(rr), label="RR intervals")
        axs[0].plot(d['Time'], d['Mu'], label="RR intervals (PP)")
        axs[0].set_title('RR intervals and fitted distribution')
        axs[0].legend()

        axs[1].plot(d['Time'], d['powLF'] / d['powHF'], label='LF/HF')
        axs[1].set_title('Sympathovagal Balance')

        axs[2].plot(d['Time'], d['powHF'], label='HF')
        axs[2].plot(d['Time'], d['powLF'], label='LF')
        axs[2].plot(d['Time'], d['powVLF'], label='VLF')
        axs[2].plot(d['Time'], d['powTOT'], label='Total', color='red',
                    linewidth=1, linestyle='--')
        axs[2].set_title('Power spectral density')
        axs[2].legend()

        axs[0].set_xlim([0, rr[-1]])
        for i in range(3):
            axs[i].set_xticks(range(0, int(rr[-1]), 60))
            axs[i].set_xlabel('Time (s)')
            axs[i].set_xticklabels(axs[i].get_xticks() / 60)
        plt.show()


    def fullRegression(self, rr, order=10, window_length=90.0, delta=0.005, alpha=0,
                       max_iter=10000, ks_plot=False):
        print(f"Computing regression for order {order}")
        result = self.compute_full_regression(
            events=rr.flatten(),
            window_length=window_length,
            delta=delta,
            ar_order=order,
            has_theta0=True,
            right_censoring=True,
            alpha=alpha,
            distribution=self.distributions.InverseGaussian,
            max_iter=max_iter
        )
        result.compute_hrv_indices()
        d = result.to_dict()
        coords = self.get_ks_coords(result.taus)
        if ks_plot:
            plt.figure(figsize=[3, 3])
            plt.plot(coords.z, coords.inner, "k", linewidth=0.8)
            plt.plot(coords.lower, coords.inner, "b", linewidth=0.5)
            plt.plot(coords.upper, coords.inner, "b", linewidth=0.5)
            plt.plot(coords.inner, coords.inner, "r", linewidth=0.5)
            plt.show()
        return d, coords

class FilesModule:
    def __init__(self, path_to_mat):
        self.matlabPath = path_to_mat
        sys.path.append(self.matlabPath)

    def importMatlab(self, id_temp):
        data = {}
        file_path = os.path.join(self.matlabPath, id_temp + '_Annotations.mat')
        print(file_path)
        if not os.path.exists(file_path):
            print('File not found!')
            return None, None, None
            
        with h5py.File(file_path, 'r') as f:
            data = {}
            for key in f['data'].keys():
                obj = f['data'][key]
                if isinstance(obj, h5py.Dataset):
                    data[key] = obj[:]
                elif isinstance(obj, h5py.Group):
                    data[key] = {}
                    for key_temp in obj.keys():
                        sub_obj = obj[key_temp]
                        if isinstance(sub_obj, h5py.Dataset):
                            data[key][key_temp] = sub_obj[:]

        keys = []
        annotation_keys = []
        for key in data.keys():
            if key == 'annotations':
                for Key in data[key].keys():
                    if isinstance(data[key][Key], h5py.Group):
                        continue
                    if len(data[key][Key]) > 5:
                        annotation_keys.append(Key)
            if len(data[key]) > 5:
                keys.append(key)

        if 'annotations' in keys:
            keys.remove('annotations')

        # FIXED: Handle multi-dimensional arrays properly
        data_dict = {}  # Dictionary to collect all column data
        
        for key in keys:
            try:
                # print(f"Processing key: {key}")
                # if hasattr(data[key], 'shape'):
                    # print(f"Data shape for {key}: {data[key].shape}")
                
                if isinstance(data[key], np.ndarray):
                    if data[key].ndim == 0:
                        # 0D array (scalar) - skip or convert to single-element array
                        # print(f"Warning: {key} is a scalar, skipping...")
                        continue
                        
                    elif data[key].ndim == 1:
                        # 1D array - use as single column
                        if len(data[key]) > 0:  # Only include non-empty arrays
                            data_dict[key] = data[key]
                        else:
                            # print(f"Warning: {key} is empty, skipping...")
                            continue
                            
                    elif data[key].ndim == 2:
                        # 2D array - split into separate columns
                        if data[key].shape[0] == 0:  # Skip empty arrays
                            # print(f"Warning: {key} is empty, skipping...")
                            continue
                            
                        if data[key].shape[1] == 3:
                            # For 3D data (like accelerometer X, Y, Z)
                            if 'ACCEL' in key:
                                data_dict[f"{key}_X"] = data[key][:, 0]
                                data_dict[f"{key}_Y"] = data[key][:, 1] 
                                data_dict[f"{key}_Z"] = data[key][:, 2]
                            else:
                                # For other 3-column data
                                data_dict[f"{key}_col1"] = data[key][:, 0]
                                data_dict[f"{key}_col2"] = data[key][:, 1]
                                data_dict[f"{key}_col3"] = data[key][:, 2]
                        
                        elif data[key].shape[1] == 1:
                            # Single column 2D array - flatten to 1D
                            data_dict[key] = data[key][:, 0]
                        
                        else:
                            # Multiple columns - create separate columns for each
                            for col_idx in range(data[key].shape[1]):
                                data_dict[f"{key}_col{col_idx}"] = data[key][:, col_idx]
                    
                    else:
                        # Higher dimensional - flatten (fallback)
                        # print(f"Warning: {key} has {data[key].ndim} dimensions, flattening...")
                        flattened = data[key].flatten()
                        if len(flattened) > 0:
                            data_dict[key] = flattened
                        else:
                            # print(f"Warning: {key} flattened to empty array, skipping...")
                            continue
                else:
                    # Non-numpy data - convert to numpy array
                    try:
                        arr = np.array(data[key])
                        if arr.size > 0 and arr.ndim > 0:  # Only include non-empty, non-scalar arrays
                            data_dict[key] = arr.flatten() if arr.ndim > 1 else arr
                        else:
                            # print(f"Warning: {key} converted to empty/scalar array, skipping...")
                            continue
                    except:
                        # print(f"Warning: Could not convert {key} to numpy array, skipping...")
                        continue
                        
            except Exception as e:
                print(f"Error processing key {key}: {e}")
                continue

        # Check if we have any data
        if not data_dict:
            # print("Warning: No valid data columns found!")
            return pd.DataFrame(), {}, 1000

        # Create DataFrame from the processed data dictionary
        # print(f"Creating DataFrame with {len(data_dict)} columns")
        
        # FIXED: Check array lengths more robustly
        def get_length(arr):
            """Get length of array, handling scalars and different types"""
            if hasattr(arr, '__len__'):
                return len(arr)
            elif hasattr(arr, 'size'):
                return arr.size if arr.ndim > 0 else 0  # Return 0 for scalars
            else:
                return 0  # For other types
        
        lengths = {k: get_length(v) for k, v in data_dict.items()}
        
        # Remove any columns with length 0
        data_dict = {k: v for k, v in data_dict.items() if lengths[k] > 0}
        lengths = {k: v for k, v in lengths.items() if v > 0}
        
        if not data_dict:
            # print("Warning: No valid data columns after filtering!")
            return pd.DataFrame(), {}, 1000
        
        unique_lengths = set(lengths.values())
        
        if len(unique_lengths) > 1:
            # print("Warning: Arrays have different lengths:")
            for k, length in lengths.items():
                print(f"  {k}: {length}")
            
            # Find the most common length
            from collections import Counter
            length_counts = Counter(lengths.values())
            target_length = length_counts.most_common(1)[0][0]
            # print(f"Using target length: {target_length}")
            
            # Trim or pad arrays to match target length
            for k, v in data_dict.items():
                current_length = lengths[k]
                if current_length != target_length:
                    if current_length > target_length:
                        data_dict[k] = v[:target_length]  # Trim
                        # print(f"Trimmed {k} from {current_length} to {target_length}")
                    else:
                        # Pad with last value or zeros
                        if current_length > 0:
                            pad_value = v[-1] if hasattr(v, '__getitem__') else 0
                        else:
                            pad_value = 0
                        padding = np.full(target_length - current_length, pad_value)
                        data_dict[k] = np.concatenate([v, padding])
                        # print(f"Padded {k} from {current_length} to {target_length}")

        try:
            df_data = pd.DataFrame(data_dict)
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
            # Debug: print problematic data
            # for k, v in data_dict.items():
                # print(f"  {k}: type={type(v)}, shape={getattr(v, 'shape', 'N/A')}")
            return pd.DataFrame(), {}, 1000

        # Process annotations (same as before)
        df_annotations = {}
        for key in annotation_keys:
            try:
                df_annotations[key] = pd.DataFrame(data['annotations'][key].astype(int))
                if key == 'ecg':
                    df_annotations[key].columns = ['R', 'P', 'Q', 'S', 'T']
                elif key == 'ppg':
                    df_annotations[key].columns = ['Sys', 'Notch', 'Dia']
                elif key == 'resp':
                    df_annotations[key].columns = ['Valley', 'Peak']
            except Exception as e:
                print(f"Error processing annotation key {key}: {e}")
                continue

        try:
            fs = data['Fs'][0].astype(int)
        except:
            fs = 256
            print("Warning: Could not extract sampling frequency, using default " + str(fs) + " Hz")
        
        print(f"Final DataFrame shape: {df_data.shape}")
        print(f"DataFrame columns: {list(df_data.columns)}")
        
        return df_data, df_annotations, fs

    def getRR(self, df_annotations, fs):
        r = df_annotations['ecg']['R'] / fs
        rr = np.diff(r)
        threshold = 3
        for i in range(3):
            mean = np.mean(rr)
            std = np.std(rr)
            z_diff = zscore(rr)
            outliers = np.array(np.where(z_diff > threshold))
            z_diff[outliers] = 3
            rr = z_diff * std + mean
        rr_new = np.cumsum(rr)
        return rr_new

    def importDataset(self, datasetPath, id_temp):
        print(f'LOADING ID {id_temp}')
        data, df_annotations, fs = self.importMatlab(id_temp)
        if os.path.exists(os.path.join(datasetPath, 'json', id_temp + '.json')):
            d = json.load(open(os.path.join(datasetPath, 'json', id_temp + '.json')))
        elif os.path.exists(os.path.join(datasetPath, 'parquet', id_temp + '.parquet')):
            d = pd.read_parquet(os.path.join(datasetPath, 'parquet', id_temp + '.parquet'), engine="pyarrow")
        d = {key: np.array(value) if isinstance(value, list)
             else value for key, value in d.items()}
        return d, df_annotations, fs, data
        
    def createTable(self, d, df_annotations, data, fs, id_temp, datasetPath = ""):

        if datasetPath and any([d, df_annotations, data, fs] is None):
            d, df_annotations, fs, data = self.importDataset(datasetPath, id_temp)
        
            
        pp_time = np.array(d['Time']).flatten()
        fs_pp = int(1 / np.mean(np.diff(pp_time)))
        pp_time = pd.to_datetime(pp_time, unit='s')
        lf = np.array(d['powLF']).flatten()
        hf = np.array(d['powHF']).flatten()
        lfhf = lf / hf
        rr = self.getRR(df_annotations, fs)
        # gsr = data['gsr']
        # resp = data['resp_edr']
        time = pd.to_datetime(np.arange(data.shape[0]) / fs, unit='s')
        hr = np.diff(rr)
        r = pd.to_datetime(rr[1:], unit='s')
        hr = 60 * np.ones(pp_time.shape) / np.interp(pp_time, r, hr)
        # gsr = np.interp(pp_time, time, gsr)
        # resp = np.interp(pp_time, time, resp)
        time_since_start = range(len(pp_time)) / np.array(fs_pp)

        cols = ["ID",
                "Recording start",
                "Time since start (s)",
                "Skin Conductance (microS)",
                "Heart Rate (bpm)",
                "Sympathovagal Balance (a.u.)",
                "Respiration (a.u.)"]
        
        df = pd.DataFrame(columns=cols)

        for var, varName in zip(['gsr', 'resp_edr'], [cols[3], cols[6]]):
            if var in data.columns:
                varTemp = data[var]
                varTemp = np.interp(pp_time, time, varTemp)
                df[varName] = varTemp
            
        df['Time since start (s)'] = time_since_start
        df['Heart Rate (bpm)'] = hr
        df['Sympathovagal Balance (a.u.)'] = lfhf
        df["Recording start"] = pp_time[0]
        df['ID'] = id_temp
        
        return df
