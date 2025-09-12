import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import json
import h5py
from scipy.stats import zscore

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
        file_path = os.path.join(self.matlabPath, id_temp + '_ann.mat')
        print(file_path)
        if not os.path.exists(file_path):
            print('File not found')
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
                        # elif isinstance(sub_obj, h5py.Group):
                        #     # You could go deeper if needed
                        #     data[key][key_temp] = "Nested group (not loaded)"
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
        df_data = pd.DataFrame(columns=keys)

        for key in keys:
            df_data[key] = pd.DataFrame(data[key])

        df_annotations = {}
        for key in annotation_keys:
            df_annotations[key] = pd.DataFrame(data['annotations'][key].astype(int))
            if key == 'ecg':
                df_annotations[key].columns = ['R', 'P', 'Q', 'S', 'T']
            elif key == 'ppg':
                df_annotations[key].columns = ['Sys', 'Notch', 'Dia']
            elif key == 'resp':
                df_annotations[key].columns = ['Valley', 'Peak']

        fs = data['Fs'][0].astype(int)
        return df_data, df_annotations, fs

    def getRR(self, df_annotations, fs):
        r = df_annotations['ecg']['R'] / fs
        rr = np.diff(r)
        mean = np.mean(rr)
        std = np.std(rr)
        z_diff = zscore(rr)
        threshold = 3
        outliers = np.array(np.where(z_diff > threshold))
        z_diff[outliers] = 3
        rr = z_diff * std + mean
        rr_new = np.cumsum(rr)
        return rr_new

    def createTable(self, id_temp, datasetPath):
        print(f'LOADING ID {id_temp}')
        data, df_annotations, fs = self.importMatlab(id_temp)
        d = json.load(open(os.path.join(datasetPath, 'json', id_temp + '.json')))
        d = {key: np.array(value) if isinstance(value, list)
             else value for key, value in d.items()}
        pp_time = np.array(d['Time']).flatten()
        fs_pp = int(1 / np.mean(np.diff(pp_time)))
        pp_time = pd.to_datetime(pp_time, unit='s')
        lf = np.array(d['powLF']).flatten()
        hf = np.array(d['powHF']).flatten()
        lfhf = lf / hf
        rr = self.getRR(df_annotations, fs)
        gsr = data['gsr']
        resp = data['resp_edr']
        time = pd.to_datetime(range(len(gsr)) / fs, unit='s')
        hr = np.diff(rr)
        r = pd.to_datetime(rr[1:], unit='s')
        hr = 60 * np.ones(pp_time.shape) / np.interp(pp_time, r, hr)
        gsr = np.interp(pp_time, time, gsr)
        resp = np.interp(pp_time, time, resp)
        time_since_start = range(len(gsr)) / np.array(fs_pp)

        cols = ["ID",
                "Recording start",
                "Time since start (s)",
                "Skin Conductance (microS)",
                "Heart Rate (bpm)",
                "Sympathovagal Balance (a.u.)",
                "Respiration (a.u.)"]

        df = pd.DataFrame(columns=cols)
        df['Time since start (s)'] = time_since_start
        df['Skin Conductance (microS)'] = gsr
        df['Heart Rate (bpm)'] = hr
        df['Sympathovagal Balance (a.u.)'] = lfhf
        df["Recording start"] = pp_time[0]
        df['Respiration (a.u.)'] = resp
        df['ID'] = id_temp
        
        return df
