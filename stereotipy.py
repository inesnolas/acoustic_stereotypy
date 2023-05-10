import librosa
import numpy as np
import sklearn
import skimage
import os
import csv
import pickle
from skimage.feature import match_template
from scipy.signal import find_peaks
import argparse
from tqdm import tqdm 
import pandas as pd
import glob




def get_Pos_events_in_time_interval(events_csv, start=0, end=50000 ): #default value for end time argument is over 10 hours! 

    
    allevents = pd.read_csv(events_csv)
    POS_events = allevents[allevents["Q"]=="POS"]

    POS_events_in_interval = POS_events[(POS_events["Starttime"] >= start) & (POS_events["Endtime"]< end)]
    
    return POS_events_in_interval


def compute_stft(audiofilename):
    waveform, sr = librosa.load(audiofilename, sr = None)
    nfft=int(sr/10)
    hop_len = int(nfft/4)
    stft = np.abs(librosa.stft(waveform, n_fft=nfft, hop_length=hop_len, window='hann', pad_mode='reflect'))
    
    # noise reduction
    # subtraction of frequency bin median value per bin and time frame median value per time frame
    stft_median = np.median(stft, axis=-1, keepdims=True)
    stft_time_median = np.median(stft, axis=0, keepdims=True)
    norm_stft = stft - stft_median
    norm_stft = norm_stft - stft_time_median
    return sr, hop_len, norm_stft



def compute_similarity_with_5examples(events_csv, audiofilename):
    events = get_Pos_events_in_time_interval(events_csv)
        #sort event by startime and select only first 5
    events = events.sort_values(by='Starttime')
    
    template_events = events[0:5] # check indexes!!!
    events =events.drop(events.index[:5])   # index=([0:4]) not accepted??>?!
    
    sr, hop_len, stft = compute_stft(audiofilename)  
    
        
    result = []
    # for each 5shot events, 
    # for index, row in df.iterrows()
    for index, template in template_events.iterrows():
        r = []
        #compute its sft
        stft_template = stft[2:-2,int(np.floor(template['Starttime'])*sr/hop_len + 1):int(np.ceil(template['Endtime'])*sr/hop_len + 1)]
        #sample N=30? ramdomly from POS events pool
        try:
            random_selected_POS_events = events.sample(n=30)
        except ValueError:
            random_selected_POS_events = events
        #compute sfts for selected POS events
        for index, e in random_selected_POS_events.iterrows():
            stime = e['Starttime']
            etime = e['Endtime']
            stft_event = stft[2:-2, int(np.floor(stime*sr/hop_len + 1)): int(np.ceil(etime*sr/hop_len + 1))]
            # run template matching 
            if stft_template.shape[1] <= stft_event.shape[1]:
                r.append(np.max(match_template(stft_event, stft_template )))
            else:
                r.append(np.max(match_template(stft_template,stft_event )))
        result.append(np.mean(r))   #result contains for each template (5) the averaged value of its similarity with the 30 events sampled (5,1)

    return np.mean(result)  #(1,) 
                

def compute_stereotipy(events_csv, audiofilename):
    events = get_Pos_events_in_time_interval(events_csv)
        #sort event by startime and select only first 5
    events = events.sort_values(by='Starttime')
    
    template_events = events.sample(n=10)   # instead of selecting just the 5 (5 shots) let's select 10? totally random POS events
    events =events.drop(index=template_events.index)  
    
    sr, hop_len, stft = compute_stft(audiofilename)  
    
        
    result = []
    # for each template event, 
    # for index, row in df.iterrows()
    for index, template in template_events.iterrows():
        r = []
        #compute its sft
        stft_template = stft[2:-2,int(np.floor(template['Starttime'])*sr/hop_len + 1):int(np.ceil(template['Endtime'])*sr/hop_len + 1)]
        #sample N=30? ramdomly from POS events pool
        try:
            random_selected_POS_events = events.sample(n=30)
        except ValueError:
            random_selected_POS_events = events
        #compute sfts for selected POS events
        for index, e in random_selected_POS_events.iterrows():
            stime = e['Starttime']
            etime = e['Endtime']
            stft_event = stft[2:-2, int(np.floor(stime*sr/hop_len + 1)): int(np.ceil(etime*sr/hop_len + 1))]
            # run template matching 
            if stft_template.shape[1] <= stft_event.shape[1]:
                r.append(np.max(match_template(stft_event, stft_template )))
            else:
                r.append(np.max(match_template(stft_template,stft_event )))
        result.append(np.mean(r))   #result contains for each template (5) the averaged value of its similarity with the 30 events sampled (5,1)

    return np.mean(result)  #(1,) 
if __name__ == "__main__":


    datasets = os.listdir("/import/c4dm-datasets/DCASE_2022_FSBioSED/Evaluation_set/")
    
    
    
    for dataset_folder in datasets:   
        print(os.path.basename(dataset_folder))
        stereotipy={}        
        files_in_dataset = glob.glob("/import/c4dm-datasets/DCASE_2022_FSBioSED/Evaluation_set/"+dataset_folder +'/*.csv')
        for events_csv in files_in_dataset:
            
            audiofilename = events_csv[0:-4]+'.wav'
            print(os.path.basename(audiofilename))
            stereotipy[os.path.basename(audiofilename)] =  [compute_stereotipy(events_csv, audiofilename)] 
        pd.DataFrame(stereotipy.items()).to_csv('/homes/in304/dcase-few-shot-bioacoustic/utils/stereotipy/'+'stereotipy_in_'+ os.path.basename(dataset_folder)+'_with_10POSrandomlySelectedexamples.csv', index=False)

