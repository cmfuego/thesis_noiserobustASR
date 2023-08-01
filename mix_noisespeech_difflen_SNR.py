# This script mixes speech files with a noise at multiple SNRs, and accounts for different lengths of speech and noise files.
# Each noise file is used multiple times, at different SNRs. If there are less noise files than speech files, noise files may be reused.
# The function add_noise is largely based on https://github.com/zqs01/data2vecnoisy/blob/7cbfa5d04151bb9f25e281e500f222f323526afd/examples/data2vec_noisy/data/raw_audio_dataset.py#L344

import glob
import numpy as np
from scipy.io.wavfile import read, write
import random
import re

### Change paths to the path where your noise files, speech files are stored, and where the output files should be stored.
noisefolderpath = "OtherNoise_leftovers/test_noises"     # "OtherNoise_leftovers/train_noises"    # NB don't include final slash
speechfolderpath = "LibriSpeech/test-clean"            # "LibriSpeech/dev-clean"                # NB don't include final slash
finalfolder = "OtherNoise_final/test_selfmade"         # "OtherNoise_final/train"               # NB don't include final slash

### Get file lists (glob.glob)
noise_list = glob.glob(fr"{noisefolderpath}/**/*.wav", recursive="True")
speech_list = glob.glob(fr"{speechfolderpath}/**/*.wav", recursive="True")
snr_list = ["0", "5", "10", "15", "20"]


def add_noise(snr, noise_wav, clean_wav):
        
        snr = int(snr)

        ### Read wav information
        global sr1, sr2
        sr1, noise_array = read(noise_wav)
        sr2, clean_array = read(clean_wav)
        assert sr1 == sr2
        
        clean_array = clean_array.astype(np.float32)
        noise_array = noise_array.astype(np.float32)
               
        ### Fit noise to length of speech
        if len(clean_array) > len(noise_array):
            ratio = int(np.ceil(len(clean_array)/len(noise_array)))
            noise_array = np.concatenate([noise_array for _ in range(ratio)])
        if len(clean_array) < len(noise_array):
            start = 0
            noise_array = noise_array[start: start + len(clean_array)]
        #print("noise_array dtype ndim shape size array \n", noise_array.dtype, noise_array.ndim, noise_array.shape, clean_array.size, "\n", noise_array)
        
        ### Calculate mean speech & noise volume
        clean_rms = np.sqrt(np.mean(np.square(clean_array), axis=-1))
        noise_rms = np.sqrt(np.mean(np.square(noise_array), axis=-1))

        ### Adjust noise volume to SNR
        adjusted_noise_rms = clean_rms / (10**(snr/20))
        adjusted_noise_array = noise_array * (adjusted_noise_rms / noise_rms)
        ### Mix speech + adjusted noise
        mixed = clean_array + adjusted_noise_array
        
        ### Avoid clipping noise
        max_float16 = np.finfo(np.float16).max
        min_float16 = np.finfo(np.float16).min
        if mixed.max(axis=0) > max_float16 or mixed.min(axis=0) < min_float16:
            if mixed.max(axis=0) >= abs(mixed.min(axis=0)): 
                reduction_rate = max_float16 / mixed.max(axis=0)
            else :
                reduction_rate = max_float16 / mixed.min(axis=0)
            mixed = mixed * (reduction_rate)
        mixed = mixed.astype(np.float16)
        return mixed  


### Execute function for all speechfiles
temp_noise_snr_list = []
for clean_wav in speech_list:
    
    ### Determine which file should get which noise at which SNR
    # Make a list of noises at each possible SNR (so for every noisename 5 types from SNR 0-20) for the function to choose from
    if temp_noise_snr_list == []:
        for item in noise_list:
            for snr in snr_list:
                temp_noise_snr_list.append(item + snr)
    
    random_item = random.choice(temp_noise_snr_list)    # choose a random noisefile (snr included) to use

    ### Get info about snr and noise_wavname (which form a combined string in temp_noise_snr_list, so the correct part has to be selected)
    snr = re.findall(r"\d+$", random_item)[0]
    noise_wav = re.findall(r"(.*wav)\d+$", random_item)[0]

    ### Execute mixing function
    result = add_noise(snr, noise_wav, clean_wav)
    result = np.rint(result)
    result = result.astype(int)

    ### Make sure that the same noise+snr cannot be chosen again (until the whole noise dataset is used once)
    temp_noise_snr_list.remove(random_item)
    
    cleanwavID = re.findall(r"\/(\d+-\d+-\d+)\.wav", clean_wav)[0]
    noisesnrID = re.findall(r"\/(\w+)\.", random_item)[0].replace("_", "") + "_" + snr + "dB"
    
    ### Save result of add_noise() as wav with sr=16000 in train_noisy_speech
    outfilename = f"{cleanwavID}_{noisesnrID}.wav"
    write(f"{finalfolder}/{outfilename}", sr1, result.astype('int16'))
