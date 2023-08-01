# This script combines two collection of files of the same length and format together, and saves the outcome files in a new folder

# Add the path of your audio folders (LibriSpeech and other recordings with the same lengths, created with the script cut_noise_to_librispeechlength.py)
noise_pieces_path = 'GymNoise_mono_16khz/noise_pieces/'
test_clean_path = 'LibriSpeech/test-clean'

import glob
import os
import re
import numpy as np
from scipy.io.wavfile import read, write

noise_pieces_list = glob.glob(noise_pieces_path + "*.wav")
test_clean_folders = [name for name in os.listdir(test_clean_path) if os.path.isdir(os.path.join(test_clean_path, name))]

# regex patterns for connecting the files in the right way
pattern_speakerID = r'/(\d+)-\d+-\d+_'
pattern_chapterID = r'-(\d+)-'
pattern_filename = r'(\d+-\d+-\d+)'
pattern_longwavID = r'(?<=_)\d{4}[a-zA-Z]*' 

for item in noise_pieces_list:
    speakerID = re.findall(pattern_speakerID, item)[0]
    chapterID = re.findall(pattern_chapterID, item)[0]
    filename = re.findall(pattern_filename, item)[0]
    longwavID = re.findall(pattern_longwavID, item)[0]

    if speakerID in test_clean_folders:
        libriset = 'test-clean'
        finalfolder = 'test'
    else:
        libriset = 'dev-clean'
        finalfolder = 'train'
        
    sr1, data1 = read(f"LibriSpeech/{libriset}/{speakerID}/{chapterID}/{filename}.wav")
    sr2, data2 = read(item)
    
    assert sr1 == sr2
    result = data1 + 15*data2    # choose a ratio that fits your audio data well
    write(f"Mixed_Gym_Libri/{finalfolder}/combined_{filename}_{longwavID}.wav", sr1, result)
