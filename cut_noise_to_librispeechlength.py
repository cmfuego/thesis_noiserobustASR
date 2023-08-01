# THIS SCRIPT EXTRACTS SMALL AUDIO SELECTIONS (length same as audio files in another folder)

longwavpath = 'GymNoise_mono_16khz/16khz_longwavs/'
outputnoisefolder = 'GymNoise_mono_16khz/noise_pieces/'

import glob
import random
from scipy.io.wavfile import read, write
import re
import numpy as np

# Make lists of the full speech datasets --> make one list of all speech files we're using
# (all dev-clean files for fine-tuning and a random selection of 120 test-clean files for testing)
speechwav_list_train = glob.glob(r'LibriSpeech/dev-clean/**/*.wav', recursive=True)
speechwav_list_test = glob.glob(r'LibriSpeech/test-clean/**/*.wav', recursive=True)
speechwav_selection = []
for fname in speechwav_list_train:
    speechwav_selection.append(fname)
for i in range(120):
    fpath = random.choice(speechwav_list_test)
    speechwav_selection.append(fpath)
with open("used_speechwavs.txt", "w") as f: # save the list of all used speech file paths to .txt
    f.write('\n'.join(speechwav_selection))

# Create a list with all lengths of the speech files that will be used
dict_of_lens = {}
for fname in speechwav_selection:
    sr, array = read(fname) # load array
    file_len = len(array) # get length
    dict_of_lens[fname] = file_len

# For every picked short sound, select a long file to extract from, choose a part from that file array, and save it as a new sound in a different folder
longwav_list = glob.glob(longwavpath + '*.wav')
with open("longwav_list.txt", "w") as f: # save the list of all longwavs to .txt - this way we can easily retrieve from which file the slices are taken after running the script if necessary
    f.write('\n'.join(longwav_list))

# Create a list of dicts, where slice information (+ corresponding speech filename) will be stored per longwav
noise_slices = []
longwav_lens = []
for longwav in longwav_list:
    noise_slices.append({})
    sr, array = read(longwav)
    longwav_lens.append(len(array))

count = 0
for fname, length in dict_of_lens.items():
    while True:
        longwav_idx = random.randrange(0,len(longwav_list)) # pick a random noisefile to extract noise from
        chosen_longwav_len = longwav_lens[longwav_idx]        

        slice_end1 = random.randint(length, chosen_longwav_len) # choose a random slice from the file (array) with len
        slice_begin1 = slice_end1-length
        
        # Make sure that no slices overlap, by looking at the earlier used slices from that file    
        for done_slice_key in noise_slices[longwav_idx]:
            interval = noise_slices[longwav_idx][done_slice_key]
            if not (slice_end1 < interval[0] or interval[1] < slice_begin1):
                break
        else:
            break

    # If an interval is found that does not overlap: save the chosen slice (value) with the filename (key) in the dict for the corresponding longwav
    chosen_slice = [slice_begin1,slice_end1]
    noise_slices[longwav_idx][fname] = chosen_slice
    count += 1
#    break # let's first break after one round to check if it works
    if count%100==0: print(count)

with open("used_noise_slices.txt", "w") as f:
    f.write('\n'.join(str(item) for item in noise_slices))

# If all works and we have a good list of slices, extract them all and save in a new folder:
for longwav_slicedict in noise_slices:
    longwav_idx = noise_slices.index(longwav_slicedict)
    longwav_fname = longwav_list[longwav_idx]
    longwav_id = re.findall(r"DR-100_([0-9a-zA-Z]+)_16khz\.wav", longwav_fname)[0]
    sr, array = read(longwav_fname)
    for key, value in longwav_slicedict.items():
        speechwav_fname = key
        speechwav_id = re.findall(r"\/(\d+-\d+-\d+)\.wav", speechwav_fname)[0]
        slice_info = value
        chosen_array = array[slice_info[0]:slice_info[1]]
        write(f"{outputnoisefolder}{speechwav_id}_{longwav_id}.wav", sr, chosen_array)
