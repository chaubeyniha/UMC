import pandas as pd 

def logmelfilterbank(audio,
                     sampling_rate,
                     fft_size=1024,
                     hop_size=256,
                     win_length=None,
                     window="hann",
                     num_mels=80,
                     fmin=None,
                     fmax=None,
                     eps=1e-10,
                     ):
    ''' essentially the same as librosa.feature.melspectrogram + log10 '''

    # get amplitude spectrogram
    x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, pad_mode="reflect")
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sampling_rate, fft_size, num_mels, fmin, fmax)

    return np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))

# get longest and shortest mel spectogram array length >> 28885
def get_arr_length(df):
    """Loads data and tracks longest mel spectogram array
    :return counter: the longest length of array
    """
    counter_longest = 0
    counter_shortest = 3000 # arbritary number
    shapes = []
    shape_audio = []
    
    for i in range(len(df)):
        mel_array = df.iloc[i, :]['Input']
        shapes.append(mel_array.shape[0])
        
        if mel_array.shape[0] > counter_longest:
            counter_longest = mel_array.shape[0]    
            
        elif mel_array.shape[0] < counter_shortest:
            counter_shortest = mel_array.shape[0]
            
    return counter_longest, counter_shortest, shapes
 
def prepare_dataframe(ema_list):
    df = pd.DataFrame(columns=['Input', 'Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5', 'Label 6',
                              'Label 7', 'Label 8', 'Label 9', 'Label 10', 'Label 11', 'Label 12'])
    
    for i in range(len(ema_list)):
        waveform = ema_list[i]['Audio'][0][0].values
        mel_arr = logmelfilterbank(waveform, 22050)
        mel_arr = librosa.util.fix_length(mel_arr, size=95)                             # add padding
        
        ema_columns = ['ul_0', 'ul_1', 'll_0', 'll_1', 'jw_0', 'jw_1', 'tt_0', 'tt_1', 'tb_0', 'tb_1', 'td_0', 'td_1']
        ema_markers = ema_list[i]['Data'][0].loc[:, ema_columns]
        ema_arr = (ema_markers.to_numpy()).T
        
        df = df.append({'Input': mel_arr, 'Label 1': ema_arr[0], 'Label 2': ema_arr[1], 'Label 3': ema_arr[2], 
                        'Label 4': ema_arr[3], 'Label 5': ema_arr[4], 'Label 6': ema_arr[5], 'Label 7': ema_arr[6], 
                        'Label 8': ema_arr[7], 'Label 9': ema_arr[8], 'Label 10': ema_arr[9], 'Label 11': ema_arr[10], 
                        'Label 12': ema_arr[11]}, ignore_index=True)
        
    return df