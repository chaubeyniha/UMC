## use these helper functions in the jupyter notebook because is uses local variables


# FUNCTION TO ADD FILE CODE TO EMA DATA
def add_file(dataframe): 
    ''' 
        Using the list, get the index as the file code and add a new column with this index information
    '''
    for idx, frame in enumerate(dataframe):
        frame['File code'] = idx
        first_column = frame.pop('File code')
        frame.insert(0, 'File code', first_column)
        

# FUNCTION TO ADD FILE CODE TO AUDIO DATA   
def add_file_audio(dataframe): 
    ''' 
        Using the list, get the index as the file code and add a new column with this index information
    '''
    for idx, frame in enumerate(dataframe):
        data_frame = frame.insert(0, 'File Code', idx)
        
# FUNCTION FOR APPENDING LIST        
def append_list(position):
    '''
        Append the list with the mat data and return list with dataframes from each file
    '''
    if(position == 0):
        list_df = audio_df.append(pandas.DataFrame.from_dict(data[0][0][2]))
    elif(position == 1):
        list_df = UL_df.append(pandas.DataFrame.from_dict(data[0][1][2]))
    elif(position == 2):
        list_df = LL_df.append(pandas.DataFrame.from_dict(data[0][2][2]))
    elif(position == 3):
        list_df = JW_df.append(pandas.DataFrame.from_dict(data[0][3][2]))
    elif(position == 4):
        list_df = TD_df.append(pandaspd.DataFrame.from_dict(data[0][4][2]))
    elif(position == 5):
        list_df = TB_df.append(pandas.DataFrame.from_dict(data[0][5][2]))
    elif(position == 6):
        list_df = TT_df.append(pandas.DataFrame.from_dict(data[0][6][2]))
    
    return list_df

# FUNCTION FOR PREPROCESSING DATA 
def preprocess(list_name, ema_name):
    '''
        Prepare the data to be used.
    '''
    # add file code to the dataframes and merge all files
    add_file(list_name)
    merged_file = pandas.concat(list_name, axis = 0)
    
    # init column names and rename 
    column1 = "{}_0".format(ema_name)
    column2 = "{}_1".format(ema_name)
    column3 = "{}_2".format(ema_name)
    merged_file.columns = ['File Code', column1 , column2, column3]
    
    return merged_file

def preprocess_audio(list_name):
    '''
         Preprocessing steps of audio dataframes
    ''' 
    add_file_audio(list_name)
    merged_file_audio = pandas.concat(list_name, axis = 0)
    
    return merged_file_audio
        
# FUNCTION FOR OBTAINING SAMPLING RATE
def get_srate(file_number):
    '''
        From the ema files get the sampling rate
    ''' 
    directory = 'data/Data/F1/mat'
    
    # still needs to ignore the .DS_Store file in a better way
    file = sorted(os.listdir(directory))[file_number + 1]
    
    f = os.path.join(directory, file)
    mat = scipy.io.loadmat(f)['usctimit_ema_f1_{:03}_{:03}'.format(file_number*5 + 1, file_number*5 + 5)]
    
    #returns the srate which is stored here
    return mat[0][1][1][0][0]


# FUNCTION FOR GETTING COORIDNATES FOR WORD TIMEFRAME
def get_values(df_list, merged_df, file_code): 
    '''
        Goes through list of dataframe and if it matches the file code it gets the coordinates for th
    '''
    for i in range(len(df_list)):
        if (file_code == i):
            current_df = merged_df.loc[merged_df['File Code'] == i]
            current_var = current_df.iloc[starting_point:ending_point, :]
            
    return current_var

# FUNCTION FOR GETTING COORIDNATES FOR WORD AUDIO SEGMENT
def get_values_audio(a_df, merged_a, file_code): 
    '''
        Goes through list of dataframe and if it matches the file code it gets the coordinates for th
    '''
    for i in range(len(a_df)):
        if (file_code == i):
            current_df = merged_a.loc[merged_a['File Code'] == i]
            segment = current_df.iloc[start_point:end_point, :]
    return segment  


# convert raw audio to mel_spectogram
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