# FUNCTIONING FOR ADDING FILE CODE
def add_file(dataframe): 
    ''' 
        Using the list, get the index as the file code and add a new column with this index information
    '''
    for idx, frame in enumerate(dataframe):
        frame['File code'] = idx
        first_column = frame.pop('File code')
        frame.insert(0, 'File code', first_column)
        

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