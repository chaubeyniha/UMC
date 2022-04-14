# UMC
 
Library use USC Timit database (https://sail.usc.edu/span/usc-timit/). The goal is to have an array of input which represents the raw waveform of a word spoken as an input to a neural network which then is able to predict the 6 ema markers. 

Dependencies: Python, NumPy, Pandas, PyTorch, Librosa, Tensorflow

# If you are using Anaconda then you can install all required
# Python packages by running the following commands in a shell:
#
#     conda create --name tf python=3
#     source activate tf
#     pip install -r requirements.txt

Dataset
is saved in the data/Data folder with information for 4 participants...

Preprocessing
Mel-Specogram in log form to mimick human hearing, where lower frequency are heard more 

Training
A Recurrent Neural Networks (RNN) based sequence-to-sequence encoder-decoder was used to acheve the goal stated above.
