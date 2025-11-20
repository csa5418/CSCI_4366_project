import os
import matplotlib.pyplot as plt
from pydub import AudioSegment

import librosa
import librosa.display
from tempfile import mktemp
from scipy.io import wavfile

AudioSegment.converter = "C:\\Users\\csa54\\OneDrive\\Documents\\CSCI_4366\\ffmpeg.exe"

path = "C:\\Users\\csa54\\OneDrive\\Documents\\CSCI_4366\\dataset\\TrainAudioFiles"
path2 = "C:\\Users\\csa54\\Documents\\dataset\\TrainAudioFiles"
audio_clips = os.listdir("C:\\Users\\csa54\\OneDrive\\Documents\\CSCI_4366\\dataset\\TrainAudioFiles")
print("No. of audio files in audio folder = ",len(audio_clips))

print(audio_clips[0])

print(path+"\\"+audio_clips[0])

cur_path = path+"\\"+audio_clips[0]

mp3_audio = AudioSegment.from_file(file = cur_path, format="mp3")  # read mp3
wname = mktemp('.wav')  # use temporary file
mp3_audio.export(wname, format="wav")  # convert to wav
FS, data = wavfile.read(wname)  # read wav file
plt.specgram(data[:, 0], Fs=FS, NFFT=128, noverlap=0)  # plot
plt.show()