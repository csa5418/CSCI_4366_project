import os
import matplotlib.pyplot as plt
from pydub import AudioSegment
from tempfile import mktemp
from scipy.io import wavfile
#imports youll need

#youll probably need this too idk why but I needed it
AudioSegment.converter = "C:\\Users\\csa54\\OneDrive\\Documents\\CSCI_4366\\ffmpeg.exe"

#paths youll need for audio clips
path = "C:\\Users\\csa54\\OneDrive\\Documents\\CSCI_4366\\dataset\\TrainAudioFiles"
audio_clips = os.listdir("C:\\Users\\csa54\\OneDrive\\Documents\\CSCI_4366\\dataset\\TrainAudioFiles")

#convert audio clips to spectrograms to image by going from mp3 to wav to spectrograms
#kinda miserable and takes an hour but like whatever
for i in range(0, 3000):
    num = audio_clips[i].split(".")
    cur_path = path+"\\"+audio_clips[i]
    mp3_audio = AudioSegment.from_file(file = cur_path, format="mp3") 
    #mp3 needs to be converted to wav to work
    wname = mktemp('.wav')  
    mp3_audio.export(wname, format="wav") 
    rate, data = wavfile.read(wname) 
    fig,ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')
    pxx, freqs, bins, im = ax.specgram(x=data[:,0], Fs=rate, noverlap=384, NFFT=512)
    ax.axis('off')
    saveImg = "C:\\Users\\csa54\\OneDrive\\Documents\\CSCI_4366\\dataset\\TrainImageFiles"
    fig.savefig(os.path.join(saveImg, num[0]+'.png'), bbox_inches='tight', dpi=300, transparent=True)
    plt.close()
    
