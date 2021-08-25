import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from pathlib import Path
import os
import sys
import getopt

def CriarMelSpecs(InputDir, OutputDir):
   count = 0
   for filename in os.listdir(InputDir):
      count += 1
      if "wav" in filename:
         try:
            file_path = os.path.join(InputDir, filename)
            file_stem = Path(file_path).stem # É o arquivo sem a extensao (.wav)
            filename = filename.split('.')[0]

            # Channels       : 1
            # Sample Rate    : 16000
            # Precision      : 16-bit
            # Duration       : 00:00:10.00 = 160000 samples ~ 750 CDDA sectors
            # File Size      : 320k = 256k x 10s / 8 (1byte = 8 bits)
            # Bit Rate       : 256k = 16k x 16-bit por segundo
            
            # parametros obtidos visualmente no SonicVisualizer
            n_fft = 2048 # cerca de 128ms this is the number of samples in a window per fft
            hop_length = 1796 # 87,5% de sobreposicao The amount of samples we are shifting after each fft
            n_mels = 64
            power = 2
            fmax = 8000
            fig= plt.figure()
            y, sr = librosa.load(file_path)
            y = librosa.util.normalize(y)
            mel_spect = librosa.feature.melspectrogram(y= y,
                                                sr= sr,
                                                n_fft= n_fft,
                                                hop_length= hop_length,
                                                center= True,
                                                n_mels= n_mels,
                                                power= power
                                                )
            log_mel_spect = librosa.power_to_db(mel_spect)
            librosa.display.specshow(log_mel_spect,
                                    y_axis='mel',
                                    fmax=fmax,
                                    x_axis='time',
                                    cmap='magma')
            plt.title('Mel-espectrograma: ' + filename)
            plt.colorbar(format='%+2.0f dB')
            print(filename)# + str(count))
            #outfile = os.path.join(OutputDir,filename + str(count) + ".png")
            outfile = os.path.join(OutputDir,filename  + ".png")
            plt.savefig(outfile)
         except:
               pass
         plt.close()



def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["idir=","odir="])
   except getopt.GetoptError:
      print('test.py -i <inputdir> -o <outputdir>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('test.py -i <inputdir> -o <outputdir>')
         sys.exit()
      elif opt in ("-i", "--idir"):
         inputfile = arg
      elif opt in ("-o", "--odir"):
         outputfile = arg
   print('Input dir is ', inputfile)
   print('Output dir is ', outputfile)

   CriarMelSpecs(inputfile, outputfile)

if __name__ == "__main__":
   main(sys.argv[1:])