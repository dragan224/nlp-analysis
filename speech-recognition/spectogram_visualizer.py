
import numpy
import matplotlib.pyplot
import sys
import scipy.io.wavfile # Potrebne biblioteke: numpy, scipy i matplotlib

def SpectogramFromWavFile(wav_file, window_size, window_hop_length,
                          apply_window):
  fs, signal = scipy.io.wavfile.read(wav_file)
  length = len(signal)
  overlap = int(fs * window_hop_length)
  framesize = int(window_size * fs)
  number_of_frames = (length / overlap)
  
  window_function = numpy.ones(framesize)
  if apply_window == 1: # hanning
    window_function = numpy.hanning(framesize)
  elif apply_window == 2: # hamming
    window_function = numpy.hamming(framesize)

  frames = numpy.ndarray((number_of_frames, framesize))
  for i in range(0, number_of_frames):
    for j in range(0, framesize):
      if ((i * overlap + j) < length):
        try:
          len(signal[i * overlap + j])
          frames[i][j] = signal[i * overlap + j][0] * window_function[j]
        except:
          frames[i][j] = signal[i * overlap + j] * window_function[j]
      else:
        frames[i][j] = 0
  
  fft_matrix = numpy.ndarray((number_of_frames, framesize)) 
  abs_fft_matrix = numpy.ndarray((number_of_frames, framesize))
  for i in range(0, number_of_frames):
    fft_matrix[i] = numpy.fft.fft(frames[i])

  matplotlib.pyplot.plot(range(len(fft_matrix)), fft_matrix)
  matplotlib.pyplot.ylabel('frekvencija')
  matplotlib.pyplot.xlabel('vreme')
  matplotlib.pyplot.show()

# Ovaj window size nije isti kao gore....
def SpectogramFromSinusoids(n, window_size, window_hop_length, 
                            apply_window):
  x = numpy.arange(0, window_size, window_hop_length)

  # prozor
  framesize = len(x)
  window_function = numpy.ones(framesize)
  if apply_window == 1: # hanning
    window_function = numpy.hanning(framesize)
  elif apply_window == 2: # hamming
    window_function = numpy.hamming(framesize)

  print "Unesite %d sinusoida u obliku a*sin(b*pi)" % n
  print "Za svaki sinusoid unesite dva broja a i b sa razmakom na odvojenim linijama."
  y = 0
  while n > 0:
    a, b = map(int, sys.stdin.readline().split())
    y += 1.*a*numpy.sin(1.*b * numpy.pi * x)
    n -= 1

  dft = numpy.fft.fft(y * window_function)
  matplotlib.pyplot.plot(range(len(dft)), dft)
  matplotlib.pyplot.show()


def main():
  print "Unesite broj 1 ili 2"
  print "1) Ucitavanje wav fajla."
  print "2) Zbir n sinusoida."
  cmd = int(raw_input())
  if cmd == 1:
    wav_file = raw_input('Unesite putanju wav fajla\n')
    print "Izaberite velicinu prozora u sekundama (razdaljina imzedju susednih semplova)"
    window_size = float(raw_input())
    print "Izaberite velicinu pomeraja prozora, utice na preklapanje."
    print "Utice na preklapanje izmedju susednih prozora."
    print "Tj preklapanje izmedju dva susedna prozora je velicina prozora minus pomeraj."
    window_hop_length = float(raw_input())
    print "Izaberite filtrirajuci prozor"
    print "0) Nema filtera."
    print "1) Hanning prozor."
    print "2) Hamming prozor."
    apply_window = int(raw_input())
    SpectogramFromWavFile(wav_file, window_size, window_hop_length, apply_window)
  else:
    print "Unesite broj sinusoida koji se sabiraju"
    n = int(raw_input())
    print "Izaberite filtrirajuci prozor"
    print "0) Nema filtera."
    print "1) Hanning prozor."
    print "2) Hamming prozor."
    apply_window = int(raw_input())
    SpectogramFromSinusoids(n, 6, 0.25, apply_window)


main()
# SpectogramFromWavFile('test.wav', 0.001, 0.01, 0)
# SpectogramFromWavFile('test.wav', 0.001, 0.01, 1)
# SpectogramFromWavFile('test.wav', 0.001, 0.01, 2)



