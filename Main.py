import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import matplotlib
from pylsl import StreamInlet, resolve_stream
import threading
import time
import scipy.signal as sig
import pygame
from datetime import datetime

pygame.init()
pygame.mixer.init()
pygame.mixer.pre_init(44100, -16, 2, 2048)

pygame.mixer.music.load('C.wav')
sounds = [pygame.mixer.Sound('C.wav'), pygame.mixer.Sound('D.wav'), pygame.mixer.Sound('E.wav')]

matplotlib.use('TkAgg')

# Create link for LSL
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])
EEG = np.zeros([1])
EEG1 = np.zeros([1])

time.sleep(2)  # Makes Python wait for 2 seconds, so as to fill buffer

# Filter parameters
fs = 256
order = 4
nyq = 0.5 * fs
low = 1 / nyq
high = 40 / nyq
b, a = sig.butter(order, [low, high], btype='bandpass')


def data_acquisition():
    global inlet, EEG, EEG1, b, a
    while True:
        ch = 1
        chunk_i = inlet.pull_sample()
        chunk = chunk_i[:][0][0]
        EEG = np.append(EEG, np.array(chunk))
        if np.size(EEG) > 256 * 5:
            EEG1 = sig.filtfilt(b, a, EEG)
    return EEG1


acq_thread = threading.Thread(target=data_acquisition)
acq_thread.start()


##
def update(frame):
    # update the line plot:
    line2[0].set_xdata(t)
    line2[0].set_ydata(EEG1[ini:fin] - np.mean(EEG1[ini:fin]))
    ax[0].set(ylim=[min(EEG1[ini:fin]), max(EEG1[ini:fin])])
    # PSD
    ax[1].cla()
    FFT = ax[1].psd(EEG1[ini:fin] - np.mean(EEG1[ini:fin]), NFFT=301, Fs=fs, scale_by_freq=True)
    ax[1].set(xlim=[0, 60], ylim=[-10, 40])
    calculate_power_sound(FFT)
    playsound_D()
    return line2, FFT


def calculate_power_sound(FFT):
    global power_dB_ts_z, power_dB_ts
    power_dB = np.array([0, 0, 0, 0, 0])
    for idx, i in enumerate(power_idx):
        power_dB[idx] = sum(FFT[0][i[0]])
    power_dB_ts = np.vstack([power_dB_ts, power_dB])
    power_dB_ts_z = (power_dB_ts[-100:] - np.mean(power_dB_ts[-100:], 0)) / np.std(power_dB_ts[-100:], 0)


# Basic parameters for the plotting window
fs = 256
plot_duration = 1 + (5 * fs)  # how many seconds of data to show
update_interval = 60  # ms between screen updates
pull_interval = 0.2  # s between each pull operation
delay = int(0.1 * fs)
ini = -plot_duration - delay
fin = -delay
time_start = datetime.now()

t = np.linspace(0, plot_duration / fs, plot_duration)
fig, ax = plt.subplots(2, 1)
line2 = ax[0].plot(t, EEG1[ini:fin] - np.mean(EEG1[ini:fin]))
ax[0].set(xlim=[0, 5], xlabel='Time [s]', ylabel='Voltage [mV]')
ax[0].grid(True)
FFT = ax[1].psd(EEG[ini:fin] - np.mean(EEG[ini:fin]), NFFT=301, Fs=fs, scale_by_freq=True)
ax[1].set(xlim=[0, 60])
fig.tight_layout()
D_idx = np.where((1 <= FFT[1]) & (FFT[1] <= 4))  # Delta: 1- 4 Hz
T_idx = np.where((4 <= FFT[1]) & (FFT[1] <= 8))  # Theta: 4-8 Hz
A_idx = np.where((8 <= FFT[1]) & (FFT[1] <= 12))  # alpha: 8-12 Hz
Lb_idx = np.where((12 <= FFT[1]) & (FFT[1] <= 16))  # low beta: 12-16 Hz
Hb_idx = np.where((16 <= FFT[1]) & (FFT[1] <= 30))  # high beta: 16-30 Hz

power_idx = [D_idx, T_idx, A_idx, Lb_idx, Hb_idx]
power_dB_ts = np.array([0, 0, 0, 0, 0])
power_dB_ts_z = np.array([0, 0, 0, 0, 0])


def playsound_D():
    global power_dB_ts_z
    for i in range(3):
        if np.mean(power_dB_ts_z[-10:-1, i] >= 2):
            sounds[i].play()


ani = animation.FuncAnimation(fig=fig, func=update, interval=pull_interval)
plt.show()

##
