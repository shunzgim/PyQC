import sys
from os.path import dirname, abspath
path = dirname(dirname(abspath(__file__)))
sys.path.append(path+'/libs')

from .full_amplitude_sim import FullAmplitudeSimulator