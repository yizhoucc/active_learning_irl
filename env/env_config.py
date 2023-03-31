# this file is for the collection of parameters

# this is for log variance noise parameter
import numpy as np
from numpy import pi

class Config:
    def __init__(self):
        # timeout
        self.timeout=500 # timesteps

        # -------------- param range ---------------------------
        self.gravity_range = [1,100]#9.8,
        self.masscart_range = [0.1,10]#1.0,
        self.masspole_range = [0.01,3]#0.1,
        self.length_range = [0.1,10]#0.5,  # actually half the pole's length
        self.force_mag_range = [1,100]#10.0
        self.tau=0.02
        # -------------- param sampling ----------------------
        self.reso = 3
        self.sample_method='log'

        # fixed
        self.gravity = 9.8
        self.masscart = 1
        self.masspole = 0.1
        self.length = None  # actually half the pole's length
        self.force_mag = 10





