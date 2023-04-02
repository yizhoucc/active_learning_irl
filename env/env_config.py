# this file is for the collection of parameters

# this is for log variance noise parameter
# import numpy as np
# from numpy import pi

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
        self.cost_range = [0.01,0.9]
        self.tau=0.02
        # -------------- param sampling ----------------------
        self.reso = 5
        self.sample_method='liner'

        # fixed params -----------------------------
        self.cost=None
        self.gravity = 9.8
        self.masscart = 1
        self.masspole = 0.1
        self.length = None  # actually half the pole's length
        self.force_mag = 10

        # inverse fix params, override training fix params
        self.length_theta=None
        self.cost_theta=None
        self.cost_phi=0.1
        self.length_phi=None





