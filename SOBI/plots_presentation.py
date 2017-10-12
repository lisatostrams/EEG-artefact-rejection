# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 10:03:40 2017

@author: Lima
"""

from scipy import signal
import numpy as np
from sim_data import SimData
from joint_diagonalizer import jacobi_angles
import time

Data = SimData()

import matplotlib.pyplot as plt