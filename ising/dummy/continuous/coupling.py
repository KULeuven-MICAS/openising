import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class Bounds:
    def __init__(self, low, high):
        if low >= high:
            raise ValueError("low bound should be strictly lower than high bound")
        self.low = low
        self.high = high
        self.delta = high - low
        self.mid = low + self.delta/2

def mcp(state, coupling, bounds):
    state_offset = state-bounds.mid
    return 1/np.delta * np.dot(coupling, state_offset)

def Hmcp(state, coupling, bounds):
    state_offset = state-bounds.mid
    return - 1/np.delta * np.dot(state_offset.T, np.dot(coupling, state_offset))

def pen(state, coef, bounds):
    return np.exp(coef * (bounds.low - state)) - np.exp(coef*(state - bounds.high))

def Hpen(state,coef, bounds):
    return -1/coef * (np.exp(coef*(bounds.low - state)) + exp(coef*(state - bounds.high))
