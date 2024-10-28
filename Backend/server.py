import torch
import numpy as np
import soundfile as sf
from flask import Flask, request, jsonify

MODEL_PATH = 'U.E.P_sweet.pth'