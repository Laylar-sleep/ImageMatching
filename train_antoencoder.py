import matplotlib

matplotlib.use("Agg")

from autoencoder import Autoencoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2


def visualize_predictions(decoded, gt, samples=10):
    outputs = None

    # loop over our number of output samples
    for i in range(0, samples):
        # grab the original image and reconstructed image
        original = (gt[i] * 255).astype("uint8")
        recon = (decoded[i] * 255).astype("uint8")
        # stack the original and reconstructed image side-by-side
        output = np.hstack([original, recon])
        # initialize it as the current side-by-side image display
        if outputs is None:
            outputs = output
        # otherwise, vertically stack the outputs
        else:
            outputs = np.vstack([outputs, output])
    # return the output images
    return outputs
