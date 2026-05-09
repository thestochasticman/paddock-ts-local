"""Fractional cover unmixing.

Inference logic and the bundled tflite model files in `_models/` are adapted
from the fractionalcover3 package by Robert Denham, distributed under the
MIT License. See PaddockTS/LICENSES/fractionalcover3.LICENSE.
"""
from importlib import resources

import numpy as np
from tensorflow import lite as tflite


_MODELS = (
    "fcModel_32x32x32.tflite",
    "fcModel_64x64x64.tflite",
    "fcModel_256x64x256.tflite",
    "fcModel_256x128x256.tflite",
)


def get_model(n: int = 2):
    """Load the n-th bundled fractional cover tflite model (1-indexed, 1-4)."""
    path = resources.files(__package__) / "_models" / _MODELS[n - 1]
    return tflite.Interpreter(model_path=str(path))


def unmix_fractional_cover(surface_reflectance, fc_model, in_null=0, out_null=0):
    """Unmix a 6-band surface reflectance array into bare/green/non-green fractions.

    The blue band is dropped before inference (the model was trained on the
    remaining 5 bands). Returns a (3, nrows, ncols) float32 array.
    """
    inshape = surface_reflectance[1:].shape
    ref_data = np.reshape(surface_reflectance[1:], (inshape[0], -1)).T

    input_details = fc_model.get_input_details()
    output_details = fc_model.get_output_details()
    fc_model.resize_tensor_input(input_details[0]["index"], ref_data.shape)
    fc_model.allocate_tensors()
    fc_model.set_tensor(input_details[0]["index"], ref_data.astype(np.float32))
    fc_model.invoke()
    fc_layers = fc_model.get_tensor(output_details[0]["index"]).T
    output_fc = np.reshape(fc_layers, (3, inshape[1], inshape[2]))
    output_fc[output_fc == in_null] = out_null
    return output_fc
