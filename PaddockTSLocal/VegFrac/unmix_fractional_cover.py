import numpy as np

def f(surface_reflectance, fc_model, inNull=0, outNull=0):
    """

    Unmixes an array of surface reflectance.

    :param surface_reflectance: The surface reflectance data organized in a 3D array
        with shape (nbands, nrows, ncolumns). There should be 6 bands with values
        scaled between 0 and 1.
    :type surface_reflectance: numpy.ndarray
    :param fc_model: The TensorFlow Lite model interpreter, which should be initialized
        as shown in the included code block.
    :type fc_model: tflite_runtime.interpreter.Interpreter
    :param inNull: The null value for the input image. Values in the input array
        equal to this will be replaced by `outNull`.
    :type inNull: float, optional
    :param outNull: The null value to replace in the output array.
    :type outNull: float, optional

    :return: A 3D array where the first layer corresponds to bare ground,
        the second layer corresponds to green vegetation, and the third layer
        corresponds to non-green vegetation.

    """

    # Drop the Blue band. Blue is yukky
    inshape = surface_reflectance[1:].shape
    # reshape and transpose so it is (nrow x ncol) x 5
    ref_data = np.reshape(surface_reflectance[1:], (inshape[0], -1)).T

    # Run the prediction
    inputDetails = fc_model.get_input_details()
    outputDetails = fc_model.get_output_details()
    fc_model.resize_tensor_input(inputDetails[0]['index'], ref_data.shape)
    fc_model.allocate_tensors()
    fc_model.set_tensor(inputDetails[0]['index'], ref_data.astype(np.float32))
    fc_model.invoke()
    fc_layers = fc_model.get_tensor(outputDetails[0]['index']).T
    output_fc = np.reshape(fc_layers, (3, inshape[1], inshape[2]))
    # now do the null value swap
    output_fc[output_fc == inNull] = outNull
    return output_fc


