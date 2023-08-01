# Copyright (c) 2018 - 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


from amd.rocal import readers
from amd.rocal import decoders
from amd.rocal import random
from amd.rocal import noise
from amd.rocal import reductions

import amd.rocal.types as types
import rocal_pybind as b
from amd.rocal.pipeline import Pipeline


def blend(*inputs, ratio=None, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - list containing the input images
    
    ratio (float, optional, default = None) - ratio used for blending one image with another

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    ratio = b.createFloatParameter(ratio) if isinstance(ratio, float) else ratio
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0], "input_image1": inputs[1], "is_output": False, "ratio": ratio,
                     "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    blend_image = b.Blend(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (blend_image)


def snow(*inputs, snow=0.5, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation
    
    snow (float, default = 0.5) - snow fill value used for the augmentation

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    snow = b.createFloatParameter(snow) if isinstance(snow, float) else snow
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "snow": snow,
                     "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    snow_image = b.Snow(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (snow_image)


def exposure(*inputs, exposure=0.5, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation
    
    exposure (float, default = 0.5) - exposure fill value used for the augmentation

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    exposure = b.createFloatParameter(exposure) if isinstance(exposure, float) else exposure
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "exposure": exposure,
                     "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    exposure_image = b.Exposure(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (exposure_image)


def fish_eye(*inputs, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation
    
    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False,
                     "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    fisheye_image = b.FishEye(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (fisheye_image)


def fog(*inputs, fog=0.5, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation
    
    fog (float, default = 0.5) - fog fill value used for the augmentation

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    fog = b.createFloatParameter(fog) if isinstance(fog, float) else fog
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0],
                     "is_output": False, "fog_value": fog, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    fog_image = b.Fog(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (fog_image)


def brightness(*inputs, alpha=None, beta=None,
               rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation
    
    alpha (float, optional, default = None) - brightness multiplier. Values >= 0 are accepted. For example: 0 - black image, 1 - no change, 2 - increase brightness twice

    beta (float, optional, default = None) - brightness shift

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    alpha = b.createFloatParameter(alpha) if isinstance(alpha, float) else alpha
    beta = b.createFloatParameter(beta) if isinstance(beta, float) else beta

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "alpha": alpha, "beta": beta,
                     "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    brightness_image = b.Brightness(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (brightness_image)


def brightness_fixed(*inputs, alpha=1.0, beta=0.0,
                     rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation
    
    alpha (float, optional, default = 1.0) - brightness multiplier. Values >= 0 are accepted. For example: 0 - black image, 1 - no change, 2 - increase brightness twice

    beta (float, optional, default = 0.0) - brightness shift

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "alpha": alpha, "beta": beta,
                     "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    brightness_image = b.BrightnessFixed(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (brightness_image)


def lens_correction(*inputs, strength=None, zoom=None,
                    rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation
    
    strength (float, optional, default = None) - strength value used for the augmentation

    zoom (float, optional, default = None) - zoom value used for the augmentation

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    strength = b.createFloatParameter(strength) if isinstance(strength, float) else strength
    zoom = b.createFloatParameter(zoom) if isinstance(zoom, float) else zoom

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "strength": strength, "zoom": zoom,
                     "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    len_corrected_image = b.LensCorrection(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (len_corrected_image)


def blur(*inputs, kernel_size=None, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation
    
    kernel_size (int, default = None) - kernel size used for the filter

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    kernel_size = b.createIntParameter(kernel_size) if isinstance(kernel_size, int) else kernel_size
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "kernel_size": kernel_size,
                     "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    blur_image = b.Blur(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (blur_image)


def contrast(*inputs, contrast_factor=None, contrast_center=None,
             rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation
    
    contrast_factor (float, optional, default = None) - contrast multiplier used for the augmentation. Values >= 0 are accepted. For example: 0 - gray image, 1 - no change, 2 - increase contrast twice

    contrast_center (float, optional, default = None) - intensity value unaffected by the augmentation

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    contrast_factor = b.createFloatParameter(contrast_factor) if isinstance(contrast_factor, float) else contrast_factor
    contrast_center = b.createFloatParameter(contrast_center) if isinstance(contrast_center, float) else contrast_center

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0],
                     "is_output": False, "contrast_factor": contrast_factor, "contrast_center": contrast_center, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    contrast_image = b.Contrast(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (contrast_image)


def flip(*inputs, h_flip=0, v_flip=0,
         rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation
    
    h_flip (int, optional, default = None) - flip the horizontal dimensions

    v_flip (int, optional, default = None) - flip the vertical dimension

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    h_flip = b.createIntParameter(h_flip) if isinstance(h_flip, int) else h_flip
    v_flip = b.createIntParameter(v_flip) if isinstance(v_flip, int) else v_flip

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0],
                     "is_output": False, "h_flip": h_flip, "v_flip": v_flip, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    flip_image = b.Flip(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (flip_image)


def gamma_correction(*inputs, gamma=0.5, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation
    
    gamma (float, default = 0.5) - gamma correction value used for the augmentation

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    gamma = b.createFloatParameter(gamma) if isinstance(gamma, float) else gamma
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0],
                     "is_output": False, "gamma": gamma, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    gamma_correction_image = b.GammaCorrection(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (gamma_correction_image)


def hue(*inputs, hue=None, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation
    
    hue (float, default = None) - hue change in degrees

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    hue = b.createFloatParameter(hue) if isinstance(hue, float) else hue
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0],
                     "is_output": False, "hue": hue, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    hue_image = b.Hue(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (hue_image)


def jitter(*inputs, kernel_size=None, seed=0, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation
    
    kernel_size (int, optional, default = None) - kernel size used for the augmentation

    seed (int, optional, default = 0) - seed used for randomization in the augmentation

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    kernel_size = b.createIntParameter(kernel_size) if isinstance(kernel_size, int) else kernel_size
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0],
                     "is_output": False, "kernel_size": kernel_size, "seed": seed, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    jitter_image = b.Jitter(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (jitter_image)


def pixelate(*inputs, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation
    
    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0],
                     "is_output": False, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    pixelate_image = b.Pixelate(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (pixelate_image)


def rain(*inputs, rain=None, rain_width=None, rain_height=None, rain_transparency=None,
         rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation
    
    rain (float, optional, default = None) - rain fill value used for the augmentation

    rain_width (int, optional, default = None) - width of the rain pixels for the augmentation
    
    rain_height (int, optional, default = None) - height of the rain pixels for the augmentation

    rain_transparency (float, optional, default = None) - transparency value used for the augmentation

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    rain = b.createFloatParameter(rain) if isinstance(rain, float) else rain
    rain_width = b.createIntParameter(rain_width) if isinstance(rain_width, int) else rain_width
    rain_height = b.createIntParameter(rain_height) if isinstance(rain_height, int) else rain_height
    rain_transparency = b.createFloatParameter(rain_transparency) if isinstance(rain_transparency, float) else rain_transparency

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "rain_value": rain, "rain_width": rain_width, "rain_height": rain_height,
                     "rain_transparency": rain_transparency, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    rain_image = b.Rain(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (rain_image)


def resize(*inputs, max_size=[], resize_longer=0, resize_shorter=0, resize_width=0, resize_height=0, scaling_mode=types.SCALING_MODE_DEFAULT, interpolation_type=types.LINEAR_INTERPOLATION,
           rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation

    max_size (int or list of int, optional, default = []) -

    Maximum size of the longer dimension when resizing with resize_shorter. When set with resize_shorter, the shortest dimension will be resized to resize_shorter if the longest dimension is smaller or equal to max_size. If not, the shortest dimension is resized to satisfy the constraint longest_dim == max_size. Can be also an array of size 2, where the two elements are maximum size per dimension (H, W).

    Example:

    Original image = 400x1200.

    Resized with:

        resize_shorter = 200 (max_size not set) => 200x600

        resize_shorter = 200, max_size =  400 => 132x400

        resize_shorter = 200, max_size = 1000 => 200x600

    resize_longer (int, optional, default = 0) - The length of the longer dimension of the resized image. This option is mutually exclusive with resize_shorter,`resize_x` and resize_y. The op will keep the aspect ratio of the original image.

    resize_shorter (int, optional, default = 0) - The length of the shorter dimension of the resized image. This option is mutually exclusive with resize_longer, resize_x and resize_y. The op will keep the aspect ratio of the original image. The longer dimension can be bounded by setting the max_size argument. See max_size argument doc for more info.

    resize_width (int, optional, default = 0) - The length of the X dimension of the resized image. This option is mutually exclusive with resize_shorter. If the resize_y is left at 0, then the op will keep the aspect ratio of the original image.

    resize_height (int, optional, default = 0) - The length of the Y dimension of the resized image. This option is mutually exclusive with resize_shorter. If the resize_x is left at 0, then the op will keep the aspect ratio of the original image.

    scaling_mode (int, optional, default = types.SCALING_MODE_DEFAULT) - resize scaling mode.

    interpolation_type (int, optional, default = types.LINEAR_INTERPOLATION) - Type of interpolation to be used.

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "dest_width:": resize_width, "dest_height": resize_height, "is_output": False, "scaling_mode": scaling_mode, "max_size": max_size, "resize_shorter": resize_shorter,
                     "resize_longer": resize_longer, "interpolation_type": interpolation_type, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    resized_image = b.Resize(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (resized_image)


def resize_crop_mirror(*inputs, resize_width=0, resize_height=0, crop_w=0, crop_h=0, mirror=1,
                       rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation

    resize_width (int, optional, default = 0) - The length of the X dimension of the resized image

    resize_height (int, optional, default = 0) - The length of the Y dimension of the resized image

    crop_w (int, optional, default = 0) - Cropping window width (in pixels).

    crop_h (int, optional, default = 0) - Cropping window height (in pixels).

    mirror (int, optional, default = 1) - flag for the horizontal flip.

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    if isinstance(mirror, int):
        if (mirror == 0):
            mirror = b.createIntParameter(0)
        else:
            mirror = b.createIntParameter(1)

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "dest_width:": resize_width, "dest_height": resize_height, "is_output": False, "crop_w": crop_w,
                     "crop_h": crop_h, "mirror": mirror, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    rcm = b.ResizeCropMirrorFixed(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (rcm)


def resize_crop(*inputs, resize_width=0, resize_height=0, crop_area_factor=None, crop_aspect_ratio=None, x_drift=None, y_drift=None,
                rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation

    resize_width (int, optional, default = 0) - The length of the X dimension of the resized image

    resize_height (int, optional, default = 0) - The length of the Y dimension of the resized image

    crop_area_factor (float, optional, default = None) - area factor used for crop generation

    crop_aspect_ratio (float, optional, default = None) - aspect ratio used for crop generation

    x_drift (float, optional, default = None) - x_drift used for crop generation

    y_drift (float, optional, default = None) - y_drift used for crop generation

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    crop_area_factor = b.createFloatParameter(crop_area_factor) if isinstance(crop_area_factor, float) else crop_area_factor
    crop_aspect_ratio = b.createFloatParameter(crop_aspect_ratio) if isinstance(crop_aspect_ratio, float) else crop_aspect_ratio
    x_drift = b.createFloatParameter(x_drift) if isinstance(x_drift, float) else x_drift
    y_drift = b.createFloatParameter(y_drift) if isinstance(y_drift, float) else y_drift

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "dest_width:": resize_width, "dest_height": resize_height, "is_output": False, "crop_area_factor": crop_area_factor,
                     "crop_aspect_ratio": crop_aspect_ratio, "x_drift": x_drift, "y_drift": y_drift, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    rcm = b.CropResize(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (rcm)


def resize_mirror_normalize(*inputs, max_size=[], resize_longer=0, resize_shorter=0, resize_width=0, resize_height=0, scaling_mode=types.SCALING_MODE_DEFAULT,
                            interpolation_type=types.LINEAR_INTERPOLATION, mean=[0.0], std=[1.0], mirror=1, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation

    max_size (int or list of int, optional, default = []) -

    Maximum size of the longer dimension when resizing with resize_shorter. When set with resize_shorter, the shortest dimension will be resized to resize_shorter if the longest dimension is smaller or equal to max_size. If not, the shortest dimension is resized to satisfy the constraint longest_dim == max_size. Can be also an array of size 2, where the two elements are maximum size per dimension (H, W).

    Example:

    Original image = 400x1200.

    Resized with:

        resize_shorter = 200 (max_size not set) => 200x600

        resize_shorter = 200, max_size =  400 => 132x400

        resize_shorter = 200, max_size = 1000 => 200x600

    resize_longer (int, optional, default = 0) - The length of the longer dimension of the resized image. This option is mutually exclusive with resize_shorter,`resize_x` and resize_y. The op will keep the aspect ratio of the original image.

    resize_shorter (int, optional, default = 0) - The length of the shorter dimension of the resized image. This option is mutually exclusive with resize_longer, resize_x and resize_y. The op will keep the aspect ratio of the original image. The longer dimension can be bounded by setting the max_size argument. See max_size argument doc for more info.

    resize_width (int, optional, default = 0) - The length of the X dimension of the resized image. This option is mutually exclusive with resize_shorter. If the resize_y is left at 0, then the op will keep the aspect ratio of the original image.

    resize_height (int, optional, default = 0) - The length of the Y dimension of the resized image. This option is mutually exclusive with resize_shorter. If the resize_x is left at 0, then the op will keep the aspect ratio of the original image.

    scaling_mode (int, optional, default = types.SCALING_MODE_DEFAULT) - resize scaling mode.

    interpolation_type (int, optional, default = types.LINEAR_INTERPOLATION) - Type of interpolation to be used.

    mean (list of floats, optional, default = [0.0]) - mean used for normalization

    std (list of floats, optional, default = [1.0]) - standard deviation used for normalization

    mirror (int, optional, default = 1) - flag for the horizontal flip.

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    if isinstance(mirror, int):
        if (mirror == 0):
            mirror = b.createIntParameter(0)
        else:
            mirror = b.createIntParameter(1)

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "dest_width:": resize_width, "dest_height": resize_height, "mean": mean, "std_dev": std, "is_output": False,
                     "scaling_mode": scaling_mode, "max_size": max_size, "resize_shorter": resize_shorter, "resize_longer": resize_longer,
                     "interpolation_type": interpolation_type, "mirror": mirror, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    rmn = b.ResizeMirrorNormalize(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (rmn)


def random_crop(*inputs, crop_area_factor=[0.08, 1], crop_aspect_ratio=[0.75, 1.333333],
                crop_pox_x=0, crop_pox_y=0, num_attempts=20, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation

    crop_area_factor (list of floats, optional, default = [0.08, 1]) - area factor used for crop generation

    crop_aspect_ratio (list of floats, optional, default = [0.75, 1.333333]) - valid range of aspect ratio of the cropping windows

    crop_pox_x (int, optional, default = 0) - crop_x position used for crop generation

    crop_pox_y (int, optional, default = 0) - crop_y position used for crop generation

    num_attempts (int, optional, default = 20) - number of attempts to get a crop window that matches the area factor and aspect ratio conditions

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False,
                     "crop_area_factor": crop_area_factor, "crop_aspect_ratio": crop_aspect_ratio, "crop_pos_x": crop_pox_x, "crop_pos_y": crop_pox_y, "num_of_attempts": num_attempts, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    random_cropped_image = b.RandomCrop(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (random_cropped_image)


def rotate(*inputs, angle=None, dest_width=0, dest_height=0, interpolation_type=types.LINEAR_INTERPOLATION,
           rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation

    angle (float, optional, default = None) - angle used for rotating the image

    dest_width (int, optional, default = 0) - The length of the X dimension of the rotated image

    dest_height (int, optional, default = 0) - The length of the Y dimension of the rotated image

    interpolation_type (int, optional, default = types.LINEAR_INTERPOLATION) - Type of interpolation to be used.

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    angle = b.createFloatParameter(angle) if isinstance(angle, float) else angle
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False,
                     "angle": angle, "dest_width": dest_width, "dest_height": dest_height, "interpolation_type": interpolation_type, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    rotated_image = b.Rotate(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (rotated_image)


def saturation(*inputs, saturation=1.0, rocal_tensor_output_layout=types.NHWC,
               rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation
    
    saturation (float, default = 1.0) - The saturation change factor. Values must be non-negative. Example values: 0 - Completely desaturated image, 1 - No change to image's saturation.

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    saturation = b.createFloatParameter(saturation) if isinstance(saturation, float) else saturation
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0],
                     "is_output": False, "sat": saturation, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    saturated_image = b.Saturation(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (saturated_image)


def ssd_random_crop(*inputs, p_threshold=None, crop_area_factor=None, crop_aspect_ratio=None,
                    crop_pos_x=None, crop_pos_y=None, num_attempts=1, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation

    p_threshold (float, optional, default = None) - threshold value used for selecting bboxes during crop generation

    crop_area_factor (float, optional, default = None) - area factor used for crop generation
    
    crop_aspect_ratio (float, optional, default = None) - aspect ratio of the cropping windows

    crop_pox_x (float, optional, default = None) - crop_x position used for crop generation

    crop_pox_y (float, optional, default = None) - crop_y position used for crop generation

    num_attempts (int, optional, default = 1) - number of attempts to get a crop window that matches the area factor and aspect ratio conditions

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    if (num_attempts == 1):
        _num_attempts = 20
    else:
        _num_attempts = num_attempts
    p_threshold = b.createFloatParameter(p_threshold) if isinstance(p_threshold, float) else p_threshold
    crop_area_factor = b.createFloatParameter(crop_area_factor) if isinstance(crop_area_factor, float) else crop_area_factor
    crop_aspect_ratio = b.createFloatParameter(crop_aspect_ratio) if isinstance(crop_aspect_ratio, float) else crop_aspect_ratio
    crop_pos_x = b.createFloatParameter(crop_pos_x) if isinstance(crop_pos_x, float) else crop_pos_x
    crop_pos_y = b.createFloatParameter(crop_pos_y) if isinstance(crop_pos_y, float) else crop_pos_y

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "p_threshold": p_threshold, "crop_area_factor": crop_area_factor,
                     "crop_aspect_ratio": crop_aspect_ratio, "crop_pos_x": crop_pos_x, "crop_pos_y": crop_pos_y, "num_of_attempts": _num_attempts, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    ssd_random_cropped_image = b.SSDRandomCrop(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (ssd_random_cropped_image)


def warp_affine(*inputs, dest_width=0, dest_height=0, transform_matrix=[0, 0, 0, 0, 0, 0],
                interpolation_type=types.LINEAR_INTERPOLATION, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation

    dest_width (int, optional, default = 0) - The length of the X dimension of the transformed image

    transform_matrix (list of ints, optional, default = [0, 0, 0, 0, 0, 0]) - Transformation matrix used to produce a new image

    dest_height (int, optional, default = 0) - The length of the Y dimension of the transformed image

    interpolation_type (int, optional, default = types.LINEAR_INTERPOLATION) - Type of interpolation to be used.

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    x0, x1, y0, y1, o0, o1 = transform_matrix
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "x0": x0, "x1": x1, "y0": y0, "y1": y1, "o0": o0,
                     "o1": o1, "is_output": False, "dest_height": dest_height, "dest_width": dest_width, "interpolation_type": interpolation_type, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    warp_affine_output = b.WarpAffineFixed(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (warp_affine_output)


def vignette(*inputs, vignette=0.5, rocal_tensor_output_layout=types.NHWC,
             rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation
    
    vignette (float, default = 0.5) - vignette value used for the augmentation output

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    vignette = b.createFloatParameter(vignette) if isinstance(vignette, float) else vignette
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "sdev": vignette,
                     "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    vignette_outputcolor_temp_output = b.Vignette(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (vignette_outputcolor_temp_output)


def crop_mirror_normalize(*inputs, crop=[0, 0], crop_pos_x=0.5, crop_pos_y=0.5,
                          crop_w=0, crop_h=0, mean=[0.0], std=[1.0], mirror=1, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation

    crop (list of ints, optional, default = [0, 0]) - list containing the crop dimensions of the cropped image

    crop_pox_x (float, optional, default = 0.5) - crop_x position used for crop generation

    crop_pox_y (float, optional, default = 0.5) - crop_y position used for crop generation

    crop_w (int, optional, default = 0) - crop width
    
    crop_h (int, optional, default = 0) - crop height

    mean (list of floats, optional, default = [0.0]) - mean used for normalization

    std (list of floats, optional, default = [1.0]) - standard deviation used for normalization

    mirror (int, optional, default = 1) - flag for the horizontal flip.

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    if (len(crop) == 2):
        crop_height = crop[0]
        crop_width = crop[1]
    elif (len(crop) == 3):
        crop_height = crop[1]
        crop_width = crop[2]
    else:
        crop_height = crop_h
        crop_width = crop_w

    if isinstance(mirror, int):
        if (mirror == 0):
            mirror = b.createIntParameter(0)
        else:
            mirror = b.createIntParameter(1)

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "crop_height": crop_height, "crop_width": crop_width, "start_x": crop_pos_x, "start_y": crop_pos_y, "mean": mean, "std_dev": std,
                     "is_output": False, "mirror": mirror, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    cmn = b.CropMirrorNormalize(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (cmn)


def center_crop(*inputs, crop=[100, 100], crop_h=0, crop_w=0, crop_d=1,
                rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation

    crop (list of ints, optional, default = [0, 0]) - list containing the crop dimensions of the cropped image

    crop_h (int, optional, default = 0) - crop height

    crop_w (int, optional, default = 0) - crop width

    crop_d (int, optional, default = 0) - crop depth

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    if (len(crop) == 2):
        crop_depth = crop_d
        crop_height = crop[0]
        crop_width = crop[1]
    elif (len(crop) == 3):
        crop_depth = crop[0]
        crop_height = crop[1]
        crop_width = crop[2]
    else:
        crop_depth = crop_d
        crop_height = crop_h
        crop_width = crop_w

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "crop_width": crop_width, "crop_height": crop_height, "crop_depth": crop_depth,
                     "is_output": False, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    centre_cropped_image = b.CenterCropFixed(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))

    return (centre_cropped_image)


def crop(*inputs, crop=[0.0, 0.0], crop_pos_x=0.5, crop_pos_y=0.5, crop_pos_z=0.5,
         crop_w=0, crop_h=0, crop_d=1, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation

    crop (list of ints, optional, default = [0, 0]) - list containing the crop dimensions of the cropped image

    crop_pox_x (float, optional, default = 0.5) - crop_x position used for crop generation

    crop_pox_y (float, optional, default = 0.5) - crop_y position used for crop generation
    
    crop_pox_z (float, optional, default = 0.5) - crop_z position used for crop generation

    crop_w (int, optional, default = 0) - crop width
    
    crop_h (int, optional, default = 0) - crop height

    crop_d (int, optional, default = 1) - crop depth

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    if (len(crop) == 2):
        crop_depth = crop_d
        crop_height = crop[0]
        crop_width = crop[1]
    elif (len(crop) == 3):
        crop_depth = crop[0]
        crop_height = crop[1]
        crop_width = crop[2]
    else:
        crop_depth = crop_d
        crop_height = crop_h
        crop_width = crop_w

    if ((crop_width == 0) and (crop_height == 0)):
        # pybind call arguments
        kwargs_pybind = {"input_image": inputs[0], "crop_width": None, "crop_height": None, "crop_depth": None, "is_output": False, "crop_pos_x": None,
                         "crop_pos_y": None, "crop_pos_z": None, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
        cropped_image = b.Crop(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    else:
        # pybind call arguments
        kwargs_pybind = {"input_image": inputs[0], "crop_width": crop_width, "crop_height": crop_height, "crop_depth": crop_depth, "is_output": False, "crop_pos_x": crop_pos_x,
                         "crop_pos_y": crop_pos_y, "crop_pos_z": crop_pos_z, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
        cropped_image = b.CropFixed(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (cropped_image)


def color_twist(*inputs, brightness=1.0, contrast=1.0, hue=0.0,
                saturation=1.0, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation

    brightness (float, optional, default = 1.0) - brightness multiplier. Values >= 0 are accepted. For example: 0 - black image, 1 - no change, 2 - increase brightness twice

    contrast (float, optional, default = 1.0) - contrast multiplier used for the augmentation. Values >= 0 are accepted. For example: 0 - gray image, 1 - no change, 2 - increase contrast twice
    
    hue (float, optional, default = 0.0) - hue change in degrees

    saturation (float, optional, default = 1.0) - The saturation change factor. Values must be non-negative. Example values: 0 - Completely desaturated image, 1 - No change to image's saturation.

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    brightness = b.createFloatParameter(brightness) if isinstance(brightness, float) else brightness
    contrast = b.createFloatParameter(contrast) if isinstance(contrast, float) else contrast
    hue = b.createFloatParameter(hue) if isinstance(hue, float) else hue
    saturation = b.createFloatParameter(saturation) if isinstance(saturation, float) else saturation

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "p_alpha": brightness, "p_beta": contrast,
                     "p_hue": hue, "p_sat": saturation, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    color_twist_image = b.ColorTwist(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (color_twist_image)


def uniform(*inputs, rng_range=[-1, 1]):
    """
    inputs - the input image passed to the augmentation
 
    rng_range (list of ints, optional, default = [-1, 1]) - uniform distribution used for random number generation
    """
    output_param = b.createFloatUniformRand(rng_range[0], rng_range[1])
    return output_param


def random_bbox_crop(*inputs, all_boxes_above_threshold=True, allow_no_crop=True, aspect_ratio=None,
                     crop_shape=None, num_attempts=1, scaling=None, seed=1, total_num_attempts=0):
    """
    inputs - the input image passed to the augmentation

    all_boxes_above_threshold (bool, optional, default = True) - If set to True, all bounding boxes in a sample should overlap with the cropping window

    allow_no_crop (bool, optional, default = True) - If set to True, one of the possible outcomes of the random process will be to not crop

    aspect_ratio (list of floats, optional, default = None) - crop_y position used for crop generation
    
    crop_shape (list of ints, optional, default = None) - crop shape used for crop generation

    num_attempts (int, optional, default = 1) - Number of attempts to get a crop window that matches the aspect_ratio and threshold
    
    scaling (list of int, optional, default = None) - Range [min, max] for the crop size with respect to the original image dimensions.

    seed (int, optional, default = 1) - Random seed

    total_num_attempts (int, optional, default = 0) - If provided, it indicates the total maximum number of attempts to get a crop window that matches the aspect_ratio and the threshold.
        After total_num_attempts attempts, the best candidate will be selected. If this value is not specified, the crop search will continue indefinitely until a valid crop is found.
    """
    aspect_ratio = aspect_ratio if aspect_ratio else [1.0, 1.0]
    crop_shape = [] if crop_shape is None else crop_shape
    scaling = scaling if scaling else [1.0, 1.0]
    if (len(crop_shape) == 0):
        has_shape = False
        crop_width = 0
        crop_height = 0
    else:
        has_shape = True
        crop_width = crop_shape[0]
        crop_height = crop_shape[1]
    scaling = b.createFloatUniformRand(scaling[0], scaling[1])
    aspect_ratio = b.createFloatUniformRand(aspect_ratio[0], aspect_ratio[1])

    # pybind call arguments
    kwargs_pybind = {"all_boxes_above_threshold": all_boxes_above_threshold, "no_crop": allow_no_crop, "p_aspect_ratio": aspect_ratio, "has_shape": has_shape,
                     "crop_width": crop_width, "crop_height": crop_height, "num_attemps": num_attempts, "p_scaling": scaling, "total_num_attempts": total_num_attempts, "seed": seed}
    random_bbox_crop = b.randomBBoxCrop(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))

    return (random_bbox_crop, [], [], [])


def one_hot(*inputs, num_classes=0):
    """
    inputs - the input image passed to the augmentation

    num_classes (int, default = 0) - Number of classes used for one hot encoding
    """
    Pipeline._current_pipeline._num_classes = num_classes
    Pipeline._current_pipeline._one_hot_encoding = True
    return ([])


def box_encoder(*inputs, anchors, criteria=0.5, means=None,
                offset=False, scale=1.0, stds=None):
    """
    inputs - the input image passed to the augmentation

    anchors (list of floats) - Anchors to be used for encoding, as the list of floats is in the ltrb format.

    criteria (float, optional, default = 0.5) - Threshold IoU for matching bounding boxes with anchors. The value needs to be between 0 and 1.

    means (list of floats, optional, default = None) - [x y w h] mean values for normalization.
    
    offset (bool, optional, default = False) - Returns normalized offsets ((encoded_bboxes * scale - anchors * scale) - mean) / stds in Encoded bboxes that use std and the mean and scale arguments.

    scale (float, optional, default = 1.0) - Rescales the box and anchor values before the offset is calculated (for example, to return to the absolute values)
    
    stds (list of float, optional, default = None) - [x y w h] standard deviations for offset normalization.
    """
    means = means if means else [0.0, 0.0, 0.0, 0.0]
    stds = stds if stds else [1.0, 1.0, 1.0, 1.0]

    # pybind call arguments
    kwargs_pybind = {"anchors": anchors, "criteria": criteria,
                     "means": means, "stds": stds, "offset": offset, "scale": scale}
    box_encoder = b.boxEncoder(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    Pipeline._current_pipeline._BoxEncoder = True
    return (box_encoder, [])


def color_temp(*inputs, adjustment_value=50, rocal_tensor_output_layout=types.NHWC,
               rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation
    
    adjustment_value (int, default = 50) - value for adjusting the color temperature

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    adjustment_value = b.createIntParameter(adjustment_value) if isinstance(adjustment_value, int) else adjustment_value
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "adjustment_value": adjustment_value,
                     "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    color_temp_output = b.ColorTemp(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (color_temp_output)


def nop(*inputs):
    """
    inputs - the input image passed to the augmentation
    """
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False}
    nop_output = b.rocalNop(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (nop_output)


def copy(*inputs):
    """
    inputs - the input image passed to the augmentation
    """
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False}
    copied_image = b.rocalCopy(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (copied_image)


def snp_noise(*inputs, p_noise=0.0, p_salt=0.0, noise_val=0.0, salt_val=0.0,
              seed=0, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    """
    inputs - the input image passed to the augmentation

    p_noise (float, default = 0.0) - noise probability

    p_salt (float, default = 0.0) - salt probability

    noise_val (float, default = 0.0) - noise value to be added to the image

    salt_val (float, default = 0.0) - salt value to be added to the image

    seed (int, optional, default = 0) - Random seed

    rocal_tensor_output_layout (int, optional, default = types.NHWC) - tensor layout for the augmentation output

    rocal_tensor_output_datatype (int, optional, default = types.UINT8) - tensor dtype for the augmentation output
    """
    p_noise = b.createFloatParameter(p_noise) if isinstance(p_noise, float) else p_noise
    p_salt = b.createFloatParameter(p_salt) if isinstance(p_salt, float) else p_salt
    noise_val = b.createFloatParameter(noise_val) if isinstance(noise_val, float) else noise_val
    salt_val = b.createFloatParameter(salt_val) if isinstance(salt_val, float) else salt_val

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "p_noise": p_noise, "p_salt": p_salt, "noise_val": noise_val,
                     "salt_val": salt_val, "seed": seed, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    snp_noise_added_image = b.SnPNoise(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (snp_noise_added_image)


def box_iou_matcher(*inputs, anchors, criteria=0.5, high_threshold=0.5,
                    low_threshold=0.4, allow_low_quality_matches=True):
    """
    inputs - the input image passed to the augmentation

    anchors (list of floats) - Anchors to be used for encoding, as the list of floats is in the ltrb format.

    criteria (float, optional, default = 0.5) - criteria value used for box iou matcher

    high_threshold (float, optional, default = 0.5) - upper threshold used for matching indices
    
    low_threshold (float, optional, default = 0.4) - lower threshold used for matching indices

    allow_low_quality_matches (bool, optional, default = True) - Whether to allow low quality matches as output
    """
    # pybind call arguments
    kwargs_pybind = {"anchors": anchors, "criteria": criteria, "high_threshold": high_threshold,
                     "low_threshold": low_threshold, "allow_low_quality_matches": allow_low_quality_matches}
    box_iou_matcher = b.BoxIOUMatcher(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    Pipeline._current_pipeline._BoxIOUMatcher = True
    return (box_iou_matcher, [])

def nonsilent_region(*inputs, rocal_tensor_output_type = types.FLOAT, bytes_per_sample_hint = [0], cutoff_db = -60, reference_power = 0.0, reset_interval = 8192, seed = -1, window_length = 2048):
    """
    Performs leading and trailing silence detection in an audio buffer.

    The operator returns the beginning and length of the non-silent region by comparing the short term power calculated for window_length of the signal with a silence cut-off threshold. The signal is considered to be silent when the short_term_power_db is less than the cutoff_db. where:

    short_term_power_db = 10 * log10( short_term_power / reference_power )

    Unless specified otherwise, reference_power is the maximum power of the signal.
    """
    kwargs_pybind = {"input_audio0": inputs[0], "is_output": False, "cutoff_db": cutoff_db,
                     "reference_power": reference_power, "reset_interval": reset_interval, "window_length": window_length}
    non_silent_region_output = b.NonSilentRegion(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return non_silent_region_output

def slice(*inputs, anchor = [], shape = [], axes = [1, 0], axis_names = "WH", bytes_per_sample_hint = [0], dtype = types.FLOAT, end = [], fill_values = [0.0], normalized_anchor = True, normalized_shape = True,  out_of_bounds_policy = types.ERROR, rel_end = [], rel_shape = [], rel_start = [], seed = -1, start = [] , rocal_tensor_output_type = types.FLOAT):
    """
    The slice can be specified by proving the start and end coordinates, or start coordinates and shape of the slice. Both coordinates and shapes can be provided in absolute or relative terms.

    The slice arguments can be specified by the following named arguments:

    start: Slice start coordinates (absolute)

    rel_start: Slice start coordinates (relative)

    end: Slice end coordinates (absolute)

    rel_end: Slice end coordinates (relative)

    shape: Slice shape (absolute)

    rel_shape: Slice shape (relative)

    """

    kwargs_pybind = {"input_audio0": inputs[0], "is_output": False, "anchor": anchor[0], "shape": shape[0], "fill_values": fill_values, "axes": axes,
                     "normalized_anchor": normalized_anchor , "normalized_shape": normalized_shape, "out_of_bounds_policy": out_of_bounds_policy, "rocal_tensor_output_type": rocal_tensor_output_type}
    slice_output = b.audioSlice(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return slice_output

def spectrogram(*inputs, bytes_per_sample_hint = [0], center_windows = True, layout = types.FT, nfft = None, power = 2, reflect_padding = True, seed = -1, window_fn = [], window_length = 512, window_step = 256, rocal_tensor_layout = None, rocal_tensor_output_type = types.FLOAT) :
    '''
    Produces a spectrogram from a 1D signal (for example, audio).

    Input data is expected to be one channel (shape being (nsamples,), (nsamples, 1), or (1, nsamples)) of type float32.
    '''
    kwargs_pybind = {"input_audio0": inputs[0], "is_output": False, "window_fn": window_fn, "center_windows": center_windows, "reflect_padding": reflect_padding,
                     "layout": layout, "power": power, "nfft": nfft, "window_length": window_length, "window_step": window_step, "rocal_tensor_output_type": rocal_tensor_output_type}
    spectrogram_output = b.Spectrogram(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return spectrogram_output

def mel_filter_bank(*inputs, bytes_per_sample_hint = [0], freq_high = 0.0, freq_low = 0.0, mel_formula = types.SLANEY, nfilter = 128, normalize = True, sample_rate = 44100.0, seed = -1, rocal_tensor_output_type = types.FLOAT):
    '''
    Converts a spectrogram to a mel spectrogram by applying a bank of triangular filters.

    The frequency (‘f’) dimension is selected from the input layout. In case of no layout, “f”, “ft”, or “*ft” is assumed, depending on the number of dimensions.
    '''
    kwargs_pybind = {"input_audio0": inputs[0], "is_output": False, "freq_high": freq_high, "freq_low": freq_low, "mel_formula": mel_formula,
                     "nfilter": nfilter, "normalize": normalize, "sample_rate": sample_rate, "rocal_tensor_output_type": rocal_tensor_output_type}
    mel_filter_bank_output = b.MelFilterBank(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return mel_filter_bank_output