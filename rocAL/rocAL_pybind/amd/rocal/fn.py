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
    ratio = b.createFloatParameter(ratio) if isinstance(ratio, float) else ratio
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0], "input_image1": inputs[1], "is_output": False, "ratio": ratio,
                     "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    blend_image = b.Blend(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (blend_image)


def snow(*inputs, snow=0.5, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    snow = b.createFloatParameter(snow) if isinstance(snow, float) else snow
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "snow": snow,
                     "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    snow_image = b.Snow(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (snow_image)


def exposure(*inputs, exposure=0.5, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    exposure = b.createFloatParameter(exposure) if isinstance(exposure, float) else exposure
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "exposure": exposure,
                     "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    exposure_image = b.Exposure(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (exposure_image)


def fish_eye(*inputs, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False,
                     "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    fisheye_image = b.FishEye(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (fisheye_image)


def fog(*inputs, fog=0.5, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    fog = b.createFloatParameter(fog) if isinstance(fog, float) else fog
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0],
                     "is_output": False, "fog_value": fog, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    fog_image = b.Fog(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (fog_image)


def brightness(*inputs, alpha=None, beta=None,
               rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    alpha = b.createFloatParameter(alpha) if isinstance(alpha, float) else alpha
    beta = b.createFloatParameter(beta) if isinstance(beta, float) else beta

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "alpha": alpha, "beta": beta,
                     "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    brightness_image = b.Brightness(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (brightness_image)


def brightness_fixed(*inputs, alpha=1.0, beta=0.0,
                     rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "alpha": alpha, "beta": beta,
                     "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    brightness_image = b.BrightnessFixed(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (brightness_image)


def lens_correction(*inputs, strength=None, zoom=None,
                    rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    strength = b.createFloatParameter(strength) if isinstance(strength, float) else strength
    zoom = b.createFloatParameter(zoom) if isinstance(zoom, float) else zoom

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "strength": strength, "zoom": zoom,
                     "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    len_corrected_image = b.LensCorrection(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (len_corrected_image)


def blur(*inputs, s_dev=None, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    s_dev = b.createIntParameter(s_dev) if isinstance(s_dev, int) else s_dev
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "s_dev": s_dev,
                     "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    blur_image = b.Blur(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (blur_image)


def contrast(*inputs, contrast_factor=None, contrast_center=None,
             rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    contrast_factor = b.createFloatParameter(contrast_factor) if isinstance(contrast_factor, float) else contrast_factor
    contrast_center = b.createFloatParameter(contrast_center) if isinstance(contrast_center, float) else contrast_center

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0],
                     "is_output": False, "contrast_factor": contrast_factor, "contrast_center": contrast_center, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    contrast_image = b.Contrast(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (contrast_image)


def flip(*inputs, h_flip=0, v_flip=0,
         rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    h_flip = b.createIntParameter(h_flip) if isinstance(h_flip, int) else h_flip
    v_flip = b.createIntParameter(v_flip) if isinstance(v_flip, int) else v_flip

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0],
                     "is_output": False, "h_flip": h_flip, "v_flip": v_flip, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    flip_image = b.Flip(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (flip_image)


def gamma_correction(*inputs, gamma=0.5, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    gamma = b.createFloatParameter(gamma) if isinstance(gamma, float) else gamma
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0],
                     "is_output": False, "gamma": gamma, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    gamma_correction_image = b.GammaCorrection(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (gamma_correction_image)


def hue(*inputs, hue=None, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    hue = b.createFloatParameter(hue) if isinstance(hue, float) else hue
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0],
                     "is_output": False, "hue": hue, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    hue_image = b.Hue(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (hue_image)


def jitter(*inputs, kernel_size=None, seed=0, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    kernel_size = b.createIntParameter(kernel_size) if isinstance(kernel_size, int) else kernel_size
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0],
                     "is_output": False, "kernel_size": kernel_size, "seed": seed, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    jitter_image = b.Jitter(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (jitter_image)


def pixelate(*inputs, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0],
                     "is_output": False, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    pixelate_image = b.Pixelate(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (pixelate_image)


def rain(*inputs, rain=None, rain_width=None, rain_height=None, rain_transparency=None,
         rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
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
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "dest_width:": resize_width, "dest_height": resize_height, "is_output": False, "scaling_mode": scaling_mode, "max_size": max_size, "resize_shorter": resize_shorter,
                     "resize_longer": resize_longer, "interpolation_type": interpolation_type, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    resized_image = b.Resize(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (resized_image)


def resize_crop_mirror(*inputs, resize_width=0, resize_height=0, crop_w=0, crop_h=0, mirror=1,
                       rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
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
    crop_area_factor = b.createFloatParameter(crop_area_factor) if isinstance(crop_area_factor, float) else crop_area_factor
    crop_aspect_ratio = b.createFloatParameter(crop_aspect_ratio) if isinstance(crop_aspect_ratio, float) else crop_aspect_ratio
    x_drift = b.createFloatParameter(x_drift) if isinstance(x_drift, float) else x_drift
    y_drift = b.createFloatParameter(y_drift) if isinstance(y_drift, float) else y_drift

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "dest_width:": resize_width, "dest_height": resize_height, "is_output": False, "crop_area_factor": crop_area_factor,
                     "crop_aspect_ratio": crop_aspect_ratio, "x_drift": x_drift, "y_drift": y_drift, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    rcm = b.CropResize(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (rcm)


def resize_mirror_normalize(*inputs, max_size=[], resize_longer=0, resize_shorter=0, resize_x=0, resize_y=0, scaling_mode=types.SCALING_MODE_DEFAULT,
                            interpolation_type=types.LINEAR_INTERPOLATION, mean=[0.0], std=[1.0], mirror=1, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    if isinstance(mirror, int):
        if (mirror == 0):
            mirror = b.createIntParameter(0)
        else:
            mirror = b.createIntParameter(1)

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "dest_width:": resize_x, "dest_height": resize_y, "mean": mean, "std_dev": std, "is_output": False,
                     "scaling_mode": scaling_mode, "max_size": max_size, "resize_shorter": resize_shorter, "resize_longer": resize_longer,
                     "interpolation_type": interpolation_type, "mirror": mirror, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    rmn = b.ResizeMirrorNormalize(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (rmn)


def random_crop(*inputs, crop_area_factor=[0.08, 1], crop_aspect_ratio=[0.75, 1.333333],
                crop_pox_x=0, crop_pox_y=0, num_attempts=20, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False,
                     "crop_area_factor": crop_area_factor, "crop_aspect_ratio": crop_aspect_ratio, "crop_pos_x": crop_pox_x, "crop_pos_y": crop_pox_y, "num_of_attempts": num_attempts, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    random_cropped_image = b.RandomCrop(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (random_cropped_image)


def rotate(*inputs, angle=None, dest_width=0, dest_height=0, interpolation_type=types.LINEAR_INTERPOLATION,
           rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    angle = b.createFloatParameter(angle) if isinstance(angle, float) else angle
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False,
                     "angle": angle, "dest_width": dest_width, "dest_height": dest_height, "interpolation_type": interpolation_type, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    rotated_image = b.Rotate(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (rotated_image)


def saturation(*inputs, saturation=1.0, rocal_tensor_output_layout=types.NHWC,
               rocal_tensor_output_datatype=types.UINT8):
    saturation = b.createFloatParameter(saturation) if isinstance(saturation, float) else saturation
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0],
                     "is_output": False, "sat": saturation, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    saturated_image = b.Saturation(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (saturated_image)


def ssd_random_crop(*inputs, p_threshold=None, crop_area_factor=None,
                    crop_aspect_ratio=None, crop_pos_x=None, crop_pos_y=None, num_attempts=1):
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
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "p_threshold": p_threshold,
                     "crop_area_factor": crop_area_factor, "crop_aspect_ratio": crop_aspect_ratio, "crop_pos_x": crop_pos_x, "crop_pos_y": crop_pos_y, "num_of_attempts": _num_attempts}
    ssd_random_cropped_image = b.SSDRandomCrop(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (ssd_random_cropped_image)


def warp_affine(*inputs, dest_width=0, dest_height=0, x0=0, x1=0, y0=0, y1=0,
                o0=0, o1=0, interpolation_type=types.LINEAR_INTERPOLATION, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "x0": x0, "x1": x1, "y0": y0, "y1": y1, "o0": o0,
                     "o1": o1, "is_output": False, "dest_height": dest_height, "dest_width": dest_width, "interpolation_type": interpolation_type, "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    warp_affine_output = b.WarpAffineFixed(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (warp_affine_output)


def vignette(*inputs, vignette=0.5, rocal_tensor_output_layout=types.NHWC,
             rocal_tensor_output_datatype=types.UINT8):
    vignette = b.createFloatParameter(vignette) if isinstance(vignette, float) else vignette
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "sdev": vignette,
                     "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    vignette_outputcolor_temp_output = b.Vignette(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (vignette_outputcolor_temp_output)


def crop_mirror_normalize(*inputs, crop=[0, 0], crop_pos_x=0.5, crop_pos_y=0.5,
                          crop_w=0, crop_h=0, mean=[0.0], std=[1.0], mirror=1, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.FLOAT):
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
    output_param = b.createFloatUniformRand(rng_range[0], rng_range[1])
    return output_param


def random_bbox_crop(*inputs, all_boxes_above_threshold=True, allow_no_crop=True, aspect_ratio=None,
                     crop_shape=None, num_attempts=1, scaling=None, seed=1, total_num_attempts=0):
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
    Pipeline._current_pipeline._num_classes = num_classes
    Pipeline._current_pipeline._one_hot_encoding = True
    return ([])


def box_encoder(*inputs, anchors, criteria=0.5, means=None,
                offset=False, scale=1.0, stds=None):
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
    adjustment_value = b.createIntParameter(adjustment_value) if isinstance(adjustment_value, int) else adjustment_value
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "adjustment_value": adjustment_value,
                     "rocal_tensor_output_layout": rocal_tensor_output_layout, "rocal_tensor_output_datatype": rocal_tensor_output_datatype}
    color_temp_output = b.ColorTemp(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (color_temp_output)


def nop(*inputs):
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False}
    nop_output = b.rocalNop(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (nop_output)


def copy(*inputs):
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False}
    copied_image = b.rocalCopy(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (copied_image)


def snp_noise(*inputs, p_noise=0.0, p_salt=0.0, noise_val=0.0, salt_val=0.0,
              seed=0, rocal_tensor_output_layout=types.NHWC, rocal_tensor_output_datatype=types.UINT8):
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

def to_decibels(*inputs, bytes_per_sample_hint = [0], cutoff_db = -200.0, multiplier = 10.0, reference = 0.0, seed = -1, rocal_tensor_output_type = types.FLOAT):
    '''
    Converts a magnitude (real, positive) to the decibel scale.

    Conversion is done according to the following formula:

    min_ratio = pow(10, cutoff_db / multiplier)
    out[i] = multiplier * log10( max(min_ratio, input[i] / reference) )
    '''
    kwargs_pybind = {"input_audio0": inputs[0], "is_output": False, "cutoff_DB": cutoff_db, "multiplier": multiplier, "reference_magnitude": reference, "rocal_tensor_output_type": rocal_tensor_output_type}
    decibel_scale = b.ToDecibels(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return decibel_scale

def normalize(*inputs, axes = [], axis_names = "", batch = False, bytes_per_sample_hint = [0], ddof = 0, epsilon = 0.0, mean = 0.0, scale = 1.0, seed = -1, shift = 0.0, stddev = 0.0, rocal_tensor_output_type=types.FLOAT):
    '''
    Normalizes the input by removing the mean and dividing by the standard deviation.

    The mean and standard deviation can be calculated internally for the specified subset of axes or can be externally provided as the mean and stddev arguments.

    The normalization is done following the formula:

    out = scale * (in - mean) / stddev + shift

    The formula assumes that out and in are equally shaped tensors, but mean and stddev might be either tensors of same shape, scalars, or a mix of these.
    '''
    kwargs_pybind = {"input_audio0": inputs[0], "is_output": False, "batch": batch, "axes": axes, "mean": mean, "stddev": stddev,
                     "scale": scale , "shift": shift, "ddof": ddof , "epsilon": epsilon, "rocal_tensor_output_type": rocal_tensor_output_type}
    normalize_output = b.audioNormalize(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return normalize_output