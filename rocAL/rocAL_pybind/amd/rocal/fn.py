
from amd.rocal import readers
from amd.rocal import decoders
from amd.rocal import random
# from amd.rocal import noise
# from amd.rocal import reductions

import amd.rocal.types as types
import rocal_pybind as b
from amd.rocal.pipeline import Pipeline


def brightness(*inputs, brightness=1.0, bytes_per_sample_hint=0, image_type=0,
               preserve=False, seed=-1, device=None):
    kwargs_pybind = {"input_image0": inputs[0], "is_output": False, "alpha": None, "beta": None}
    brightness_image = b.Brightness(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (brightness_image)

def brightness_fixed(*inputs, alpha=None, beta=None, seed=-1, device=None):
    alpha = b.CreateFloatParameter(alpha) if isinstance(alpha, float) else alpha
    beta = b.CreateFloatParameter(beta) if isinstance(beta, float) else beta
    kwargs_pybind = {"input_image0": inputs[0], "is_output": False, "alpha": alpha, "beta": beta}
    brightness_image = b.Brightness(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (brightness_image)

def resize(*inputs, bytes_per_sample_hint=0, image_type=0, interp_type=1, mag_filter= 1, max_size = [], min_filter = 1,
            minibatch_size=32, preserve=False, resize_longer=0, resize_shorter= 0, resize_depth = 0, resize_width = 0, resize_height = 0,  scaling_mode=types.SCALING_MODE_DEFAULT, interpolation_type=types.LINEAR_INTERPOLATION,
            save_attrs=False, seed=1, rocal_tensor_layout=types.NCHW, rocal_tensor_output_type=types.FLOAT, temp_buffer_hint=0, device = None):
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0], "rocal_tensor_layout" : rocal_tensor_layout, "rocal_tensor_output_type" : rocal_tensor_output_type,  "dest_width:" : resize_width , "dest_height": resize_height, "is_output": False, "scaling_mode": scaling_mode, "max_size": max_size, "resize_shorter": resize_shorter,
                     "resize_longer": resize_longer, "interpolation_type": interpolation_type}
    resized_image = b.Resize(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (resized_image)

def crop_mirror_normalize(*inputs, bytes_per_sample_hint=0, crop=[0, 0], crop_d=0, crop_h=0, crop_pos_x=0.5, crop_pos_y=0.5, crop_pos_z=0.5,
                          crop_w=0, image_type=0, mean=[0.0], mirror=1, output_dtype=types.FLOAT, rocal_tensor_layout =types.NCHW, rocal_tensor_output_type = types.FLOAT,output_layout=types.NCHW, pad_output=False,
                          preserve=False, seed=1, std=[1.0], device=None):

    if(len(crop) == 2):
        crop_depth = crop_d
        crop_height = crop[0]
        crop_width = crop[1]
    elif(len(crop) == 3):
        crop_depth = crop[0]
        crop_height = crop[1]
        crop_width = crop[2]
    else:
        crop_depth = crop_d
        crop_height = crop_h
        crop_width = crop_w
    #Set Seed
    b.setSeed(seed)

    if isinstance(mirror,int):
        if(mirror == 0):
            mirror = b.CreateIntParameter(0)
        else:
            mirror = b.CreateIntParameter(1)

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0],"rocal_tensor_layout" : rocal_tensor_layout, "rocal_tensor_output_type" : rocal_tensor_output_type, "crop_depth":crop_depth, "crop_height":crop_height, "crop_width":crop_width, "start_x":1, "start_y":1, "start_z":1, "mean":mean, "std_dev":std,
                     "is_output": False, "mirror": mirror}
    b.setSeed(seed)
    cmn = b.CropMirrorNormalize(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    Pipeline._current_pipeline._tensor_layout = output_layout
    Pipeline._current_pipeline._tensor_dtype = output_dtype
    Pipeline._current_pipeline._multiplier = list(map(lambda x: 1/x ,std))
    Pipeline._current_pipeline._offset = list(map(lambda x,y: -(x/y), mean, std))
    return (cmn)

def resize_shorter(*inputs, resize_size=0, rocal_tensor_layout=types.NHWC, rocal_tensor_output_type=types.UINT8):
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0], "rocal_tensor_layout": rocal_tensor_layout, "rocal_tensor_output_type" :rocal_tensor_output_type, "resize_size": resize_size,
                     "is_output": False}
    resized_image = b.ResizeShorter(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (resized_image)

def centre_crop(*inputs, bytes_per_sample_hint=0, crop=[100, 100], crop_d=1, crop_h= 0, crop_pos_x = 0.5, crop_pos_y = 0.5, crop_pos_z = 0.5,
                 crop_w=0, image_type=0, output_dtype=types.FLOAT, preserve = False, seed = 1,rocal_tensor_layout=types.NHWC, rocal_tensor_output_type=types.UINT8,  device = None):

    if(len(crop) == 2):
        crop_depth = crop_d
        crop_height = crop[0]
        crop_width = crop[1]
    elif(len(crop) == 3):
        crop_depth = crop[0]
        crop_height = crop[1]
        crop_width = crop[2]
    else:
        crop_depth = crop_d
        crop_height = crop_h
        crop_width = crop_w
    #Set Seed
    b.setSeed(seed)
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0], "rocal_tensor_layout": rocal_tensor_layout, "rocal_tensor_output_type" :rocal_tensor_output_type, "crop_width":crop_width, "crop_height":crop_height, "crop_depth":crop_depth, "is_output": False}
    centre_cropped_image = b.CenterCropFixed(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (centre_cropped_image)

def random_bbox_crop(*inputs,all_boxes_above_threshold = True, allow_no_crop =True, aspect_ratio = None, bbox_layout = "", bytes_per_sample_hint = 0,
                crop_shape = None, input_shape = None, ltrb = True, num_attempts = 1 ,scaling =  None,  preserve = False, seed = 1, shape_layout = "",
                threshold_type ="iou", thresholds = None, total_num_attempts = 0, device = None, labels = None ):
    aspect_ratio = aspect_ratio if aspect_ratio else [1.0, 1.0]
    crop_shape = [] if crop_shape is None else crop_shape
    scaling = scaling if scaling else [1.0, 1.0]
    if(len(crop_shape) == 0):
        has_shape = False
        crop_width = 0
        crop_height = 0
    else:
        has_shape = True
        crop_width = crop_shape[0]
        crop_height = crop_shape[1]
    scaling = b.CreateFloatUniformRand(scaling[0], scaling[1])
    aspect_ratio = b.CreateFloatUniformRand(aspect_ratio[0], aspect_ratio[1])

    # pybind call arguments
    kwargs_pybind = {"all_boxes_above_threshold":all_boxes_above_threshold, "no_crop": allow_no_crop, "p_aspect_ratio":aspect_ratio, "has_shape":has_shape, "crop_width":crop_width, "crop_height":crop_height, "num_attemps":num_attempts, "p_scaling":scaling, "total_num_attempts":total_num_attempts, "seed":seed }
    random_bbox_crop = b.RandomBBoxCrop(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))

    return (random_bbox_crop,[],[],[])

def uniform(*inputs,rng_range=[-1, 1], device=None):
    output_param = b.CreateFloatUniformRand(rng_range[0], rng_range[1])
    return output_param

def color_twist(*inputs, brightness=1.0, bytes_per_sample_hint=0, contrast=1.0, hue=0.0, image_type=0,
                preserve=False, saturation=1.0, seed=-1,rocal_tensor_layout=types.NHWC, rocal_tensor_output_type=types.UINT8, device=None):
    brightness = b.CreateFloatParameter(brightness) if isinstance(
        brightness, float) else brightness
    contrast = b.CreateFloatParameter(
        contrast) if isinstance(contrast, float) else contrast
    hue = b.CreateFloatParameter(hue) if isinstance(hue, float) else hue
    saturation = b.CreateFloatParameter(saturation) if isinstance(
        saturation, float) else saturation
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0],  "rocal_tensor_layout": rocal_tensor_layout, "rocal_tensor_output_type" :rocal_tensor_output_type, "is_output": False,
                     "alpha": brightness, "beta": contrast, "hue": hue, "sat": saturation}
    color_twist_image = b.ColorTwist(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (color_twist_image)

def box_encoder(*inputs, anchors, bytes_per_sample_hint=0, criteria=0.5, means=None, offset=False, preserve=False, scale=1.0, seed=-1, stds=None ,device = None):
    means = means if means else [0.0, 0.0, 0.0, 0.0]
    stds = stds if stds else [1.0, 1.0, 1.0, 1.0]
    kwargs_pybind ={"anchors":anchors, "criteria":criteria, "means":means, "stds":stds, "offset":offset, "scale":scale}
    box_encoder = b.BoxEncoder(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    Pipeline._current_pipeline._BoxEncoder = True
    return (box_encoder , [])

def flip(*inputs, h_flip=0, v_flip=0, preserve=False, seed=-1, bytes_per_sample_hint=0, rocal_tensor_layout=types.NHWC, rocal_tensor_output_type=types.UINT8, device=None):
    if isinstance(h_flip, int):
        if(h_flip == 0):
            h_flip = b.CreateIntParameter(0)
        else:
            h_flip = b.CreateIntParameter(1)

    if isinstance(v_flip, int):
        if(v_flip == 0):
            v_flip = b.CreateIntParameter(0)
        else:
            v_flip = b.CreateIntParameter(1)

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0],"rocal_tensor_layout" : rocal_tensor_layout, "rocal_tensor_output_type" : rocal_tensor_output_type,
                     "is_output": False, "h_flip": h_flip, "v_flip": v_flip}

    flip_image = b.Flip(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (flip_image)

def warp_affine(*inputs, fill_value=0.0, interp_type = 1, matrix = None, size = None, is_train = False, rotate_probability = 0, half_body_probability = 0, rotation_factor = 0, scale_factor = 0, preserve=False, seed=-1, bytes_per_sample_hint=0, rocal_tensor_layout=types.NHWC, rocal_tensor_output_type=types.UINT8, device=None):
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0],"rocal_tensor_layout" : rocal_tensor_layout, "rocal_tensor_output_type" : rocal_tensor_output_type,
                     "is_output": False, "is_train": is_train, "dest_height": size[0], "dest_width":size[1], "rotate_probability": rotate_probability, "half_body_probability": half_body_probability, "rotation_factor": rotation_factor, "scale_factor": scale_factor,
                     "x0": None, "x1": None, "y0": None, "y1": None, "o0": None, "o1": None, "interpolation_type": 1}

    warp_affine_image = b.WarpAffine(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (warp_affine_image)
