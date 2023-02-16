import amd.rocal.types as types
import rocal_pybind as b
from amd.rocal.pipeline import Pipeline

def image(*inputs, user_feature_key_map = None, path='', file_root ='', annotations_file= '', shard_id = 0, num_shards = 1, random_shuffle = False, affine=True, bytes_per_sample_hint=0, cache_batch_copy= True, cache_debug = False, cache_size = 0, cache_threshold = 0,
                 cache_type='', device_memory_padding=16777216, host_memory_padding=8388608, hybrid_huffman_threshold= 1000000, output_type = types.RGB,
                 preserve=False, seed=-1, split_stages=False, use_chunk_allocator= False, use_fast_idct = False, device = None):
    reader = Pipeline._current_pipeline._reader

    if( reader == 'COCOReader'):
        kwargs_pybind = {
            "source_path": file_root,
            "json_path": annotations_file,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": types.USER_GIVEN_SIZE_ORIG,
            "max_width": 1024,
            "max_height":1024}
        decoded_image = b.COCO_ImageDecoderShard(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))

    else:
        kwargs_pybind = {
            "source_path": file_root,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": types.USER_GIVEN_SIZE,
            "max_width": 2000,
            "max_height":2000,
            "dec_type":types.DECODER_TJPEG
            }
        decoded_image = b.ImageDecoderShard(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))

    return (decoded_image)


def image_random_crop(*inputs, user_feature_key_map=None, path='', file_root='', annotations_file='', num_shards=1, shard_id=0, random_shuffle=False, affine=True, bytes_per_sample_hint=0, device_memory_padding=16777216, host_memory_padding=8388608, hybrid_huffman_threshold=1000000,
                      output_type=types.RGB, preserve=False, seed=1, split_stages=False, use_chunk_allocator=False, use_fast_idct=False, random_aspect_ratio=[0.8, 1.25], random_area=[0.08, 1.0], num_attempts=100, device=None):

    reader = Pipeline._current_pipeline._reader
    # Internally calls the C++ Partial decoder's
    b.setSeed(seed)
    if( reader == 'COCOReader'):

        kwargs_pybind = {
            "source_path": file_root,
            "json_path": annotations_file,
            "color_format": output_type,
            "shard_id": shard_id,
            "shard_count": num_shards,
            'is_output': False,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": types.USER_GIVEN_SIZE,
            "max_width": 640, #TODO: what happens when we give user given size = multiplier * max_decoded_width
            "max_height":640, #TODO: what happens when we give user given size = multiplier * max_decoded_width
            "area_factor": None,
            "aspect_ratio": None,
            "x_drift_factor": None,
            "y_drift_factor": None}
        image_decoder_slice = b.COCO_ImageDecoderSliceShard(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    elif reader == "labelReader":
        kwargs_pybind = {
            "source_path": file_root,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "area_factor": random_area,
            "aspect_ratio": random_aspect_ratio,
            "num_attempts": num_attempts,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": types.USER_GIVEN_SIZE,
            "max_width": 1000,
            "max_height":1000}
        crop_output_image = b.FusedDecoderCropShard(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))

    return (crop_output_image)


def image_slice(*inputs,file_root='',path='',annotations_file='',shard_id = 0, num_shards = 1, random_shuffle = False, affine = True, axes = None, axis_names = "WH",bytes_per_sample_hint = 0, device_memory_padding = 16777216,
                device_memory_padding_jpeg2k = 0, host_memory_padding = 8388608,
                host_memory_padding_jpeg2k = 0, hybrid_huffman_threshold = 1000000,
                 memory_stats = False, normalized_anchor = True, normalized_shape = True, output_type = types.RGB,
                preserve = False, seed = 1, split_stages = False, use_chunk_allocator = False, use_fast_idct = False, random_aspect_ratio=[0.8, 1.25], random_area=[0.08, 1.0], num_attempts=100, device = None):

    reader = Pipeline._current_pipeline._reader
    b.setSeed(seed)
    #Reader -> Randon BBox Crop -> ImageDecoderSlice
    if( reader == 'COCOReader'):

        kwargs_pybind = {
            "source_path": file_root,
            "json_path": annotations_file,
            "color_format": output_type,
            "shard_id": shard_id,
            "shard_count": num_shards,
            'is_output': False,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": types.USER_GIVEN_SIZE,
            "max_width": 640, #TODO: what happens when we give user given size = multiplier * max_decoded_width
            "max_height":640, #TODO: what happens when we give user given size = multiplier * max_decoded_width
            "area_factor": None,
            "aspect_ratio": None,
            "x_drift_factor": None,
            "y_drift_factor": None}
        image_decoder_slice = b.COCO_ImageDecoderSliceShard(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))

    elif reader == "labelReader":
        kwargs_pybind = {
            "source_path": file_root,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "area_factor": random_area,
            "aspect_ratio": random_aspect_ratio,
            "num_attempts": num_attempts,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": types.USER_GIVEN_SIZE,
            "max_width": 1000,
            "max_height":1000}
        image_decoder_slice = b.FusedDecoderCropShard(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (image_decoder_slice)


