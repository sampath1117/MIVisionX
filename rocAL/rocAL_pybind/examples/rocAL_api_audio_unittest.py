
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
from amd.rocal.plugin.pytorch import ROCALAudioIterator
import torch
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import sys
import matplotlib.pyplot as plt
import os
import numpy as np

def compare_output(output, roi, augmentation_name, idx):
    reference_path = os.getcwd() + "/REFERENCE_OUTPUTS_AUDIO/" + augmentation_name + "/"
    reference_file = reference_path + augmentation_name + "_ref_" + str(idx) + ".txt"
    fn = open(reference_file, "r")
    buffer_size = roi[0] * roi[1] 
    match_cnt = 0
    for row in range(roi[1]):
        for col in range(roi[0]):
            ref_data = float((fn.readline()).strip())
            out_data = output[row][col]
            if abs(out_data - ref_data) < 1e-20:
                match_cnt += 1
    is_match = (match_cnt == buffer_size)
    return is_match
                 
def draw_patches(audio, label, device):
    audio_data = audio.flatten()
    # Saving the array in a text file
    file = open("results/rocal_data_new"+str(label)+".txt", "w+")
    content = str(audio_data)
    file.write(content)
    file.close()
    plt.plot(audio_data)
    plt.savefig("results/rocal_data_new"+str(label)+".png")
    plt.close()

def main():
    if  len(sys.argv) < 6:
        print ('Please pass audio_folder file_list cpu/gpu batch_size case_number qa_mode')
        exit(0)
    try:
        path= "OUTPUT_IMAGES_PYTHON/NEW_API/FILE_READER/" + "audio"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    data_path = sys.argv[1]
    file_list = sys.argv[2]
    device_type = sys.argv[3]
    if(device_type == "cpu"):
        _rocal_cpu = True
    else:
        _rocal_cpu = False
    batch_size = int(sys.argv[4])
    case_number = int(sys.argv[5])
    qa_mode = int(sys.argv[6])
    if qa_mode:
        batch_size = 8
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    print("*********************************************************************")
    augmentation_list = {0: "pre_emphasis_filter", 1: "slice", 2: "spectrogram", 3: "mel_filter_bank", 4: "to_decibels", 5: "normalize"}
    print("RUNNING ", augmentation_list[case_number])
    audio_pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=_rocal_cpu)
    with audio_pipeline:
        audio, label = fn.readers.file(
            file_root=data_path,
            file_list=file_list,
            )
        audio_decode = fn.decoders.audio(audio, file_root=data_path, file_list_path=file_list, downmix=False, shard_id=0, num_shards=1, storage_type=9, stick_to_shard=False)
        if case_number == 0:
            outputs = fn.preemphasis_filter(audio_decode)
            audio_pipeline.setOutputs(outputs)
        elif case_number == 1:
            begin, length = fn.nonsilent_region(audio_decode, cutoff_db=-60)
            outputs = fn.slice(audio_decode,
                               anchor=[begin],
                               shape=[length],
                               normalized_anchor=False,
                               normalized_shape=False,
                               axes=[0],
                               rocal_tensor_output_type = types.FLOAT)
            audio_pipeline.setOutputs(outputs)
        elif case_number == 2:
            outputs = fn.spectrogram(audio_decode,
                                     nfft=512,
                                     window_length=320,
                                     window_step=160,
                                     rocal_tensor_output_type = types.FLOAT)
            audio_pipeline.setOutputs(outputs)
        elif case_number == 3:
            spec = fn.spectrogram(audio_decode,
                                  nfft=512,
                                  window_length=320,
                                  window_step=160,
                                  rocal_tensor_output_type = types.FLOAT)
            outputs = fn.mel_filter_bank(spec,
                                         sample_rate=16000,
                                         nfilter=80)
            audio_pipeline.setOutputs(outputs)
        elif case_number == 4:
            outputs = fn.to_decibels(audio_decode,
                                     multiplier=np.log(10),
                                     reference=1.0,
                                     cutoff_db=np.log(1e-20),
                                     rocal_tensor_output_type=types.FLOAT)
            audio_pipeline.setOutputs(outputs)
        elif case_number == 5:
            outputs = fn.normalize(audio_decode, axes=[1])
            audio_pipeline.setOutputs(outputs)
        else:
            audio_pipeline.setOutputs(audio_decode)
        
    audio_pipeline.build()
    audioIteratorPipeline = ROCALAudioIterator(audio_pipeline, auto_reset=True, device=device_type)
    augmentation_name = augmentation_list[case_number]
    match_cnt = 0
    for e in range(1):
        print("Epoch :: ", e)
        cnt = 0
        torch.set_printoptions(threshold=5000, profile="full", edgeitems=100)
        for i , it in enumerate(audioIteratorPipeline):
            print("************************************** i *************************************", i)
            for audio, label, roi in zip(it[0], it[1], it[2]):
                audio = audio.cpu().detach().numpy()
                roi = roi.cpu().detach().numpy()
                label = label.cpu().detach().numpy()
                if qa_mode == 0:
                    print("label: ", label)
                    print("roi: ", roi)
                    print("audio: ", audio)
                elif qa_mode == 1 and cnt < 8:
                    match = compare_output(audio, roi, augmentation_name, label)
                    match_cnt += 1
                cnt = cnt + 1
        print("EPOCH DONE", e)
        
        if qa_mode:
            if match_cnt == batch_size:
                print("PASSED!")
            else:
                print("FAILED! {%d / %d} matching with reference outputs".format(match_cnt, batch_size))

if __name__ == '__main__':
    main()
