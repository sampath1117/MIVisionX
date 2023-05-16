/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <vx_ext_rpp.h>
#include "node_resample.h"
#include "exception.h"

ResampleNode::ResampleNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs)
{
}

void ResampleNode::create_node()
{
    // std::cerr << "ResampleNode::create_node()";
    if(_node)
        return;
    //        auto in_sample_rate = _inputs[0]->info().get_sample_rate();
    // for (uint i=0;i < _batch_size; i++)
    // {
    //     std::cerr << "\n in_sample_rate : " << in_sample_rate->at(i);
    // }

    _src_frames_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, _batch_size);
    _src_channels_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, _batch_size);
    _src_sample_rate_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _batch_size);
    vx_status status , status1, status2;

    status = vxAddArrayItems(_src_frames_array, _batch_size, _src_frames.data(), sizeof(vx_int32));
    status1 = vxAddArrayItems(_src_channels_array, _batch_size, _src_channels.data(), sizeof(vx_int32));
    status2 = vxAddArrayItems(_src_sample_rate_array, _batch_size, _inputs[0]->info().get_sample_rate()->data(), sizeof(vx_float32));
    
    if(status != 0 )
        THROW(" vxAddArrayItems for _src_frames_array failed in the resample node (vxExtrppNode_Resample)  node: "+ TOSTR(status) )
    if(status1 != 0 )
        THROW(" vxAddArrayItems for _src_channels_array failed in the resample node (vxExtrppNode_Resample)  node: "+ TOSTR(status) )
    if(status2 != 0 )
        THROW(" vxAddArrayItems for _src_sample_rate_array failed in the resample node (vxExtrppNode_Resample)  node: "+ TOSTR(status) )

    vx_scalar quality = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, &_quality);
    vx_scalar _max_dst_width_scalar = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, &_max_dst_width);
    _node = vxExtrppNode_Resample(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _src_tensor_roi, _dst_tensor_roi,
                                 _resample_rate->handle(), _src_sample_rate_array, _src_frames_array, _src_channels_array, quality, _batch_size, _max_dst_width_scalar);
    
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the copy (vxExtrppNode_Resample) node failed: "+ TOSTR(status))
}

void ResampleNode::update_node()
{

    vx_status src_roi_status = vxCopyArrayRange((vx_array)_src_sample_rate_array, 0, _batch_size , sizeof(vx_float32), _inputs[0]->info().get_sample_rate()->data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    // vx_status resample_rate_status  = vxQueryTensor((vx_tensor)_resample_rate->handle(), VX_TENSOR_BUFFER_HOST, &_out_sample_rate_array, sizeof(vx_float32));
    
    if((src_roi_status) != 0)
        THROW(" Failed calling vxCopyArrayRange with status in resample node : "+ TOSTR(src_roi_status) )

    // auto audio_roi = _inputs[0]->info().get_roi();
    // auto audio_input_sample_rate = _inputs[0]->info().get_sample_rate();
    _max_dst_width = 0;
    // for (uint i=0; i < _batch_size; i++)
    // {
        // _src_frames[i] = audio_roi[i].x1;
        // _src_channels[i] = audio_roi[i].y1;
        // // TODO: Formula shared by Sampath - update the dst width & height later - calc ratio & then update it.
        // std::cerr << "\n _out_sample_rate_array[i] :" << _out_sample_rate_array[i];
        // std::cerr << "\n audio_input_sample_rate->at(i) : " << audio_input_sample_rate->at(i);
        // _scale_ratio = _out_sample_rate_array[i] / (float)audio_input_sample_rate->at(i);
        // _resample_rate_vec[i] = _out_sample_rate_array[i];
        // _dst_roi_width_vec[i] = (int)std::ceil(_scale_ratio * _src_frames[i]); 
        // _dst_roi_height_vec[i] = _src_channels[i];
        // std::cerr << "_dst_roi_width_vec[i] : " <<_dst_roi_width_vec[i];
        // std::cerr << "_dst_roi_height_vec[i] :" << _dst_roi_height_vec[i];
        // _max_dst_width = std::max(_max_dst_width, _dst_roi_width_vec[i]);
        // std::cerr << "_max_dst_width : " << _max_dst_width;
    // }

    // vx_status status1, status2, status3;
    // status2 = vxCopyArrayRange((vx_array)_src_frames_array, 0, _batch_size, sizeof(vx_int32), _src_frames.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    // status3 = vxCopyArrayRange((vx_array)_src_channels_array, 0, _batch_size, sizeof(vx_int32), _src_channels.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

    // if(status2 != 0 )
    //     THROW("ERROR: vxCopyArrayRange failed in the resample node (vxExtrppNode_Resample)  node for _src_frames_array: "+ TOSTR(status2))
    // if(status3 != 0 )
    //     THROW("ERROR: vxCopyArrayRange failed in the resample node (vxExtrppNode_Resample)  node for _src_channels_arra: "+ TOSTR(status3))

    // _outputs[0]->update_tensor_roi(_dst_roi_width_vec, _dst_roi_height_vec);
    // _outputs[0]->update_audio_tensor_sample_rate(_resample_rate_vec);

}

void ResampleNode::init(RocalTensor resample_rate, float quality)
{
    _resample_rate = resample_rate;
    _quality = quality;
    _max_dst_width = 0;
    _resample_rate_dims = _resample_rate->info().num_of_dims();
    _resample_rate_vec.resize(_batch_size * _resample_rate_dims);
    _src_frames.resize(_batch_size);
    _src_channels.resize(_batch_size);
    _dst_roi_width_vec.resize(_batch_size);
    _dst_roi_height_vec.resize(_batch_size);
}