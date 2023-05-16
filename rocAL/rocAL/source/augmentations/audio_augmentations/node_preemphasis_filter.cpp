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
#include "node_preemphasis_filter.h"
#include "exception.h"

PreemphasisFilterNode::PreemphasisFilterNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs),
        _preemph_coeff(PREEMPH_COEFF_RANGE[0], PREEMPH_COEFF_RANGE[1])
{
}

void PreemphasisFilterNode::create_node()
{
    if(_node)
        return;
    _src_samples_size.resize(_batch_size);
    // _src_samples_size_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, _batch_size);
    _preemph_coeff.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    vx_status status = VX_SUCCESS;
    // status |= vxAddArrayItems(_src_samples_size_array, _batch_size, _src_samples_size.data(), sizeof(vx_int32));
    if(status != 0)
        THROW(" vxAddArrayItems failed in the PreemphasisFilter node node: "+ TOSTR(status) + "  "+ TOSTR(status))

    vx_scalar border_type = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_preemph_border);
    _node = vxExtrppNode_PreemphasisFilter(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _src_tensor_roi, _dst_tensor_roi, _preemph_coeff.default_array(), border_type, _batch_size);

    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the copy (vxExtrppNode_ToDecibels) node failed: "+ TOSTR(status))

}

void PreemphasisFilterNode::update_node()
{
    auto audio_roi = _inputs[0]->info().get_roi();
    auto output_audio_roi = _outputs[0]->info().get_roi();
    for (uint i=0; i < _batch_size; i++)
    {
        // Calculating size = frames * channel
        _src_samples_size[i] = audio_roi[i].x1 * audio_roi[i].y1;
        _dst_roi_width_vec[i] = audio_roi[i].x1;
        _dst_roi_height_vec[i] = audio_roi[i].y1;
        // std::cerr << "\n In PreEmphasis Filter : " <<"\n audio_roi[i].x1" << _src_samples_size[i]<< "\n audio_roi[i].y1" << audio_roi[i].y1;
        // std::cerr << "\n In PreEmphasis Filter : " << "\n output_audio_roi[i].x1 : " << output_audio_roi[i].x1;
    }
    // vx_status src_roi_status;
    // src_roi_status = vxCopyArrayRange((vx_array)_src_samples_size_array, 0, _batch_size, sizeof(vx_uint32), _src_samples_size.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    // if(src_roi_status != 0)
    //     THROW(" Failed calling vxCopyArrayRange for src / dst roi status : "+ TOSTR(src_roi_status))
     _preemph_coeff.update_array();
    // _outputs[0]->update_tensor_roi(_dst_roi_width_vec, _dst_roi_height_vec);
}

void PreemphasisFilterNode::init(FloatParam* preemph_coeff, RocalAudioBorderType preemph_border)
{
    _preemph_coeff.set_param(core(preemph_coeff));
    _preemph_border = preemph_border;
    _dst_roi_width_vec.resize(_batch_size);
    _dst_roi_height_vec.resize(_batch_size);
}