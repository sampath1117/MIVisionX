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
#include "node_to_decibles.h"
#include "exception.h"

ToDeciblesNode::ToDeciblesNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs)
{
}

void ToDeciblesNode::create_node()
{
    if(_node)
        return;
    _src_samples_length.resize(_batch_size);
    _src_samples_length_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, _batch_size);
    vx_status status = VX_SUCCESS;
    status |= vxAddArrayItems(_src_samples_length_array, _batch_size, _src_samples_length.data(), sizeof(vx_int32));
    _src_samples_channels.resize(_batch_size);

    _src_samples_channels_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, _batch_size);
    status |= vxAddArrayItems(_src_samples_channels_array, _batch_size, _src_samples_channels.data(), sizeof(vx_int32));
    if(status != 0)
        THROW(" vxAddArrayItems failed in the Todecibles node node: "+ TOSTR(status) + "  "+ TOSTR(status))

    vx_scalar cut_off_db = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, &_cut_off_db);
    vx_scalar multiplier = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, &_multiplier);
    vx_scalar magnitude_reference= vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, &_magnitude_reference);
    _node = vxExtrppNode_ToDecibels(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _src_tensor_roi, _dst_tensor_roi, cut_off_db, multiplier, magnitude_reference, _batch_size);

    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the copy (vxExtrppNode_ToDecibels) node failed: "+ TOSTR(status))

}

void ToDeciblesNode::update_node()
{
auto audio_roi = _inputs[0]->info().get_roi();
    for (uint i=0; i < _batch_size; i++)
    {
        _src_samples_length[i] = audio_roi[i].x1;
        _src_samples_channels[i] = audio_roi[i].y1;
        _dst_roi_width_vec[i] = (audio_roi[i].x1);
        _dst_roi_height_vec[i] = (audio_roi[i].y1);
    }
}

void ToDeciblesNode::init(float cut_off_db, float multiplier, float magnitude_reference)
{
    _cut_off_db = cut_off_db;
    _multiplier = multiplier;
    _magnitude_reference = magnitude_reference;
    _dst_roi_width_vec.resize(_batch_size);
    _dst_roi_height_vec.resize(_batch_size);

}