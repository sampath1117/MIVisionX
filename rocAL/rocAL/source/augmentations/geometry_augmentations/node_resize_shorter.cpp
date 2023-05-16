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
#include <graph.h>
#include "node_resize_shorter.h"
#include "exception.h"


ResizeShorterNode::ResizeShorterNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs)
{
}

void ResizeShorterNode::create_node()
{
    if (_node)
        return;

    _dest_width_val.resize(_batch_size);
    _dest_height_val.resize(_batch_size);

    std::vector<uint32_t> dst_roi_width(_batch_size, _outputs[0]->info().max_dims()[0]);
    std::vector<uint32_t> dst_roi_height(_batch_size, _outputs[0]->info().max_dims()[1]);

    _dst_roi_width = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _dst_roi_height = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);

    vx_status width_status, height_status;

    width_status = vxAddArrayItems(_dst_roi_width, _batch_size, dst_roi_width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(_dst_roi_height, _batch_size, dst_roi_height.data(), sizeof(vx_uint32));
    if (width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the resize (vxExtrppNode_Resizetensor) node: " + TOSTR(width_status) + "  " + TOSTR(height_status))

    vx_scalar interpolation = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, &_interpolation_type);
    unsigned int chnShift = 0;
    bool packed;
    vx_scalar chnToggle = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, &chnShift);
    vx_scalar is_packed = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_BOOL, &packed);

    vx_scalar layout = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, &_layout);
    vx_scalar roi_type = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, &_roi_type);

    // _node = vxExtrppNode_Resize(_graph->get(), _inputs[0]->handle(), _src_tensor_roi, _outputs[0]->handle(), _src_tensor_roi,
                                // _dst_roi_width, _dst_roi_height, interpolation, is_packed, chnToggle, layout, roi_type, _batch_size);

    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the resize (vxExtrppNode_Resize) node failed: " + TOSTR(status))
}

void ResizeShorterNode::update_node()
{
    RocalROI* src_roi = _inputs[0]->info().get_roi();

    for (uint i = 0; i < _batch_size; i++)
    {
        uint w = src_roi[i].x2;
        uint h = src_roi[i].y2;

        uint old_short, old_long, new_short, new_long, new_w, new_h;

        old_short = (w <= h) ? w : h;
        old_long = (w <= h) ? h : w;
        new_short = _size;
        new_long = static_cast<int>(_size * old_long / old_short);

        new_w = (w <= h) ? new_short : new_long;
        new_h = (w <= h) ? new_long : new_short;

        _dest_height_val[i] = new_h;
        _dest_width_val[i] = new_w;
    }
    vxCopyArrayRange((vx_array)_dst_roi_width, 0, _batch_size, sizeof(uint), _dest_width_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_dst_roi_height, 0, _batch_size, sizeof(uint), _dest_height_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

    _outputs[0]->update_tensor_roi(_dest_width_val, _dest_height_val);
}

void ResizeShorterNode::init(int size)
{
    _size = size;
    _interpolation_type = 1; // Default value for resize Bilinear
    _layout = (unsigned) _outputs[0]->info().layout();
}
