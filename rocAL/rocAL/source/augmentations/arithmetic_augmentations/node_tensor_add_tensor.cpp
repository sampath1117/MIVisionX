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
#include "node_tensor_add_tensor.h"
#include "exception.h"

TensorAddTensorNode::TensorAddTensorNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs)
{
}

void TensorAddTensorNode::create_node()
{
    if(_node)
        return;
    std::cerr <<" \n Here in create Node of Tensor to Tensor addition ";

    _node = vxExtrppNode_TensorAddTensor(_graph->get(), _inputs[0]->handle(), _inputs[1]->handle(), _outputs[0]->handle(), _src_tensor_roi, _dst_tensor_roi, _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the (vxExtrppNode_TensorAddTensor) node failed: "+ TOSTR(status))

}

void TensorAddTensorNode::update_node()
{
    // vx_status src1_roi_status = vxCopyArrayRange((vx_array)_src_tensor_roi, 0, _batch_size  * 4, sizeof(vx_uint32), _inputs[0]->info().get_roi(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    // if(src1_roi_status != 0)
    //     THROW(" Failed calling vxCopyArrayRange for src / dst roi status : "+ TOSTR(src1_roi_status))
}

void TensorAddTensorNode::init()
{

}