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
#include "node_crop_mirror_normalize.h"
#include "exception.h"

CropMirrorNormalizeNode::CropMirrorNormalizeNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs),
        _mirror(MIRROR_RANGE[0], MIRROR_RANGE[1])
{
    _crop_param = std::make_shared<RocalCropParam>(_batch_size);
}

void CropMirrorNormalizeNode::create_node()
{
    if(_node)
        return;

    if(_crop_param->crop_h == 0 || _crop_param->crop_w == 0)
        THROW("Uninitialized destination dimension - Invalid Crop Sizes")
    _crop_param->create_array(_graph);
    _mean_vx.resize(_batch_size*3);
    _std_dev_vx.resize(_batch_size*3);
    for (uint i=0; i < _batch_size; i++ ) {
        _mean_vx[3*i] = _mean[0];
        _mean_vx[3*i+1] = _mean[1];
        _mean_vx[3*i+2] = _mean[2];

        _std_dev_vx[3*i] = _std_dev[0];
        _std_dev_vx[3*i+1] = _std_dev[1];
        _std_dev_vx[3*i+2] = _std_dev[2];
    }

    _mean_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _batch_size*3);
    _std_dev_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _batch_size*3);
    vx_status status = VX_SUCCESS;
    status |= vxAddArrayItems(_mean_array,_batch_size*3, _mean_vx.data(), sizeof(vx_float32));
    status |= vxAddArrayItems(_std_dev_array,_batch_size*3, _std_dev_vx.data(), sizeof(vx_float32));
    _mirror.create_array(_graph ,VX_TYPE_UINT32, _batch_size);
    if(status != 0)
        THROW(" vxAddArrayItems failed in the crop resize node (vxExtrppNode_CropMirrorNormalizeCropbatchPD    )  node: "+ TOSTR(status) + "  "+ TOSTR(status))

    vx_scalar layout = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_layout);
    vx_scalar roi_type = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_roi_type);
    // _node = vxExtrppNode_CropMirrorNormalize(_graph->get(), _inputs[0]->handle(),
    //                                                _src_tensor_roi, _outputs[0]->handle(),_src_tensor_roi,_crop_param->cropw_arr, _crop_param->croph_arr, _crop_param->x1_arr, _crop_param->y1_arr,
    //                                                 _mean_array, _std_dev_array, _mirror.default_array(), layout, roi_type, _batch_size);
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Error adding the crop mirror normalize tensor (vxExtrppNode_CropMirrorNormalizeCropbatchPD    ) failed: "+TOSTR(status))

}

void CropMirrorNormalizeNode::update_node()
{
    _crop_param->set_image_dimensions(_inputs[0]->info().get_roi());
    _crop_param->update_array();
    std::vector<uint32_t> crop_h_dims, crop_w_dims;
    _crop_param->get_crop_dimensions(crop_w_dims, crop_h_dims);
    _outputs[0]->update_tensor_roi(crop_w_dims, crop_h_dims);
    _mirror.update_array();

}

void CropMirrorNormalizeNode::init(int crop_h, int crop_w, float start_x, float start_y, std::vector<float>& mean, std::vector<float>& std_dev, IntParam *mirror,int layout)
{
    _crop_param->crop_h = crop_h;
    _crop_param->crop_w = crop_w;
    _crop_param->x1 = 0;
    _crop_param->y1 = 0;
    _mean   = mean;
    _std_dev = std_dev;
    _mirror.set_param(core(mirror));
    _layout = (unsigned) _outputs[0]->info().layout();
}