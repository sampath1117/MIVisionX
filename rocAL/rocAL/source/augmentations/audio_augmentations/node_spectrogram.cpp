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
#include "node_spectrogram.h"
#include "exception.h"

SpectrogramNode::SpectrogramNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs)
{
}

void SpectrogramNode::create_node()
{
    if(_node)
        return;
    _src_samples_length.resize(_batch_size);
    _window_fn.resize(_window_length);
    vx_status status = VX_SUCCESS;

    _window_fn_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _window_length);
    if(_is_window_empty != true)
    {
        status |= vxAddArrayItems(_window_fn_array, _window_length, _window_fn.data(), sizeof(vx_float32));
        if(status != 0)
            THROW(" vxAddArrayItems failed in the Spectrogram node node: "+ TOSTR(status) + "  "+ TOSTR(status))
    }
    vx_scalar center_windows = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_BOOL, &_center_windows);
    vx_scalar reflect_padding = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_BOOL, &_reflect_padding);
    vx_scalar spec_layout = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, &_spec_layout);
    vx_scalar power = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_power);
    vx_scalar nfft_size = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_nfft_size);
    vx_scalar window_length = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_window_length);
    vx_scalar window_step = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_window_step);
    vx_scalar is_window_empty = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_BOOL, &_is_window_empty);
    _node = vxExtrppNode_Spectrogram(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _src_tensor_roi, _dst_tensor_roi, _window_fn_array,
                                     center_windows, reflect_padding, spec_layout, power, nfft_size, window_length,
                                     window_step, is_window_empty, _batch_size);

    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the copy (vxExtrppNode_Spectrogram) node failed: "+ TOSTR(status))

}

void SpectrogramNode::update_node()
{
    auto audio_roi = _inputs[0]->info().get_roi();
    for (uint i=0; i < _batch_size; i++)
    {
        _src_samples_length[i] = audio_roi[i].x1;
        // std::cerr << "\n spec :: _src_samples_length"<< _src_samples_length[i];
        // std::cerr<<"\n  audio_roi[i].x1 :" <<  audio_roi[i].x1;
        // std::cerr<<"\n  audio_roi[i].y1 :" <<  audio_roi[i].y1;
        _dst_roi_width_vec[i] =(( _src_samples_length[i] - _window_offset ) / _window_step) + 1;
        _dst_roi_height_vec[i] =  (_nfft_size / 2 )+ 1;
    }
    _outputs[0]->update_tensor_roi(_dst_roi_width_vec, _dst_roi_height_vec);

}

void SpectrogramNode::init(bool center_windows, bool reflect_padding, RocalSpectrogramLayout spec_layout,
                            int power, int nfft_size, int window_length, int window_step, std::vector<float> &window_fn)
{
    _center_windows = center_windows;
    _reflect_padding = reflect_padding;
    _spec_layout = spec_layout;
    _power = power;
    _nfft_size = nfft_size;
    _window_length = window_length;
    if(!center_windows)
        _window_offset = _window_length;
    _window_step = window_step;
    _dst_roi_width_vec.resize(_batch_size);
    _dst_roi_height_vec.resize(_batch_size);
    if(window_fn.empty())
    {
        _is_window_empty = true;
    }
}