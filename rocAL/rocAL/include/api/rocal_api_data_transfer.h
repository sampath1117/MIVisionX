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

#ifndef MIVISIONX_ROCAL_API_DATA_TRANSFER_H
#define MIVISIONX_ROCAL_API_DATA_TRANSFER_H
#include "rocal_api_types.h"

///
/// \param rocal_context
/// \param output_tensor_list Returns a list of the tensor (decoder/augmentation outputs) with set_output = True
extern "C"  RocalTensorList ROCAL_API_CALL rocalGetOutputTensors(RocalContext p_context);

///
/// \param rocal_context
/// \param output_images The buffer that will be filled with output images with set_output = True
extern "C" void ROCAL_API_CALL rocalSetOutputs(RocalContext p_context, unsigned int num_of_outputs, std::vector<RocalTensor> &output_images);

#endif //MIVISIONX_ROCAL_API_DATA_TRANSFER_H
