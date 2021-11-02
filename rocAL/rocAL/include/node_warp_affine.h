/*
Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"
#include "graph.h"

class WarpAffineNode : public Node
{
public:
    WarpAffineNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    WarpAffineNode() = delete;
    void init(float x0, float x1, float y0, float y1, float o0, float o1);
    void init(bool is_train, float rotate_probability, float half_body_probability, FloatParam *scale_factor, FloatParam *rotation_factor, FloatParam *x0, FloatParam *x1, FloatParam *y0, FloatParam *y1, FloatParam *o0, FloatParam *o1);
    vx_array get_src_width() { return _src_roi_width; }
    vx_array get_src_height() { return _src_roi_height; }
    float* get_affine_array() { return _inv_affine.data(); }
    void half_body_transform(int index, Center & box_center, Scale  & box_scale, float aspect_ratio);

protected:
    void create_node() override;
    void update_node() override;

private:
    ParameterVX<float> _scale_factor;
    ParameterVX<float> _rotation_factor;
    ParameterVX<float> _x0;
    ParameterVX<float> _x1;
    ParameterVX<float> _y0;
    ParameterVX<float> _y1;
    ParameterVX<float> _o0;
    ParameterVX<float> _o1;
    //ParameterVX<float> _rotate_probability;

    std::vector<float> _affine;
    std::vector<float> _inv_affine;
    vx_array _dst_roi_width, _dst_roi_height;
    vx_array _affine_array;
    std::vector<float> _dst_width, _dst_height;
    float _rotate_probability = 0.5;
    float _half_body_probability = 0.35;
    bool _is_train = false;
    constexpr static float COEFFICIENT_RANGE_0[2] = {-0.35, 0.35};
    constexpr static float COEFFICIENT_RANGE_1[2] = {0.65, 1.35};
    constexpr static float COEFFICIENT_RANGE_OFFSET[2] = {-10.0, 10.0};
    constexpr static float SCALE_RANGE[2] = {0.65, 1.35};
    constexpr static float ROTATION_RANGE[2] = {-90, 90};
    //constexpr static float ROTATION_PROBABILITY_RANGE[2] = {0.0, 1.0};
    void update_affine_array();
};

inline void get_dir(float point[], float dir[], float r)
{
    float sn = std::sin(r);
    float cs = std::cos(r);
    dir[0] = point[0] * cs - point[1] * sn;
    dir[1] = point[0] * sn + point[1] * cs;
}

inline void get_3rd_point(float mat[][3])
{
    float direct[2] = {mat[0][0] - mat[0][1], mat[1][0] - mat[1][1]};
    mat[0][2] = mat[0][1] - direct[1];
    mat[1][2] = mat[1][1] + direct[0];
}

inline void get_inverse(float m[3][3], float inv_m[3][3])
{
    float det = 0.0;

    //Calculate determinant of the matrix
    for (int i = 0; i < 3; i++)
    {
        det = det + (m[0][i] * (m[1][(i + 1) % 3] * m[2][(i + 2) % 3] - m[1][(i + 2) % 3] * m[2][(i + 1) % 3]));
    }

    if (det == 0)
    {
        return;
    }

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            inv_m[j][i] = ((m[(i + 1) % 3][(j + 1) % 3] * m[(i + 2) % 3][(j + 2) % 3]) - (m[(i + 1) % 3][(j + 2) % 3] * m[(i + 2) % 3][(j + 1) % 3])) / det;
        }
    }
}

inline void matrix_mult(float src[2][3], float dst[3][3], float *affine)
{
    // TO DO sampath change affine array calculation from 2d mat to pointer.
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            *affine = 0.0;
            for (int k = 0; k < 3; k++)
            {
                *affine += src[i][k] * dst[k][j];
            }
            affine++;
        }
    }
}

inline void invert_affine_tranform(float *affine,float *inv_affine)
{
    // std::vector<float> inv_affine;
    int step = 3;
    float det = affine[0]*affine[step+1]- affine[1]*affine[step];
    det = det!=0? (1.0/det):0;
    float a11,a12,a21,a22,b1,b2;
    a11 = affine[step+1]*det ;
    a12 = -affine[1]*det;
    a21 =  -affine[step]*det;
    a22 = affine[0]*det;
    b1 = -a11*affine[2] - a12*affine[step+2];
    b2 = -a21*affine[2] - a22*affine[step+2];

    inv_affine[0] = a11;
    inv_affine[1] = a12;
    inv_affine[2] = b1;
    inv_affine[3] = a21;
    inv_affine[4] = a22;
    inv_affine[5] = b2;
}