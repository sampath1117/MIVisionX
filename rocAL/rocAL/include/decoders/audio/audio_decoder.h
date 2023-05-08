/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#include <cstddef>
#include <iostream>
#include <vector>
#include "parameter_factory.h"
#include "sndfile.h"

#if _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#endif

inline double Hann(double x) {
    return 0.5 * (1 + std::cos(x * M_PI));
}

inline float sinc(float x)
{
    x *= M_PI;
    return (std::abs(x) < 1e-5f) ? (1.0f - x * x * (1.0f / 6)) : std::sin(x) / x;
}

struct ResamplingWindow {
    inline std::pair<int, int> input_range(float x) {
        int xc = ceilf(x);
        int i0 = xc - lobes;
        int i1 = xc + lobes;
        return {i0, i1};
    }

    inline float operator()(float x) {
        float fi = x * scale + center;
        int i = floorf(fi);
        float di = fi - i;
        i = std::max(std::min(i, lookup_size - 2), 0);
        float curr = lookup[i];
        float next = lookup[i + 1];
        return curr + di * (next - curr);
    }

    inline __m128 operator()(__m128 x) {
        __m128 fi = _mm_add_ps(x * _mm_set1_ps(scale), _mm_set1_ps(center));
        __m128i i = _mm_cvttps_epi32(fi);
        __m128 fifloor = _mm_cvtepi32_ps(i);
        __m128 di = _mm_sub_ps(fi, fifloor);
        // i = _mm_max_epi32(_mm_min_epi32(i, pxLookupMax), xmm_px0);
        int idx[4];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(idx), i);
        __m128 curr = _mm_setr_ps(lookup[idx[0]],   lookup[idx[1]],
                                lookup[idx[2]],   lookup[idx[3]]);
        __m128 next = _mm_setr_ps(lookup[idx[0]+1], lookup[idx[1]+1],
                                lookup[idx[2]+1], lookup[idx[3]+1]);
        return _mm_add_ps(curr, _mm_mul_ps(di, _mm_sub_ps(next, curr)));
    }

    float scale = 1, center = 1;
    int lobes = 0, coeffs = 0;
    int lookup_size = 0;
    __m128i pxLookupMax;
    std::vector<float> lookup;
};


enum class AudioDecoderType
{
    SOFTWARE_DECODE = 0
};

class AudioDecoderConfig
{
public:
    AudioDecoderConfig() {}
    explicit AudioDecoderConfig(AudioDecoderType type) : _type(type) {}
    virtual AudioDecoderType type() { return _type; };
    AudioDecoderType _type = AudioDecoderType::SOFTWARE_DECODE;
};

class AudioDecoder
{
public:
    enum class Status
    {
        OK = 0,
        HEADER_DECODE_FAILED,
        CONTENT_DECODE_FAILED,
        UNSUPPORTED,
        FAILED,
        NO_MEMORY
    };
    virtual AudioDecoder::Status initialize(const char *src_filename) = 0;
    virtual AudioDecoder::Status decode(float* buffer, ResamplingWindow &window, bool resample = false, FloatParam* sample_rate_dist = NULL, float sample_rate = 16000) = 0; //to pass buffer & number of frames/samples to decode
    virtual AudioDecoder::Status decode_info(int* samples, int* channels, float* sample_rates) = 0; //to decode info about the audio samples
    virtual void release() = 0;
    virtual ~AudioDecoder() = default;
protected:
    const char *_src_filename = NULL;
    SF_INFO _sfinfo;
    SNDFILE* _sf_ptr;
};