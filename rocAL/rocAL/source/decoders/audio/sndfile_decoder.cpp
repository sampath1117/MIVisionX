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

#include <cstdio>
#include <cstring>
#include <commons.h>
#include "sndfile_decoder.h"

#if _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#endif

SndFileDecoder::SndFileDecoder(){};

AudioDecoder::Status SndFileDecoder::decode(float* buffer, ResamplingWindow &window, bool resample, FloatParam* sample_rate_dist, float sample_rate)
{
    if(!resample) {
        int readcount = 0;
        readcount = sf_readf_float(_sf_ptr, buffer, _sfinfo.frames);
        if(readcount != _sfinfo.frames)
        {
            printf("Not able to decode all frames. Only decoded %d frames\n", readcount);
            sf_close(_sf_ptr);
            AudioDecoder::Status status = Status::CONTENT_DECODE_FAILED;
            return status;
        }
    } else {
        // Allocate temporary memory for input
        float *srcPtrTemp = (float *)malloc(_sfinfo.frames * sizeof(float));

        int readcount = 0;
        readcount = sf_readf_float(_sf_ptr, srcPtrTemp, _sfinfo.frames);
        // std::cerr<<"completed sf_readf_float"<<std::endl;
        if(readcount != _sfinfo.frames)
        {
            printf("Not able to decode all frames. Only decoded %d frames\n", readcount);
            sf_close(_sf_ptr);
            AudioDecoder::Status status = Status::CONTENT_DECODE_FAILED;
            return status;
        }

        float *dstPtrTemp = buffer;
        uint srcLength = _sfinfo.frames;
        float outRate = 16000 * sample_rate_dist->core->get();
        float inRate = 16000;
        int64_t outEnd = std::ceil(srcLength * outRate / inRate);
        int64_t inPos = 0;
        int64_t block = 1 << 8;
        double scale = (double)inRate / outRate;
        float fscale = scale;
        int64_t outBegin = 0;

        for (int64_t outBlock = outBegin; outBlock < outEnd; outBlock += block) {
            int64_t blockEnd = std::min(outBlock + block, outEnd);
            double inBlockRaw = outBlock * scale;
            int64_t inBlockRounded = std::floor(inBlockRaw);
            float inPos = inBlockRaw - inBlockRounded;
            const float * __restrict__ inBlockPtr = srcPtrTemp + inBlockRounded;

            for (int64_t outPos = outBlock; outPos < blockEnd; outPos++, inPos += fscale) {
                int i0, i1;
                std::tie(i0, i1) = window.input_range(inPos);
                if (i0 + inBlockRounded < 0)
                    i0 = -inBlockRounded;
                if (i1 + inBlockRounded > srcLength)
                    i1 = srcLength - inBlockRounded;
                float f = 0.0f;
                int i = i0;

                __m128 f4 = _mm_setzero_ps();
                __m128 x4 = _mm_setr_ps(i - inPos, i + 1 - inPos, i + 2 - inPos, i + 3 - inPos);
                for (; i + 3 < i1; i += 4) {
                    __m128 w4 = window(x4);

                    f4 = _mm_add_ps(f4, _mm_mul_ps(_mm_loadu_ps(inBlockPtr + i), w4));
                    x4 = _mm_add_ps(x4, _mm_set1_ps(4));
                }

                f4 = _mm_add_ps(f4, _mm_shuffle_ps(f4, f4, _MM_SHUFFLE(1, 0, 3, 2)));
                f4 = _mm_add_ps(f4, _mm_shuffle_ps(f4, f4, _MM_SHUFFLE(0, 1, 0, 1)));
                f = _mm_cvtss_f32(f4);

                float x = i - inPos;
                for (; i < i1; i++, x++) {
                    float w = window(x);
                    f += inBlockPtr[i] * w;
                }

                dstPtrTemp[outPos] = f;
            }
        }

        // free temporary allocated memory
        free(srcPtrTemp);
    }

    AudioDecoder::Status status = Status::OK;
    return status;
}

AudioDecoder::Status SndFileDecoder::decode_info(int* samples, int* channels, float* sample_rate)
{
    // Set the samples and channels using the struct variables _sfinfo.samples and _sfinfo.channels
    *samples = _sfinfo.frames;
    *channels = _sfinfo.channels;
    *sample_rate = _sfinfo.samplerate;

    if (_sfinfo.channels < 1)
	{	printf("Not able to process less than %d channels\n", *channels);
        sf_close(_sf_ptr);
		AudioDecoder::Status status = Status::HEADER_DECODE_FAILED;
		return status;
	};
    if (_sfinfo.frames < 1)
	{	printf("Not able to process less than %d frames\n", *samples);
        sf_close(_sf_ptr);
		AudioDecoder::Status status = Status::HEADER_DECODE_FAILED;
		return status;
	};
    AudioDecoder::Status status = Status::OK;
    return status;
}

// Initialize will open a new decoder and initialize the context
AudioDecoder::Status SndFileDecoder::initialize(const char *src_filename)
{
    _src_filename = src_filename;
    memset(&_sfinfo, 0, sizeof(_sfinfo)) ;
    if (!(_sf_ptr = sf_open(src_filename, SFM_READ, &_sfinfo)))
	{	/* Open failed so print an error message. */
		printf("Not able to open input file %s.\n", src_filename);
		/* Print the error message from libsndfile. */
		puts(sf_strerror(NULL));
        sf_close(_sf_ptr);
        AudioDecoder::Status status = Status::HEADER_DECODE_FAILED;
		return status;
	};
    //std::cout << "SRC FILENAME:" << src_filename << std::endl;
    AudioDecoder::Status status = Status::OK;
    return status;
}

void SndFileDecoder::release()
{
    if(_sf_ptr != NULL) {
      sf_close(_sf_ptr);
    }
}

SndFileDecoder::~SndFileDecoder()
{
}