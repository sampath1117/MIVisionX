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

#include <cstdio>
#include <cstring>
#include <commons.h>
#include "sndfile_decoder.h"

SndFileDecoder::SndFileDecoder(){};

AudioDecoder::Status SndFileDecoder::decode(float* buffer)
{
    int readcount = 0;
    readcount = sf_readf_float(_sf_ptr, buffer, _sfinfo.frames);
    if(readcount != _sfinfo.frames)
    {
        printf("Not able to decode all frames. Only decoded %d frames\n", readcount);
        sf_close(_sf_ptr);
        AudioDecoder::Status status = Status::CONTENT_DECODE_FAILED;
        return status;
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