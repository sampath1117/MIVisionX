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

#pragma once
#include <memory>
#include <map>
#include "sndfile_decoder.h"
#include "reader_factory.h"
#include "timing_debug.h"
#include "loader_module.h"
enum class AudioSourceEvaluatorStatus
{
    OK = 0,
    UNSUPPORTED_DECODER_TYPE,
    UNSUPPORTED_STORAGE_TYPE,
};
// enum class MaxSizeEvaluationPolicy
// {
//     MAXIMUM_FOUND_SIZE,
//     MOST_FREQUENT_SIZE
// };

class AudioSourceEvaluator
{
public:
    AudioSourceEvaluatorStatus create(ReaderConfig reader_cfg, DecoderConfig decoder_cfg);
    void find_max_dimension();
    // void set_size_evaluation_policy(MaxSizeEvaluationPolicy arg);
    size_t max_samples();
    size_t max_channels();

private:
    class FindMaxSize
    {
    public:
        // void set_policy(MaxSizeEvaluationPolicy arg) { _policy = arg; }
        void process_sample(unsigned val);
        unsigned get_max() { return _max; };
    private:
        // MaxSizeEvaluationPolicy _policy = MaxSizeEvaluationPolicy::MAXIMUM_FOUND_SIZE;
        std::map<unsigned,unsigned> _hist;
        unsigned _max = 0;
        unsigned _max_count = 0;
    };
    FindMaxSize _samples_max;
    FindMaxSize _channels_max;
    DecoderConfig _decoder_cfg_cv;
    std::shared_ptr<AudioDecoder> _decoder;
    std::shared_ptr<Reader> _reader;
    std::shared_ptr<MetaDataReader> _meta_data_reader;
    std::vector<unsigned char> _header_buff;
    static const size_t COMPRESSED_SIZE = 1024 * 1024; // 1 MB
    std::string _input_path;
};

