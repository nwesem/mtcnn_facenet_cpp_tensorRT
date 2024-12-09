//
// Created by zhou on 18-4-30.
//

#ifndef _TRT_COMMON_H_
#define _TRT_COMMON_H_
#include "NvInfer.h"
#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <numeric>
#include <dirent.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime_api.h>

#define CHECK(status)									\
{														\
	if (status != 0)									\
	{													\
		std::cout << "Cuda failure: " << status;		\
		abort();										\
	}													\
}


// Logger for GIE info/warning/errors
class Logger : public nvinfer1::ILogger
{
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        //if (severity == Severity::kINFO) return;

        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
            case Severity::kERROR: std::cerr << "ERROR: "; break;
            case Severity::kWARNING: std::cerr << "WARNING: "; break;
            case Severity::kINFO: std::cerr << "INFO: "; break;
            default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }
};

struct Paths {
    std::string absPath;
    std::string fileName;
};

struct KnownID {
    std::string className;
    int classNumber;
    std::vector<float> embeddedFace;
};

inline bool fileExists(const std::string &name) {
    std::ifstream f(name.c_str());
    return f.good();
}

void* safeCudaMalloc(size_t memSize);
inline int64_t volume(const nvinfer1::Dims& d);
std::vector<std::pair<int64_t, nvinfer1::DataType>>
calculateBindingBufferSizes(const nvinfer1::ICudaEngine& engine, int nbBindings, int batchSize);
void getFilePaths(std::string imagesPath, std::vector<struct Paths>& paths);
void loadInputImage(std::string inputFilePath, cv::Mat& image, int videoFrameWidth, int videoFrameHeight);

#endif // _TRT_COMMON_H_
