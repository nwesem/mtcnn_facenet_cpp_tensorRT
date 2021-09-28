//
// Created by zhou on 18-5-4.
//
#include "common.h"
#include <string>
#include <cassert>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvCaffeParser.h"
#ifndef MAIN_BASEENGINE_H
#define MAIN_BASEENGINE_H
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace std;


class baseEngine {
public:
    static int det1_relu_counter;
    baseEngine(const char *prototxt,const char*model,const char*out_name,
               const char*location_name,const char*prob_name,const char *point_name = NULL);
    virtual ~baseEngine();
    virtual void caffeToGIEModel(const std::string& deployFile,				// name for caffe prototxt
                                 const std::string& modelFile,				// name for model
                                 const std::vector<std::string>& outputs,   // network outputs
                                 unsigned int maxBatchSize,					// batch size - NB must be at least as large as the batch we want to run with)
                                 IHostMemory *&gieModelStream);             // output buffer for the GIE model
    virtual void init(int row,int col);
    friend class Pnet;
    const string prototxt;
    const string model   ;
    const char *INPUT_BLOB_NAME;
    const char *OUTPUT_PROB_NAME;
    const char *OUTPUT_LOCATION_NAME;
    const char *OUTPUT_POINT_NAME;
    Logger gLogger;
    IExecutionContext *context;
};


#endif //MAIN_BASEENGINE_H
