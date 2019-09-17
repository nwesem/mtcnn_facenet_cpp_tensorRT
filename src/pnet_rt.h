//
// Created by zhou on 18-4-30.
//

#ifndef MAIN_PNET_RT_H
#define MAIN_PNET_RT_H

#include "network.h"
#include "common.h"
#include "baseEngine.h"
#endif //MAIN_PNET_RT_H
using namespace nvinfer1;
using namespace nvcaffeparser1;

class Pnet_engine:public baseEngine
{

public:
    Pnet_engine();
    ~Pnet_engine();
    void init(int row,int col);
    friend class Pnet;

};



class Pnet
{
public:
    Pnet(int row,int col,const Pnet_engine& pnet_engine);
    ~Pnet();
    void run(cv::Mat &image, float scale,const Pnet_engine& engine);
    float nms_threshold;
    mydataFmt Pthreshold;
    cudaStream_t stream;

    vector<struct Bbox> boundingBox_;
    vector<orderScore> bboxScore_;
private:

    const int BatchSize ;
    const int INPUT_C ;
    const ICudaEngine &Engine;
    //must be computed at runtime
    int INPUT_H ;
    int INPUT_W ;
    int OUT_PROB_SIZE;
    int OUT_LOCATION_SIZE;
    int     inputIndex,
            outputProb,
            outputLocation;
    void *buffers[3];
    struct pBox *location_;
    struct pBox *score_;
    struct pBox *rgb;

    void generateBbox(const struct pBox *score, const struct pBox *location, mydataFmt scale);
};

