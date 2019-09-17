//
// Created by zhou on 18-10-2.
//

#ifndef MAIN_ONET_RT_H
#define MAIN_ONET_RT_H
#include "baseEngine.h"
#include "network.h"


class Onet_engine : public baseEngine {

public:
    Onet_engine();
    ~Onet_engine();
    void init(int row, int col);
    friend class Onet;

};

class Onet {
public:
    Onet(const Onet_engine &onet_engine);
    ~Onet();
    void run(cv::Mat &image, const Onet_engine &engine);
    mydataFmt Othreshold;
    cudaStream_t stream;
    struct pBox *location_;
    struct pBox *score_;
    struct pBox *points_;
    struct pBox *rgb;
private:
    const int BatchSize;
    const int INPUT_C;
    const ICudaEngine &Engine;
    //must be computed at runtime
    int INPUT_H;
    int INPUT_W;
    int OUT_PROB_SIZE;
    int OUT_LOCATION_SIZE;
    int OUT_POINTS_SIZE;
    int inputIndex,outputProb,outputLocation,outputPoints;
    void *buffers[4];

};
#endif //MAIN_ONET_RT_H
