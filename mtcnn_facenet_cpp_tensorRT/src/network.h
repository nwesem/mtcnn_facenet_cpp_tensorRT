//c++ network  author : liqi
//Nangjing University of Posts and Telecommunications
//date 2017.5.21,20:27
#ifndef NETWORK_H
#define NETWORK_H
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <algorithm>
#include <stdlib.h>
#include <memory.h>
#include <fstream>
#include <cstring>
#include <cblas.h>
#include <string>
#include <math.h>
#include "pBox.h"
#include <assert.h>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvCaffeParser.h"

void image2Matrix(const cv::Mat &image, const struct pBox *pbox);
bool cmpScore(struct orderScore lsh, struct orderScore rsh);
void nms(vector<struct Bbox> &boundingBox_, vector<struct orderScore> &bboxScore_, const float overlap_threshold, string modelname = "Union");
void refineAndSquareBbox(vector<struct Bbox> &vecBbox, const int &height, const int &width,bool square);

#endif