#include <iostream>
#include <string>
#include <chrono>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <l2norm_helper.h>
#include <opencv2/highgui.hpp>
#include "faceNet.h"
#include "videoStreamer.h"
#include "network.h"
#include "mtcnn.h"

using namespace nvinfer1;
using namespace nvuffparser;


int main()
{
    Logger gLogger = Logger();
    // Register default TRT plugins (e.g. LRelu_TRT)
    if (!initLibNvInferPlugins(&gLogger, "")) { return 1; }

    // USER DEFINED VALUES
    const string uffFile="../facenetModels/facenet.uff";
    const string engineFile="../facenetModels/facenet.engine";
    DataType dtype = DataType::kHALF;       // ToDo calibrator for INT8
    //DataType dtype = DataType::kFLOAT;
    bool serializeEngine = true;
    int batchSize = 1;
    int nbFrames = 0;
    time_t timeStart, timeEnd;
    int videoFrameWidth = 640;
    int videoFrameHeight = 480;
    int maxFacesPerScene = 5;
    float knownPersonThreshold = 1.;
    bool isCSICam = true;

    // init facenet
    FaceNetClassifier faceNet = FaceNetClassifier(gLogger, dtype, uffFile, engineFile, batchSize, serializeEngine,
            knownPersonThreshold, maxFacesPerScene, videoFrameWidth, videoFrameHeight);

    // init opencv stuff
    VideoStreamer videoStreamer = VideoStreamer(0, videoFrameWidth, videoFrameHeight, 60, isCSICam);
    cv::Mat frame;

    // init mtCNN
    mtcnn mtCNN(videoFrameHeight, videoFrameWidth);

    //init Bbox and allocate memory for "maxFacesPerScene" faces per scene
    std::vector<struct Bbox> outputBbox;
    outputBbox.reserve(maxFacesPerScene);

    // get embeddings of known faces
    std::vector<struct Paths> paths;
    cv::Mat image;
    getFilePaths("../imgs", paths);
    for(int i=0; i < paths.size(); i++) {
        loadInputImage(paths[i].absPath, image, videoFrameWidth, videoFrameHeight);
        outputBbox = mtCNN.findFace(image);
        faceNet.forwardPreprocessing(image, outputBbox, paths, i);
        faceNet.resetVariables();
    }
    outputBbox.clear();

    // loop over frames with inference
    auto start = chrono::steady_clock::now();
    while (true) {
        videoStreamer.getFrame(frame);
        if (frame.empty()) {
            std::cout << "Empty frame! Exiting..." << std::endl;
            break;
        }
        auto startMTCNN = chrono::steady_clock::now();
        outputBbox = mtCNN.findFace(frame);
        auto endMTCNN = chrono::steady_clock::now();

        auto startFW = chrono::steady_clock::now();
        faceNet.forward(frame, outputBbox);
        faceNet.featureMatching(frame);
        faceNet.resetVariables();
        auto endFW = chrono::steady_clock::now();

        cv::imshow("InputFrame", frame);
        nbFrames++;
        char keyboard = cv::waitKey(1);
        if (keyboard == 'q' || keyboard == 27)
            break;
        std::cout << "mtCNN took " << std::chrono::duration_cast<chrono::milliseconds>(endMTCNN - startMTCNN).count() <<
                  "ms" << std::endl;
        std::cout << "Inference took " << std::chrono::duration_cast<chrono::milliseconds>(endFW - startFW).count() <<
                  "ms" << std::endl;

        outputBbox.clear();
    }
    auto end = chrono::steady_clock::now();
    cv::destroyAllWindows();
    auto milliseconds = chrono::duration_cast<chrono::milliseconds>(end-start).count();
    double seconds = double(milliseconds)/1000.;
    double fps = nbFrames/seconds;

    std::cout << "Counted " << nbFrames << " frames in " << double(milliseconds)/1000. << " seconds!" <<
              " This equals " << fps << "fps." << std::endl;

    return 0;
}

