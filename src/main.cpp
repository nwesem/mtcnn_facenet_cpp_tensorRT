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

// Uncomment to print timings in milliseconds
// #define LOG_TIMES

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
    DataType dtype = DataType::kHALF;
    //DataType dtype = DataType::kFLOAT;
    bool serializeEngine = true;
    int batchSize = 1;
    int nbFrames = 0;
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
        std::size_t index = paths[i].fileName.find_last_of(".");
        std::string rawName = paths[i].fileName.substr(0,index);
        faceNet.forwardAddFace(image, outputBbox, rawName);
        faceNet.resetVariables();
    }
    outputBbox.clear();

    // loop over frames with inference
    auto globalTimeStart = chrono::steady_clock::now();
    while (true) {
        videoStreamer.getFrame(frame);
        if (frame.empty()) {
            std::cout << "Empty frame! Exiting...\n Try restarting nvargus-daemon by "
                         "doing: sudo systemctl restart nvargus-daemon" << std::endl;
            break;
        }
        auto startMTCNN = chrono::steady_clock::now();
        outputBbox = mtCNN.findFace(frame);
        auto endMTCNN = chrono::steady_clock::now();
        auto startForward = chrono::steady_clock::now();
        faceNet.forward(frame, outputBbox);
        auto endForward = chrono::steady_clock::now();
        auto startFeatM = chrono::steady_clock::now();
        faceNet.featureMatching(frame);
        auto endFeatM = chrono::steady_clock::now();
        faceNet.resetVariables();
        
        cv::imshow("VideoSource", frame);
        nbFrames++;
        outputBbox.clear();
        frame.release();

        char keyboard = cv::waitKey(1);
        if (keyboard == 'q' || keyboard == 27)
            break;
        else if(keyboard == 'n') {
            auto dTimeStart = chrono::steady_clock::now();
            videoStreamer.getFrame(frame);
            outputBbox = mtCNN.findFace(frame);
            cv::imshow("VideoSource", frame);
            faceNet.addNewFace(frame, outputBbox);
            auto dTimeEnd = chrono::steady_clock::now();
            globalTimeStart += (dTimeEnd - dTimeStart);
        }

        #ifdef LOG_TIMES
        std::cout << "mtCNN took " << std::chrono::duration_cast<chrono::milliseconds>(endMTCNN - startMTCNN).count() << "ms\n";
        std::cout << "Forward took " << std::chrono::duration_cast<chrono::milliseconds>(endForward - startForward).count() << "ms\n";
        std::cout << "Feature matching took " << std::chrono::duration_cast<chrono::milliseconds>(endFeatM - startFeatM).count() << "ms\n\n";
        #endif  // LOG_TIMES
    }
    auto globalTimeEnd = chrono::steady_clock::now();
    cv::destroyAllWindows();
    videoStreamer.release();
    auto milliseconds = chrono::duration_cast<chrono::milliseconds>(globalTimeEnd-globalTimeStart).count();
    double seconds = double(milliseconds)/1000.;
    double fps = nbFrames/seconds;

    std::cout << "Counted " << nbFrames << " frames in " << double(milliseconds)/1000. << " seconds!" <<
              " This equals " << fps << "fps.\n";

    return 0;
}

