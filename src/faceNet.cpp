#include "faceNet.h"


FaceNetClassifier::FaceNetClassifier
(Logger gLogger, DataType dtype, const string uffFile, const string engineFile, int batchSize, bool serializeEngine,
        float knownPersonThreshold, int maxFacesPerScene) {

    m_INPUT_C = static_cast<const int>(3);
    m_INPUT_H = static_cast<const int>(160);
    m_INPUT_W = static_cast<const int>(160);
    m_gLogger = gLogger;
    m_dtype = dtype;
    m_uffFile = static_cast<const string>(uffFile);
    m_engineFile = static_cast<const string>(engineFile);
    m_batchSize = batchSize;
    m_serializeEngine = serializeEngine;
    m_maxFacesPerScene = maxFacesPerScene;
    m_croppedFaces.reserve(maxFacesPerScene);
    m_embeddings.reserve(128);
    m_knownPersonThresh = knownPersonThreshold;

    // load engine from .engine file or create new engine
    this->createOrLoadEngine();
}


void FaceNetClassifier::createOrLoadEngine() {
    if(fileExists(m_engineFile)) {
        std::vector<char> trtModelStream_;
        size_t size{ 0 };

        std::ifstream file(m_engineFile, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream_.resize(size);
            std::cout << "size" << trtModelStream_.size() << std::endl;
            file.read(trtModelStream_.data(), size);
            file.close();
        }
        std::cout << "size" << size;
        IRuntime* runtime = createInferRuntime(m_gLogger);
        assert(runtime != nullptr);
        m_engine = runtime->deserializeCudaEngine(trtModelStream_.data(), size, nullptr);
    }
    else {
        IBuilder *builder = createInferBuilder(m_gLogger);
        INetworkDefinition *network = builder->createNetwork();
        IUffParser *parser = createUffParser();
        parser->registerInput("input", DimsCHW(160, 160, 3), UffInputOrder::kNHWC);
        parser->registerOutput("embeddings");
        //  parser->registerOutput("orientation/l2_normalize");
        //  parser->registerOutput("dimension/LeakyRelu");
        //  parser->registerOutput("confidence/Softmax");
        // END USER DEFINED VALUES

        if (!parser->parse(m_uffFile.c_str(), *network, m_dtype))
        {
            cout << "Failed to parse UFF\n";
            builder->destroy();
            parser->destroy();
            network->destroy();
            throw std::exception();
        }

        /* build engine */
        if (m_dtype == DataType::kHALF)
        {
            builder->setFp16Mode(true);
        }
        else if (m_dtype == DataType::kINT8) {
            builder->setInt8Mode(true);
            // ToDo
            //builder->setInt8Calibrator()
        }
        builder->setMaxBatchSize(m_batchSize);
        builder->setMaxWorkspaceSize(1<<30);
        // strict will force selected datatype, even when another was faster
        //builder->setStrictTypeConstraints(true);
        // Disable DLA, because many layers are still not supported
        // and this causes additional latency.
        //builder->allowGPUFallback(true);
        //builder->setDefaultDeviceType(DeviceType::kDLA);
        //builder->setDLACore(1);
        m_engine = builder->buildCudaEngine(*network);

        /* serialize engine and write to file */
        if(m_serializeEngine) {
            ofstream planFile;
            planFile.open(m_engineFile);
            IHostMemory *serializedEngine = m_engine->serialize();
            planFile.write((char *) serializedEngine->data(), serializedEngine->size());
            planFile.close();
        }

        /* break down */
        builder->destroy();
        parser->destroy();
        network->destroy();
    }
    m_context = m_engine->createExecutionContext();
}


void FaceNetClassifier::getCroppedFacesAndAlign(cv::Mat frame, std::vector<struct Bbox> outputBbox) {
    for(vector<struct Bbox>::iterator it=outputBbox.begin(); it!=outputBbox.end();it++){
        if((*it).exist){
            cv::Rect facePos(cv::Point((*it).y1, (*it).x1), cv::Point((*it).y2, (*it).x2));
            cv::Mat tempCrop = frame(facePos);
            cv::Mat finalCrop;
            cv::resize(tempCrop, finalCrop, cv::Size(160, 160), 0, 0, cv::INTER_CUBIC);
            m_croppedFaces.push_back(finalCrop);
            
            // DEBUG
            // cv::imshow("croppedFace1", finalCrop);
        }
    }
    //ToDo align?
}

void FaceNetClassifier::preprocessFaces() {
    // preprocess according to facenet training and flatten for input to runtime engine
    for (int i = 0; i < m_croppedFaces.size(); i++) {
        //mean and std
        cvtColor(m_croppedFaces[i], m_croppedFaces[i], CV_RGB2BGR);
        cv::Mat temp = m_croppedFaces[i].reshape(1, m_croppedFaces[i].rows * 3);
        cv::Mat mean3;
        cv::Mat stddev3;
        cv::meanStdDev(temp, mean3, stddev3);

        double mean_pxl = mean3.at<double>(0);
        double stddev_pxl = stddev3.at<double>(0);
        cv::Mat image2;
        m_croppedFaces[i].convertTo(image2, CV_64FC1);
        m_croppedFaces[i] = image2;
        m_croppedFaces[i] = m_croppedFaces[i] - cv::Vec3d(mean_pxl, mean_pxl, mean_pxl);
        m_croppedFaces[i] = m_croppedFaces[i] / stddev_pxl;

        //new
        m_croppedFaces[i].convertTo(image2, CV_32FC3);
        m_croppedFaces[i] = image2;
    }
}


void FaceNetClassifier::doInference(float* inputData, float* output) {
    int size_of_single_input = 3 * 160 * 160 * sizeof(float);
    int size_of_single_output = 128 * sizeof(float);
    int inputIndex = m_engine->getBindingIndex("input");
    int outputIndex = m_engine->getBindingIndex("embeddings");

    void* buffers[2];

    cudaMalloc(&buffers[inputIndex], m_batchSize * size_of_single_input);
    cudaMalloc(&buffers[outputIndex], m_batchSize * size_of_single_output);

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // copy data to GPU and execute
    CHECK(cudaMemcpyAsync(buffers[inputIndex], inputData, m_batchSize * size_of_single_input, cudaMemcpyHostToDevice, stream));
    m_context->enqueue(m_batchSize, &buffers[0], stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], m_batchSize * size_of_single_output, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}


void FaceNetClassifier::forwardPreprocessing(cv::Mat image, std::vector<struct Bbox> outputBbox,
        std::vector<struct Paths> paths, int nbFace) {
    
    //cv::resize(image, image, cv::Size(1280, 720), 0, 0, cv::INTER_CUBIC);
    getCroppedFacesAndAlign(image, outputBbox);
    if(!m_croppedFaces.empty()) {
        preprocessFaces();
        doInference((float*)m_croppedFaces[0].ptr<float>(0), m_output);
        struct KnownID person;
        std::size_t index = paths[nbFace].fileName.find_last_of(".");
        std::string rawName = paths[nbFace].fileName.substr(0,index);
        person.className = rawName;
        person.classNumber = nbFace;
        person.embeddedFace.insert(person.embeddedFace.begin(), m_output, m_output+128);
        m_knownFaces.push_back(person);
    }
    m_croppedFaces.clear();
}

void FaceNetClassifier::forward(cv::Mat frame, std::vector<struct Bbox> outputBbox) {
    this->getCroppedFacesAndAlign(frame, outputBbox); // ToDo align faces according to points
    preprocessFaces();
    for(int i = 0; i < m_croppedFaces.size(); i++) {
        doInference((float*)m_croppedFaces[i].ptr<float>(0), m_output);
        m_embeddings.insert(m_embeddings.end(), m_output, m_output+128);
        //std::cout << "embeddings.size() = " << embeddings.size() << std::endl;
    }
}

void FaceNetClassifier::featureMatching() {

    for(int i = 0; i < (m_embeddings.size()/128); i++) {
        double minDistance = 100.;
        float currDistance = 0.;
        int winner;
        for (int j = 0; j < m_knownFaces.size(); j++) {
            std:vector<float> currEmbedding(128);
            std::copy_n(m_embeddings.begin()+(i*128), 128, currEmbedding.begin());
            // currDistance = std::sqrt(std::inner_product(currEmbedding.begin(), currEmbedding.end(), m_knownFaces[j].embeddedFace.begin(), 0));
            currDistance = vectors_distance(currEmbedding, m_knownFaces[j].embeddedFace);
            // std::cout << "Distance to " << m_knownFaces[j].className << " is " << currDistance << std::endl;
            // std::cout << "currEmbedding = " << std::endl;
            // for (auto it = currEmbedding.begin(); it!=currEmbedding.end(); ++it) std::cout << *it << " ";
            // std::cout << std::endl;
            // std::cout << "knownFace of " << m_knownFaces[j].className << " = " << std::endl;
            // for (auto it = m_knownFaces[j].embeddedFace.begin(); it!=m_knownFaces[j].embeddedFace.end(); ++it) std::cout << *it << " ";
            // std::cout << std::endl;
            printf("The distance to %s is %.10f \n", m_knownFaces[j].className.c_str(), currDistance);
            if (currDistance < minDistance) {
                    minDistance = currDistance;
                    winner = j;
            }
            currEmbedding.clear();
        }
        if (minDistance < m_knownPersonThresh) {
            std::cout << m_knownFaces[winner].className << std::endl;
        }
        else {
            std::cout << "New Person?" << std::endl;
        }
    }
    std::cout << "\n";
}

void FaceNetClassifier::resetVariables() {
    m_embeddings.clear();
    m_croppedFaces.clear();
}

FaceNetClassifier::~FaceNetClassifier() {
    // this leads to segfault if engine or context could not be created during class instantiation
    this->m_engine->destroy();
    this->m_context->destroy();
    std::cout << "FaceNet was destructed" << std::endl;
}


// HELPER FUNCTIONS
// Computes the distance between two std::vectors
float vectors_distance(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<double>	auxiliary;
    std::transform (a.begin(), a.end(), b.begin(), std::back_inserter(auxiliary),//
                    [](float element1, float element2) {return pow((element1-element2),2);});
    auxiliary.shrink_to_fit();

    float loopSum = 0.;
    for(auto it=auxiliary.begin(); it!=auxiliary.end(); ++it) loopSum += *it;

    return  std::sqrt(loopSum);
} 



inline unsigned int elementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32:
            // Fallthrough, same as kFLOAT
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
    }
    assert(0);
    return 0;
}
