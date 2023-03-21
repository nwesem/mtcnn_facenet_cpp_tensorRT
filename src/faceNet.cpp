#include "faceNet.h"
#include <vector>
#include <cmath>

int FaceNetClassifier::m_classCount = 0;

FaceNetClassifier::FaceNetClassifier
(Logger gLogger, DataType dtype, const string uffFile, const string engineFile, int batchSize, bool serializeEngine,
        float knownPersonThreshold, int maxFacesPerScene, int frameWidth, int frameHeight) {

    m_INPUT_C = static_cast<const int>(3);
    m_INPUT_H = static_cast<const int>(160);
    m_INPUT_W = static_cast<const int>(160);
    m_frameWidth = static_cast<const int>(frameWidth);
    m_frameHeight = static_cast<const int>(frameHeight);
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
        // std::cout << "size" << size;
        IRuntime* runtime = createInferRuntime(m_gLogger);
        assert(runtime != nullptr);
        m_engine = runtime->deserializeCudaEngine(trtModelStream_.data(), size, nullptr);
        std::cout << std::endl;
    }
    else {
        IBuilder *builder = createInferBuilder(m_gLogger);
        IBuilderConfig* config = builder->createBuilderConfig();
        INetworkDefinition *network = builder->createNetworkV2(0U);
        IUffParser *parser = createUffParser();
        parser->registerInput("input", Dims3(160, 160, 3), UffInputOrder::kNHWC);
        parser->registerOutput("Bottleneck/BatchNorm/batchnorm/add_1");

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
            config->setFlag(BuilderFlag::kFP16);
        }
        else if (m_dtype == DataType::kINT8) {
            config->setFlag(BuilderFlag::kINT8);
            // ToDo
            //builder->setInt8Calibrator()
        }
        builder->setMaxBatchSize(m_batchSize);
        config->setMaxWorkspaceSize(1<<30);
        // strict will force selected datatype, even when another was faster
        //builder->setStrictTypeConstraints(true);
        // Disable DLA, because many layers are still not supported
        // and this causes additional latency.
        //builder->allowGPUFallback(true);
        //builder->setDefaultDeviceType(DeviceType::kDLA);
        //builder->setDLACore(1);
        m_engine = builder->buildEngineWithConfig(*network, *config);

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
            struct CroppedFace currFace;
            cv::resize(tempCrop, currFace.faceMat, cv::Size(160, 160), 0, 0, cv::INTER_CUBIC);
            currFace.x1 = it->x1;
            currFace.y1 = it->y1;
            currFace.x2 = it->x2;
            currFace.y2 = it->y2;            
            m_croppedFaces.push_back(currFace);
        }
    }
    //ToDo align
}

void FaceNetClassifier::preprocessFaces() {
    // preprocess according to facenet training and flatten for input to runtime engine
    for (int i = 0; i < m_croppedFaces.size(); i++) {
        //mean and std
        cv::cvtColor(m_croppedFaces[i].faceMat, m_croppedFaces[i].faceMat, cv::COLOR_RGB2BGR);
        cv::Mat temp = m_croppedFaces[i].faceMat.reshape(1, m_croppedFaces[i].faceMat.rows * 3);
        cv::Mat mean3;
        cv::Mat stddev3;
        cv::meanStdDev(temp, mean3, stddev3);

        double mean_pxl = mean3.at<double>(0);
        double stddev_pxl = stddev3.at<double>(0);
        cv::Mat image2;
        m_croppedFaces[i].faceMat.convertTo(image2, CV_64FC1);
        m_croppedFaces[i].faceMat = image2;
        // fix by peererror
        cv::Mat mat(4, 1, CV_64FC1);
		mat.at <double>(0, 0) = mean_pxl;
		mat.at <double>(1, 0) = mean_pxl;
		mat.at <double>(2, 0) = mean_pxl;
		mat.at <double>(3, 0) = 0;
        m_croppedFaces[i].faceMat = m_croppedFaces[i].faceMat - mat;
        // end fix
        m_croppedFaces[i].faceMat = m_croppedFaces[i].faceMat / stddev_pxl;
        m_croppedFaces[i].faceMat.convertTo(image2, CV_32FC3);
        m_croppedFaces[i].faceMat = image2;
    }
}


void FaceNetClassifier::doInference(float* inputData, float* output) {
    int size_of_single_input = 3 * 160 * 160 * sizeof(float);
    int size_of_single_output = 128 * sizeof(float);
    int inputIndex = m_engine->getBindingIndex("input");
    int outputIndex = m_engine->getBindingIndex("Bottleneck/BatchNorm/batchnorm/add_1");

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


void FaceNetClassifier::forwardAddFace(cv::Mat image, std::vector<struct Bbox> outputBbox,
        const string className) {
    
    //cv::resize(image, image, cv::Size(1280, 720), 0, 0, cv::INTER_CUBIC);
    getCroppedFacesAndAlign(image, outputBbox);
    if(!m_croppedFaces.empty()) {
        preprocessFaces();
        doInference((float*)m_croppedFaces[0].faceMat.ptr<float>(0), m_output);
        struct KnownID person;
        person.className = className;
        person.classNumber = m_classCount;
        person.embeddedFace.insert(person.embeddedFace.begin(), m_output, m_output+128);
        m_knownFaces.push_back(person);
        m_classCount++;
    }
    m_croppedFaces.clear();
}

void FaceNetClassifier::forward(cv::Mat frame, std::vector<struct Bbox> outputBbox) {
    getCroppedFacesAndAlign(frame, outputBbox); // ToDo align faces according to points
    preprocessFaces();
    for(int i = 0; i < m_croppedFaces.size(); i++) {
        doInference((float*)m_croppedFaces[i].faceMat.ptr<float>(0), m_output);
        m_embeddings.insert(m_embeddings.end(), m_output, m_output+128);
    }
}

void FaceNetClassifier::featureMatching(cv::Mat &image) {

    for(int i = 0; i < (m_embeddings.size()/128); i++) {
        double minDistance = 10.* m_knownPersonThresh;
        float currDistance = 0.;
        int winner = -1;
        for (int j = 0; j < m_knownFaces.size(); j++) {
            std:vector<float> currEmbedding(128);
            std::copy_n(m_embeddings.begin()+(i*128), 128, currEmbedding.begin());
            currDistance = vectors_distance(currEmbedding, m_knownFaces[j].embeddedFace);
            // printf("The distance to %s is %.10f \n", m_knownFaces[j].className.c_str(), currDistance);
            // if ((currDistance < m_knownPersonThresh) && (currDistance < minDistance)) {
            if (currDistance < minDistance) {
                    minDistance = currDistance;
                    winner = j;
            }
            currEmbedding.clear();
        }
        float fontScaler = static_cast<float>(m_croppedFaces[i].x2 - m_croppedFaces[i].x1)/static_cast<float>(m_frameWidth);
        cv::rectangle(image, cv::Point(m_croppedFaces[i].y1, m_croppedFaces[i].x1), cv::Point(m_croppedFaces[i].y2, m_croppedFaces[i].x2), 
                        cv::Scalar(0,0,255), 2,8,0);
        if (minDistance <= m_knownPersonThresh) {
            cv::putText(image, m_knownFaces[winner].className, cv::Point(m_croppedFaces[i].y1+2, m_croppedFaces[i].x2-3),
                    cv::FONT_HERSHEY_DUPLEX, 0.1 + 2*fontScaler,  cv::Scalar(0,0,255,255), 1);
        }
        else if (minDistance > m_knownPersonThresh || winner == -1){
            cv::putText(image, "New Person", cv::Point(m_croppedFaces[i].y1+2, m_croppedFaces[i].x2-3),
                    cv::FONT_HERSHEY_DUPLEX, 0.1 + 2*fontScaler ,  cv::Scalar(0,0,255,255), 1);
        }
    }
}

void FaceNetClassifier::addNewFace(cv::Mat &image, std::vector<struct Bbox> outputBbox) {
    std::cout << "Adding new person...\nPlease make sure there is only one face in the current frame.\n"
              << "What's your name? ";
    string newName;
    std::cin >> newName;
    std::cout << "Hi " << newName << ", you will be added to the database.\n";
    forwardAddFace(image, outputBbox, newName);
    string filePath = "../imgs/";
    filePath.append(newName);
    filePath.append(".jpg");
    cv::imwrite(filePath, image);
}

void FaceNetClassifier::resetVariables() {
    m_embeddings.clear();
    m_croppedFaces.clear();
}

FaceNetClassifier::~FaceNetClassifier() {
    // this leads to segfault 
    // this->m_engine->destroy();
    // this->m_context->destroy();
    // std::cout << "FaceNet was destructed" << std::endl;
}

std::vector<float> l2Normalize(const std::vector<float>& vec) {
    float norm = 0.0;
    for (const auto& element : vec) {
        norm += element * element;
    }
    norm = std::sqrt(norm);
    std::vector<float> normalizedVec(vec.size());
    for (std::size_t i = 0; i < vec.size(); ++i) {
        normalizedVec[i] = vec[i] / norm;
    }
    return normalizedVec;
}

// HELPER FUNCTIONS
// Computes the distance between two std::vectors
float vectors_distance(const std::vector<float>& aa, const std::vector<float>& bb) {
    std::vector<float>	a = l2Normalize(aa);
    std::vector<float>	b = l2Normalize(bb);
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
