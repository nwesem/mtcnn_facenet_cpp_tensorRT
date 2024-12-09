//
// Created by zhou on 18-5-4.
//

#include "baseEngine.h"

int baseEngine::det1_relu_counter = 1;

baseEngine::baseEngine(const char * prototxt,const char* model,const  char* input_name,const char*location_name,
                       const char* prob_name, const char *point_name) :
                             prototxt(prototxt),
                             model(model),
                             INPUT_BLOB_NAME(input_name),
                             OUTPUT_LOCATION_NAME(location_name),
                             OUTPUT_PROB_NAME(prob_name),
                             OUTPUT_POINT_NAME(point_name)
{
};
baseEngine::~baseEngine() {
    shutdownProtobufLibrary();
}

void baseEngine::init(int row,int col) {

}
void baseEngine::caffeToGIEModel(const std::string &deployFile,                // name for caffe prototxt
                                  const std::string &modelFile,                // name for model
                                  const std::vector<std::string> &outputs,   // network outputs
                                  unsigned int maxBatchSize,                    // batch size - NB must be at least as large as the batch we want to run with)
                                  IHostMemory *&gieModelStream)    // output buffer for the GIE model
{
    size_t lastIdx = model.find_last_of(".");
    string enginePath = model.substr(0, lastIdx);
    if(enginePath.find("det1_relu") != std::string::npos) {
        enginePath.append(std::to_string(det1_relu_counter));
        enginePath.append(".engine");
        det1_relu_counter++;
    }
    else {
        enginePath.append(".engine");
    }
    std::cout << "rawName = " << enginePath << std::endl;
    if(fileExists(enginePath)) {
        std::vector<char> trtModelStream_;
        size_t size{ 0 };

        std::ifstream file(enginePath, std::ios::binary);
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
        IRuntime* runtime = createInferRuntime(gLogger);
        assert(runtime != nullptr);
        ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream_.data(), size, nullptr);
        assert(engine);
        context = engine->createExecutionContext();
        std::cout << std::endl;
    }
    else {
        // create the builder
        IBuilder *builder = createInferBuilder(gLogger);
        IBuilderConfig* config = builder->createBuilderConfig();

        // parse the caffe model to populate the network, then set the outputs
        INetworkDefinition *network = builder->createNetworkV2(0U);
        ICaffeParser *parser = createCaffeParser();

        const IBlobNameToTensor *blobNameToTensor = parser->parse(deployFile.c_str(),
                                                                  modelFile.c_str(),
                                                                  *network,
                                                                  nvinfer1::DataType::kHALF);
        // specify which tensors are outputs
        for (auto &s : outputs)
            network->markOutput(*blobNameToTensor->find(s.c_str()));

        // Build the engine
        builder->setMaxBatchSize(maxBatchSize);
        config->setMaxWorkspaceSize(1 << 25);
        ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
        assert(engine);

        context = engine->createExecutionContext();

        // Serialize engine
        ofstream planFile;
        planFile.open(enginePath);
        IHostMemory *serializedEngine = engine->serialize();
        planFile.write((char *) serializedEngine->data(), serializedEngine->size());
        planFile.close();


        // we don't need the network any more, and we can destroy the parser
        network->destroy();
        parser->destroy();
        builder->destroy();
    }
}