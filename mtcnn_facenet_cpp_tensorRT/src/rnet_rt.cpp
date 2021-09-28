
//Created by zhou on 18-5-4.

#include "rnet_rt.h"


Rnet_engine::Rnet_engine() : baseEngine("../mtCNNModels/det2_relu.prototxt",
                                        "../mtCNNModels/det2_relu.caffemodel",
                                        "data",
                                        "conv5-2",
                                        "prob1"

                                        ) {
};

Rnet_engine::~Rnet_engine() {
    shutdownProtobufLibrary();
}

void Rnet_engine::init(int row, int col) {

    IHostMemory *gieModelStream{nullptr};
    const int max_batch_size = 1;
    //generate Tensorrt model
    caffeToGIEModel(prototxt, model, std::vector<std::string>{OUTPUT_PROB_NAME, OUTPUT_LOCATION_NAME}, max_batch_size,
                    gieModelStream);

}


Rnet::Rnet(const Rnet_engine &rnet_engine) : BatchSize(1),
                                             INPUT_C(3),
                                             Engine(rnet_engine.context->getEngine()) {

    Rthreshold = 0.7;
    this->score_ = new pBox;
    this->location_ = new pBox;
    this->rgb = new pBox;
    INPUT_W = 24;
    INPUT_H = 24;
    //calculate output shape
    this->score_->width     = 1;
    this->score_->height    = 1;
    this->score_->channel   = 2;

    this->location_->width  = 1;
    this->location_->height = 1;
    this->location_->channel= 4;

    OUT_PROB_SIZE = this->score_->width * this->score_->height * this->score_->channel;
    OUT_LOCATION_SIZE = this->location_->width * this->location_->height * this->location_->channel;
    //allocate memory for outputs
    this->rgb->pdata = (float *) malloc(INPUT_C * INPUT_H * INPUT_W * sizeof(float));
    this->score_->pdata = (float *) malloc(2 * sizeof(float));
    this->location_->pdata = (float *) malloc(4 * sizeof(float));

    assert(Engine.getNbBindings() == 3);
    inputIndex = Engine.getBindingIndex(rnet_engine.INPUT_BLOB_NAME);
    outputProb = Engine.getBindingIndex(rnet_engine.OUTPUT_PROB_NAME);
    outputLocation = Engine.getBindingIndex(rnet_engine.OUTPUT_LOCATION_NAME);
    //creat GPU buffers and stream
    CHECK(cudaMalloc(&buffers[inputIndex], BatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputProb], BatchSize * OUT_PROB_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputLocation], BatchSize * OUT_LOCATION_SIZE * sizeof(float)));
    CHECK(cudaStreamCreate(&stream));
}

Rnet::~Rnet()  {
    delete (score_);
    delete (location_);
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputProb]));
    CHECK(cudaFree(buffers[outputLocation]));
}

void Rnet::run(cv::Mat &image,  const Rnet_engine &rnet_engine) {
    //DMA the input to the GPU ,execute the batch asynchronously and DMA it back;
    image2Matrix(image, this->rgb);
    CHECK(cudaMemcpyAsync(buffers[inputIndex], this->rgb->pdata,
                          BatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    rnet_engine.context->enqueue(BatchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(this->location_->pdata, buffers[outputLocation], BatchSize * OUT_LOCATION_SIZE* sizeof(float),
                          cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(this->score_->pdata, buffers[outputProb], BatchSize * OUT_PROB_SIZE* sizeof(float),
                          cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

}
