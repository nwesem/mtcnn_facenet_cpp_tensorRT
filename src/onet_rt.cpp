//
// Created by zhou on 18-10-2.
//

#include "onet_rt.h"

Onet_engine::Onet_engine() : baseEngine("../mtCNNModels/det3_relu.prototxt",
                                        "../mtCNNModels/det3_relu.caffemodel",
                                        "data",
                                        "conv6-2",
                                        "prob1",
                                        "conv6-3"
                                        ) {
};

Onet_engine::~Onet_engine() {
    shutdownProtobufLibrary();
}

void Onet_engine::init(int row, int col) {
    IHostMemory *gieModelStream{nullptr};
    const int max_batch_size = 1;
    //generate Tensorrt model
    caffeToGIEModel(prototxt, model, std::vector<std::string>{OUTPUT_PROB_NAME, OUTPUT_LOCATION_NAME,OUTPUT_POINT_NAME}, max_batch_size,
                    gieModelStream);

}


Onet::Onet(const Onet_engine &onet_engine) : BatchSize(1),
                                             INPUT_C(3),
                                             Engine(onet_engine.context->getEngine()) {

    Othreshold      = 0.8;
    this->score_    = new pBox;
    this->location_ = new pBox;
    this->rgb       = new pBox;
    this->points_   = new pBox;
    INPUT_W = 48;
    INPUT_H = 48;
    //calculate output shape
    this->score_->width         = 1;
    this->score_->height        = 1;
    this->score_->channel       = 2;

    this->location_->width      = 1;
    this->location_->height     = 1;
    this->location_->channel    = 4;

    this->points_->width        = 1;
    this->points_->height       = 1;
    this->points_->channel      = 10;


    OUT_PROB_SIZE       = this->score_->width * this->score_->height * this->score_->channel;
    OUT_LOCATION_SIZE   = this->location_->width * this->location_->height * this->location_->channel;
    OUT_POINTS_SIZE     = this->points_->width * this->points_->height * this->points_->channel;
    //allocate memory for outputs
    this->rgb->pdata        = (float *) malloc(INPUT_C * INPUT_H * INPUT_W * sizeof(float));
    this->score_->pdata     = (float *) malloc(2 * sizeof(float));
    this->location_->pdata  = (float *) malloc(4 * sizeof(float));
    this->points_->pdata    = (float *) malloc(10 * sizeof(float));

    assert(Engine.getNbBindings() == 4);
    inputIndex      = Engine.getBindingIndex(onet_engine.INPUT_BLOB_NAME);
    outputProb      = Engine.getBindingIndex(onet_engine.OUTPUT_PROB_NAME);
    outputLocation  = Engine.getBindingIndex(onet_engine.OUTPUT_LOCATION_NAME);
    outputPoints    = Engine.getBindingIndex(onet_engine.OUTPUT_POINT_NAME);

    //creat GPU buffers and stream
    CHECK(cudaMalloc(&buffers[inputIndex], BatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputProb], BatchSize * OUT_PROB_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputLocation], BatchSize * OUT_LOCATION_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputPoints], BatchSize * OUT_POINTS_SIZE * sizeof(float)));
    CHECK(cudaStreamCreate(&stream));
}

Onet::~Onet()  {

    delete (score_);
    delete (location_);
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputProb]));
    CHECK(cudaFree(buffers[outputLocation]));
    CHECK(cudaFree(buffers[outputPoints]));
}

void Onet::run(cv::Mat &image,  const Onet_engine &onet_engine) {


    //DMA the input to the GPU ,execute the batch asynchronously and DMA it back;
    image2Matrix(image, this->rgb);
    CHECK(cudaMemcpyAsync(buffers[inputIndex], this->rgb->pdata,
                          BatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    onet_engine.context->enqueue(BatchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(this->location_->pdata, buffers[outputLocation], BatchSize * OUT_LOCATION_SIZE* sizeof(float),
                          cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(this->score_->pdata, buffers[outputProb], BatchSize * OUT_PROB_SIZE* sizeof(float),
                          cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(this->points_->pdata, buffers[outputPoints], BatchSize * OUT_POINTS_SIZE* sizeof(float),
                          cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

}
