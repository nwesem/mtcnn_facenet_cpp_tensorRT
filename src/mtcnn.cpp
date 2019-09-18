#include "mtcnn.h"
// #define LOG
mtcnn::mtcnn(int row, int col){
    //set NMS thresholds
    nms_threshold[0] = 0.7;
    nms_threshold[1] = 0.7;
    nms_threshold[2] = 0.7;
    //set minimal face size (weidth in pixels)
    int minsize = 60;
    /*config  the pyramids */
    float minl = row<col?row:col;
    int MIN_DET_SIZE = 12;
    float m = (float)MIN_DET_SIZE/minsize;
    minl *= m;
    float factor = 0.709;
    int factor_count = 0;
    while(minl>MIN_DET_SIZE){
        if(factor_count>0)m = m*factor;
        scales_.push_back(m);
        minl *= factor;
        factor_count++;
    }
    float minside = row<col ? row : col;
    int count = 0;
    for (vector<float>::iterator it = scales_.begin(); it != scales_.end(); it++){
        if (*it > 1){
            cout << "the minsize is too small" << endl;
            while (1);
        }
        if (*it < (MIN_DET_SIZE / minside)){
            scales_.resize(count);
            break;
        }
        count++;
    }

    cout<<"\nStart generating mtCNN TenosrRT runtime models"<<endl;
    //generate pnet models
    pnet_engine = new Pnet_engine[scales_.size()];
    simpleFace_ = (Pnet**)malloc(sizeof(Pnet*)*scales_.size());
    // std::cout << "scales.size() = " << scales_.size() << std::endl;
    for (size_t i = 0; i < scales_.size(); i++) {
        int changedH = (int)ceil(row*scales_.at(i));
        int changedW = (int)ceil(col*scales_.at(i));
        std::cout << "changedH = " << changedH << ", changedW = " << changedW << std::endl;
        pnet_engine[i].init(changedH,changedW);
        simpleFace_[i] =  new Pnet(changedH,changedW,pnet_engine[i]);
    }
    cout<<"End generate pnet runtime models"<<endl;

    //generate rnet model
    rnet_engine = new Rnet_engine();
    rnet_engine->init(24,24);
    refineNet = new Rnet(*rnet_engine);
    cout<<"End generate rnet runtime models"<<endl;

    //generate onet model
    onet_engine = new Onet_engine();
    onet_engine->init(48,48);
    outNet = new Onet(*onet_engine);
    cout<<"End generating TensorRT runtime models"<<endl;
}

mtcnn::~mtcnn(){
    //delete []simpleFace_;
}

vector<struct Bbox> mtcnn::findFace(cv::Mat &image){
    firstBbox_.clear();
    firstOrderScore_.clear();
    secondBbox_.clear();
    secondBboxScore_.clear();
    thirdBbox_.clear();
    thirdBboxScore_.clear();

    struct orderScore order;
    int count = 0;

    clock_t first_time = clock();
    for (size_t i = 0; i < scales_.size(); i++) {
        int changedH = (int)ceil(image.rows*scales_.at(i));
        int changedW = (int)ceil(image.cols*scales_.at(i));
        clock_t run_first_time = clock();
        resize(image, reImage, cv::Size(changedW, changedH), 0, 0, cv::INTER_LINEAR);
        (*simpleFace_[i]).run(reImage, scales_.at(i),pnet_engine[i]);

#ifdef LOG
        run_first_time = clock() - run_first_time;
        cout<<"first model inference time is  "<<1000*(double)run_first_time/CLOCKS_PER_SEC<<endl;
#endif
        nms((*simpleFace_[i]).boundingBox_, (*simpleFace_[i]).bboxScore_, (*simpleFace_[i]).nms_threshold);

        for(vector<struct Bbox>::iterator it=(*simpleFace_[i]).boundingBox_.begin(); it!= (*simpleFace_[i]).boundingBox_.end();it++){
            if((*it).exist){
                firstBbox_.push_back(*it);
                order.score = (*it).score;
                order.oriOrder = count;
                firstOrderScore_.push_back(order);
                count++;
            }
        }
        (*simpleFace_[i]).bboxScore_.clear();
        (*simpleFace_[i]).boundingBox_.clear();
    }
    //the first stage's nms
    vector<struct Bbox> emptyBbox;
    if(count<1)return emptyBbox;
    nms(firstBbox_, firstOrderScore_, nms_threshold[0]);
    refineAndSquareBbox(firstBbox_, image.rows, image.cols,true);
#ifdef LOG
    first_time = clock() - first_time;
    cout<<"first time is  "<<1000*(double)first_time/CLOCKS_PER_SEC<<endl;
#endif
    //second stage
    count = 0;
    clock_t second_time = clock();
    for(vector<struct Bbox>::iterator it=firstBbox_.begin(); it!=firstBbox_.end();it++){
        if((*it).exist){
            cv::Rect temp((*it).y1, (*it).x1, (*it).y2-(*it).y1, (*it).x2-(*it).x1);
            cv::Mat secImage;
            resize(image(temp), secImage, cv::Size(24, 24), 0, 0, cv::INTER_LINEAR);
            transpose(secImage,secImage);
            refineNet->run(secImage,*rnet_engine);
            if(*(refineNet->score_->pdata+1)>refineNet->Rthreshold){
                memcpy(it->regreCoord, refineNet->location_->pdata, 4*sizeof(mydataFmt));
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = *(refineNet->score_->pdata+1);
                secondBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                secondBboxScore_.push_back(order);
            }
            else{
                (*it).exist=false;
            }
        }
    }
    if(count<1)return emptyBbox;
    nms(secondBbox_, secondBboxScore_, nms_threshold[1]);
    refineAndSquareBbox(secondBbox_, image.rows, image.cols,true);
    second_time = clock() - second_time;
#ifdef LOG
    cout<<"second time is  "<<1000*(double)second_time/CLOCKS_PER_SEC<<endl;
#endif
    //third stage
    count = 0;
    clock_t third_time = clock();
    for(vector<struct Bbox>::iterator it=secondBbox_.begin(); it!=secondBbox_.end();it++){
        if((*it).exist){
            cv::Rect temp((*it).y1, (*it).x1, (*it).y2-(*it).y1, (*it).x2-(*it).x1);
            cv::Mat thirdImage;
            resize(image(temp), thirdImage, cv::Size(48, 48), 0, 0, cv::INTER_LINEAR);
            transpose(thirdImage,thirdImage);
            outNet->run(thirdImage,*onet_engine);
            mydataFmt *pp=NULL;
            if(*(outNet->score_->pdata+1)>outNet->Othreshold){
                memcpy(it->regreCoord, outNet->location_->pdata, 4*sizeof(mydataFmt));
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = *(outNet->score_->pdata+1);
                pp = outNet->points_->pdata;
                for(int num=0;num<5;num++){
                    (it->ppoint)[num] = it->y1 + (it->y2 - it->y1)*(*(pp+num));
                }
                for(int num=0;num<5;num++){
                    (it->ppoint)[num+5] = it->x1 + (it->x2 - it->x1)*(*(pp+num+5));
                }
                thirdBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                thirdBboxScore_.push_back(order);
            }
            else{
                it->exist=false;
            }
        }
    }

    if(count<1)return emptyBbox;
    refineAndSquareBbox(thirdBbox_, image.rows, image.cols, true);
    nms(thirdBbox_, thirdBboxScore_, nms_threshold[2], "Min");
#ifdef LOG
    third_time = clock() - third_time;
    cout<<"third time is  "<<1000*(double)third_time/CLOCKS_PER_SEC<<endl;
#endif

    // This draws the rectangle and points on the input frame
    // for(vector<struct Bbox>::iterator it=thirdBbox_.begin(); it!=thirdBbox_.end();it++){
    //     if((*it).exist){
    //         rectangle(image, cv::Point((*it).y1, (*it).x1), cv::Point((*it).y2, (*it).x2), cv::Scalar(0,0,255), 2,8,0);
    //         for(int num=0;num<5;num++)
    //             circle(image,cv::Point((int)*(it->ppoint+num), (int)*(it->ppoint+num+5)),3,cv::Scalar(0,255,255), -1);
    //     }
    // }


    return thirdBbox_;

}
