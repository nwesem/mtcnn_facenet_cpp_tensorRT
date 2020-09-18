#include "videoStreamer.h"

VideoStreamer::VideoStreamer(int nmbrDevice, int videoWidth, int videoHeight, int frameRate, bool isCSICam) {
	if(isCSICam) {
    	m_videoWidth = videoWidth;
    	m_videoHeight = videoHeight;
	m_frameRate = frameRate;

    	std::string pipeline = gstreamer_pipeline(videoWidth, videoHeight, videoWidth, 
					videoHeight, frameRate);
	std::cout << "Using pipeline: \n\t" << pipeline << "\n";
	 
	m_capture = new cv::VideoCapture(pipeline, cv::CAP_GSTREAMER);
	if(!m_capture->isOpened()) {
		std::cerr << "Failed to open CSI camera."<< std::endl;
	}
	}
	else {
    	m_capture = new cv::VideoCapture(nmbrDevice);
    	if (!m_capture->isOpened()){
        	//error in opening the video input
        	std::cerr << "Failed to open USB camera." << std::endl;
    	}
    	m_videoWidth = videoWidth;
    	m_videoHeight = videoHeight;
    	m_capture->set(cv::CAP_PROP_FRAME_WIDTH, m_videoWidth);
    	m_capture->set(cv::CAP_PROP_FRAME_HEIGHT, m_videoHeight);
	}
}

VideoStreamer::VideoStreamer(std::string filename, int videoWith, int videoHeight) {
    m_capture = new cv::VideoCapture(filename);
    if (!m_capture->isOpened()){
        //error in opening the video input
        std::cerr << "Unable to open file!" << std::endl;
    }
    // ToDo set filename width+height doesn't work with m_capture.set(...)
}

void VideoStreamer::setResolutionDevice(int width, int height) {
    m_videoWidth = width;
    m_videoHeight = height;
    m_capture->set(cv::CAP_PROP_FRAME_WIDTH, m_videoWidth);
    m_capture->set(cv::CAP_PROP_FRAME_HEIGHT, m_videoHeight);
}

void VideoStreamer::setResoltionFile(int width, int height) {
    // ToDo set resolution for input files
}

void VideoStreamer::getFrame(cv::Mat &frame) {
    *m_capture >> frame;
}

void VideoStreamer::assertResolution() {
    // currently wrong, since m_capture->get returns max/default width, height
    // but a function like this would be good to ensure good performance
    assert(m_videoWidth == m_capture->get(cv::CAP_PROP_FRAME_WIDTH));
    assert(m_videoHeight == m_capture->get(cv::CAP_PROP_FRAME_HEIGHT));
}

std::string VideoStreamer::gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int frameRate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(frameRate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

void VideoStreamer::release() {
	m_capture->release();
}

VideoStreamer::~VideoStreamer() {
	
}
