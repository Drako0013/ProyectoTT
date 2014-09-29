#include "VideoFactory.h"
#include "opencv2/core/core.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

VideoFactory::VideoFactory(int width, int height, int fps){
	nombreArchivo = "salida_default.avi";
    cv::Size S = cv::Size(width, height);
	output.open(nombreArchivo, CV_FOURCC('M','S','V','C'), fps, S, true);
}

VideoFactory::VideoFactory(std::string &nombre, int width, int height, int fps){
	nombreArchivo = nombre;
	cv::Size S = cv::Size(width, height);
	output.open(nombreArchivo, CV_FOURCC('M','S','V','C'), fps, S, true);
}

void VideoFactory::agregaFrame(cv::Mat &frame){
	output << frame;
}