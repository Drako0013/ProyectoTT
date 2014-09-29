// CFramer.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <ctype.h>

#include "constants.h"
#include "VideoFactory.h"

using namespace cv;
using namespace std;

Point2f point;
bool addRemovePt = false;

void smoothe(cv::Mat &prevFrame, cv::Mat &newFrame, cv::Mat &dest){
	int width = prevFrame.cols;
	int height = prevFrame.rows;
	prevFrame.copyTo(dest);
	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			dest.at<uchar>(i, j) = ( (1.0f - alfa) * (float)prevFrame.at<uchar>(i, j) ) + ( alfa * (float)newFrame.at<uchar>(i, j) );
		}
	}
}

void estimateGradient(cv::Mat &orig, cv::Mat &gradX, cv::Mat &gradY, std::vector<cv::Mat> &grad, cv::Mat &gradT){
	int width = orig.cols;
	int height = orig.rows;
	orig.copyTo(gradX);
	orig.copyTo(gradY);
	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			gradX.at<uchar>(i, j) = 0;
			gradY.at<uchar>(i, j) = 0;
			for(int k = -2; k <= 2; k++){
				if(i + k >= 0 && i + k < height)
					gradX.at<uchar>(i, j) += orig.at<uchar>(i + k, j) * gradient[k + 2];
				if(j + k >= 0 && j + k < width)
					gradY.at<uchar>(i, j) += orig.at<uchar>(i, j + k) * gradient[k + 2];
			}
			gradX.at<uchar>(i, j) /= 12;
			gradY.at<uchar>(i, j) /= 12;
		}
	}
	if( !(grad[2].empty()) ){
		for(int i = 0; i < height; i++){
			for(int j = 0; j < width; j++){
				gradT.at<uchar>(i, j) = 0;
				for(int k = 0; k < 5; k++){
					if( !(grad[k].empty()) ){
						gradT.at<uchar>(i, j) += grad[i].at<uchar>(i, j) * gradient[k];
					}
				}
				gradT.at<uchar>(i, j) /= 12;
			}
		}
	}
	
}

void smootheGradient(cv::Mat orig, cv::Mat dest){
	int width = orig.cols;
	int height = orig.rows;
	orig.copyTo(dest);
	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			dest.at<uchar>(i, j) = 0;
			for(int k = 0; k <= 5; k++){
				for(int l = 0; l <= 5; l++){
					if( (i + k >= 0 && i + k) < height || (j + l >= 0 && j + l) ){
						dest.at<uchar>(i, j) += orig.at<uchar>(i + k, j + l) * kernel[k][l];
					}
				}
			}
		}
	}
}

int _tmain(int argc, char* argv[]){
    VideoCapture cap;
    TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
    Size subPixWinSize(10,10), winSize(31,31);

    const int MAX_COUNT = 500;
    bool needToInit = false;
    bool nightMode = false;

	cap.open("C:\\Users\\Drako\\Desktop\\nope.avi");
    if( !cap.isOpened() ){
        cout << "Could not initialize capturing...\n";
        return 0;
    }
	VideoFactory vF(std::string("C:\\Users\\Drako\\Desktop\\salida.avi"), 
						(int) cap.get(CV_CAP_PROP_FRAME_WIDTH), 
						(int) cap.get(CV_CAP_PROP_FRAME_HEIGHT),
						cap.get(CV_CAP_PROP_FPS));
	//int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));     // Get Codec Type- Int form
    // Transform from int to char via Bitwise operators
    //Size S = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
   //               (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    //VideoWriter outputVideo;                                        // Open the output
    //outputVideo.open("C:\\Users\\Drako\\Desktop\\nope2.avi", CV_FOURCC('M','S','V','C'), cap.get(CV_CAP_PROP_FPS), S, true);

    Mat gray, prevGray, image, dest;
    vector<Point2f> points[2];
    for(int i = 0; ; i++){
		cout << "Evaluando frame #" << i << endl;
        Mat frame;
        cap >> frame;
		if( frame.empty() )
            break;
        frame.copyTo(image);
		
		cvtColor(image, gray, COLOR_BGR2GRAY);
		/*
		if(i != 0){
			smoothe(prevGray, gray, dest);
			string s = "C:\\Users\\Drako\\Desktop\\images\\smoothed" + to_string(i) + ".bmp";
			imwrite(s, dest);
		}
        cv::swap(prevGray, gray);
		*/
		vF.agregaFrame(gray);
		//outputVideo << gray;
    }
	return 0;
}

