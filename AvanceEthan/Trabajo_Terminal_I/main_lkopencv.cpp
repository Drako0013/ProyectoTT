#include <stdio.h>
#include <string>
#include <iostream>

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "frame.h"
#include "lucas_kanade.h"
#include "video_factory.h"
#include "horn_schunck.h"
#include "simple_flow.h"
#include "test_generator.h"

const int kUp = 0x00FFFF; // LIGHT BLUE
const int kDown = 0x00FF00; // GREEN
const int kLeft = 0xFF0000; // RED
const int kRight = 0xFFFF00; // YELLOW
const double kIntensity = 4.7;

cv::Point2f point;
bool addRemovePt = false;

int main(int argc, char** argv) {
	cv::VideoCapture cap;
    cv::TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
    cv::Size subPixWinSize(10,10), winSize(31,31);

    const int MAX_COUNT = 500;
    bool needToInit = true;

    if( argc == 2 )
        cap.open(argv[1]);

    if( !cap.isOpened() ) {
        std::cout << "Could not initialize capturing...\n";
        return 0;
    }

    cv::Mat gray, prevGray, image;
    std::vector<cv::Point2f> points[2];

    for(int nframe = 0; ; nframe++)
    {
		std::cout << "Processing " << nframe << std::endl;
        cv::Mat frame;
        cap >> frame;
        if( frame.empty() )
            break;

        frame.copyTo(image);
        cvtColor(image, gray, cv::COLOR_BGR2GRAY);
		
        if( needToInit ){
			points[0].clear();
			for(int i = 0; i < frame.rows; i++){
				for(int j = 0; j < frame.cols; j++){
					points[0].push_back(cv::Point2f((float)j, (float)i)); 
				}
			}
        }
        if( !(points[0].empty()) ){
            std::vector<uchar> status;
            std::vector<float> err;
            if(prevGray.empty())
                gray.copyTo(prevGray);
            calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
                                 3, termcrit, 0, 0.001);
            size_t i, j;
            
			char fileName[30];
			sprintf(fileName, "salida_opencv%d.txt", nframe);
			FILE *out = fopen(fileName, "w");
			for(i = j = 0; i < points[0].size() && j < points[1].size(); i++){
				if(i && i % frame.cols == 0) fprintf(out, "\n");

				if( status[i] == 1 || err[i] == 0 ){
					fprintf(out, "(%.2f, %.2f) ", points[1][j].y - points[0][i].y,  points[1][j].x - points[0][i].x);
					j++;
				} else {
					std::cout << err[i] << std::endl;
					fprintf(out, "(%.2f, %.2f) ", -1.0, -1.0);
				}
			}
			fclose(out);

        }

		needToInit = false;
        //std::swap(points[1], points[0]);
        cv::swap(prevGray, gray);
    }
}