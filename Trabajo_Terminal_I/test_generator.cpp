#include "test_generator.h"
#include "video_factory.h"
#include "frame.h"

#include <cstdlib>
#include <cstdio>
#include <iostream>

TestGenerator::TestGenerator()
{
}


TestGenerator::~TestGenerator()
{
}

void TestGenerator::GenerateTest(std::string dir, int width, int height, int speed)
{

	int f, i, j, sqx = 20, sqy = 20, sqs = 50;

	VideoFactory vf(dir, width, height, 24);

	cv::Mat v(height, width, CV_8U);

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			v.at<uchar>(i, j) = 0;
		}
	}

	for (i = sqx; i < sqx + sqs; i++) {
		for (j = sqy; j < sqy + sqs; j++) {
			v.at<uchar>(i, j) = 255;
		}
	}

	for (f = 0; f < 24 * 3; f++) {
		std::cout << "Processing frame " << (f + 1) << ".\n";
		for (i = sqx; i < sqx + sqs; i++) {
			for (j = sqy; j < sqy + speed; j++) {
				if (j < width) {
					v.at<uchar>(i, j) = 0;
				}
				if (j + sqs < width) {
					v.at<uchar>(i, j + sqs) = 255;
				}
			}
		}
		sqy += speed;
		vf.AddFrame(v);
	}
}
