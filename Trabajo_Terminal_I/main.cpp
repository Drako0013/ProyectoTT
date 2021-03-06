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

int main(int argc, char** argv) {
	if (argc < 3) {
		std::cout << "Video input file and output directory required.";
		return 0;
	}

	cv::VideoCapture vcapture;

	vcapture.open(argv[1]);
	if (!vcapture.isOpened()) {
		std::cout << "Could not initialize capturing.\n";
		return 0;
	}

	cv::Mat capture;
	std::string dir = std::string(argv[1]);

	int width = 320;
	int height = 240;

	TestGenerator::GenerateTest("C:\\Users\\Drako\\Desktop\\test.avi", 320, 240, 30);

	return 0;

	/*LucasKanade lk;

	VideoFactory lk_vf(dir + "-lk-flow.avi", width, height, vcapture.get(CV_CAP_PROP_FPS));

	cv::Mat vx, vy;
	std::cout << "\n\nStarting process.\n";

	Frame* lk_result = new Frame(false);
	for (int i = 0; true; ++i) {
	std::cout << "Processing frame " << i << ".\n";

	vcapture >> capture;
	if (capture.empty()) break;

	if (!i) {
	lk_result->SetMatrix(&capture);
	lk_result->Rescale(width, height);
	lk_result->GetMatrixOnCache();
	}

	Frame* frame = new Frame(&capture);
	frame->Rescale(width, height);
	frame->GetMatrixOnCache();
	lk.AddFrame(frame);

	lk.CalculateFlow(vx, vy);

	for (int x = 0; x < height; ++x) {
	double* ptr_vx = vx.ptr<double>(x);
	double* ptr_vy = vy.ptr<double>(x);
	for (int y = 0; y < width; ++y, ++ptr_vx, ++ptr_vy) {
	double X = *ptr_vx, Y = *ptr_vy;

	int hor_color = 0;
	double intensity = std::min(std::abs(Y) / kIntensity, 1.0);
	for (int k = 0; k <= 16; k += 8) {
	int color = (Y < 0)? (kLeft >> k) & 255: (kRight >> k) & 255;
	hor_color |= static_cast<int>(color * intensity) << k;
	}
	int ver_color = 0;
	intensity = std::min(std::abs(X) / kIntensity, 1.0);
	for (int k = 0; k <= 16; k += 8) {
	int color = (X < 0)? (kUp >> k) & 255: (kDown >> k) & 255;
	ver_color |= static_cast<int>(color * intensity) << k;
	}

	lk_result->SetPixel(x, y, hor_color);
	}
	}

	lk_result->GetCacheOnMatrix();
	lk_vf.AddFrame(lk_result->GetMatrix());
	}*/

	/*
	HornSchunck hs;

	VideoFactory hs_vf(dir + "-hs-flow.avi", width, height, vcapture.get(CV_CAP_PROP_FPS));

	double* u = NULL, *v = NULL;
	Frame* hs_result = new Frame(false);
	std::cout << "\n\nStarting process.\n";
	for (int i = 0; 1 < 1000; ++i) {
	std::cout << "Processing frame " << i << ".\n";

	vcapture >> capture;
	if (capture.empty()) break;

	if (!i) {
	hs_result->SetMatrix(&capture);
	hs_result->Rescale(width, height);
	hs_result->GetMatrixOnCache();
	}

	Frame* frame = new Frame(&capture);
	frame->Rescale(width, height);
	frame->GetMatrixOnCache();
	hs.AddFrame(frame);

	if (i % 1 == 0) {
	delete [] u;
	delete [] v;
	hs.CalculateFlow(&u, &v);
	}

	int rows = frame->Rows();
	int cols = frame->Columns();
	double* ptr_x = u, *ptr_y = v;
	for (int x = 0; x < rows; ++x) {
	for (int y = 0; y < cols; ++y, ++ptr_x, ++ptr_y) {
	double X = *ptr_x, Y = *ptr_y;

	int hor_color = 0;
	double intensity = std::min(std::abs(Y) / kIntensity, 1.0);
	for (int k = 0; k <= 16; k += 8) {
	int color = (Y < 0)? (kLeft >> k) & 255: (kRight >> k) & 255;
	hor_color |= static_cast<int>(color * intensity) << k;
	}
	int ver_color = 0;
	intensity = std::min(std::abs(X) / kIntensity, 1.0);
	for (int k = 0; k <= 16; k += 8) {
	int color = (X < 0)? (kUp >> k) & 255: (kDown >> k) & 255;
	ver_color |= static_cast<int>(color * intensity) << k;
	}

	hs_result->SetPixel(x, y, hor_color | ver_color);
	}
	}

	hs_result->GetCacheOnMatrix();
	hs_vf.AddFrame(hs_result->GetMatrix());
	}
	*/

	/*
	SimpleFlow hs;

	VideoFactory hs_vf(dir + "-sf-flow.avi", width, height, vcapture.get(CV_CAP_PROP_FPS));

	cv::Mat u, v;
	Frame* hs_result = new Frame(false);
	std::cout << "\n\nStarting process.\n";
	for (int i = 0; i < 24 * 3; ++i) {
		std::cout << "Processing frame " << i << ".\n";

		vcapture >> capture;
		if (capture.empty()) break;

		if (!i) {
			hs_result->SetMatrix(&capture);
			hs_result->Rescale(width, height);
			hs_result->GetMatrixOnCache();
		}

		Frame* frame = new Frame(&capture);
		frame->Rescale(width, height);
		frame->GetMatrixOnCache();
		hs.AddFrame(frame);

		if (i % 1 == 0) {
			hs.CalculateFlow(u, v);
		}

		if (i == 0) {
			continue;
		}

		int rows = u.rows;
		int cols = u.cols;

		for (int x = 0; x < rows; ++x) {
			double* ptr_x = u.ptr<double>(x);
			double* ptr_y = v.ptr<double>(x);
			for (int y = 0; y < cols; ++y, ++ptr_x, ++ptr_y) {
				double X = *ptr_x, Y = *ptr_y;

				int hor_color = 0;
				double intensity = std::min(std::abs(Y) / kIntensity, 1.0);
				for (int k = 0; k <= 16; k += 8) {
					int color = (Y < 0) ? (kLeft >> k) & 255 : (kRight >> k) & 255;
					hor_color |= static_cast<int>(color * intensity) << k;
				}
				int ver_color = 0;
				intensity = std::min(std::abs(X) / kIntensity, 1.0);
				for (int k = 0; k <= 16; k += 8) {
					int color = (X < 0) ? (kUp >> k) & 255 : (kDown >> k) & 255;
					ver_color |= static_cast<int>(color * intensity) << k;
				}

				hs_result->SetPixel(x, y, hor_color | ver_color);
			}
		}

		hs_result->GetCacheOnMatrix();
		hs_vf.AddFrame(hs_result->GetMatrix());
	}*/

	//TestGenerator tG;
	//tG.GenerateTest("C:/Users/Drako//Desktop//Pruebas//pGenerada.avi", 10, 20, 30);

	return 0;
}