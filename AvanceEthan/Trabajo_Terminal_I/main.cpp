#include <ctime>
#include <string>
#include <iostream>

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "frame.h"
#include "simple_flow.h"
#include "horn_schunck.h"
#include "lucas_kanade.h"
#include "video_factory.h"
#include "test_generator.h"

const int kUp = 0x00FFFF;    // LIGHT BLUE
const int kDown = 0x00FF00;  // GREEN
const int kLeft = 0xFF0000;  // RED
const int kRight = 0xFFFF00; // YELLOW
const double kIntensity = 4.7;

int main(int argc, char** argv) {
  // Video file must be specified.
	if (argc < 3) {
		std::cout << "Video input file and output directory required.";
		return 0;
	}
  
  // Capturing video from file.
	cv::VideoCapture vcapture;
	vcapture.open(argv[1]);

  // If something fail, abort mission.
	if (!vcapture.isOpened()) {
		std::cout << "Could not initialize capturing.\n";
		return 0;
	}

	cv::Mat capture;
	std::string dir(argv[2]);
	int width = 352, height = 288;
	
  
	LucasKanade lk; // Lucas Kanade algorithm.
	VideoFactory lk_vf(dir + "lk-flow.avi", width, height,
                     vcapture.get(CV_CAP_PROP_FPS));

	cv::Mat vx, vy;
	std::cout << "\n\nStarting process.\n";
	int fps = (int) vcapture.get(CV_CAP_PROP_FPS);
	Frame* lk_result = new Frame(false);
	
	for (int i = 0; i < 30; ++i) {
		char fileName[30];
		sprintf(fileName, "salida_lkown%d.txt", i);
		FILE *out = fopen(fileName, "w");
		vcapture >> capture;
		if (capture.empty()) break;
    std::cout << "\nCaptured frame " << i << ".\n";

		std::cout << "Processing frame " << i << "...\n";
    std::clock_t start_time = std::clock();

		if (i == 0) {
			lk_result->SetMatrix(&capture);
			lk_result->Rescale(width, height);
			lk_result->GetMatrixOnCache();
		}

    // Frame treatment for better performance.
		Frame* frame = new Frame(&capture);
		frame->Rescale(width, height);
		frame->GetMatrixOnCache();
		lk.AddFrame(frame);

		lk.CalculateFlow(vx, vy);
    // Print processing time for this frame.
    std::cout << "Finished processing frame " << i << ".\n";
    std::clock_t ptime = (std::clock() - start_time) / (double) (CLOCKS_PER_SEC / 1000);
    std::cout << "Processing time: " << ptime << " ms.\n";

		for (int x = 0; x < height; ++x) {
			double* ptr_vx = vx.ptr<double>(x);
			double* ptr_vy = vy.ptr<double>(x);
			for (int y = 0; y < width; ++y, ++ptr_vx, ++ptr_vy) {
				double X = *ptr_vx, Y = *ptr_vy;

				int hor_color = 0;
        // Color scheme for the horizontal movement.
				double intensity = std::min(std::abs(Y) / kIntensity, 1.0);
				for (int k = 0; k <= 16; k += 8) {
					int color = (Y < 0)? (kLeft >> k) & 255: (kRight >> k) & 255;
					hor_color |= static_cast<int>(color * intensity) << k;
				}

				int ver_color = 0;
        // Color scheme for the vertical movement.
				intensity = std::min(std::abs(X) / kIntensity, 1.0);
				for (int k = 0; k <= 16; k += 8) {
					int color = (X < 0)? (kUp >> k) & 255: (kDown >> k) & 255;
					ver_color |= static_cast<int>(color * intensity) << k;
				}

        // Set the pixel for video result.
				lk_result->SetPixel(x, y, hor_color | ver_color);
				fprintf(out, "(%.2lf,%.2lf)", Y, X);
			}
			fprintf(out, "\n");
		}
		fclose(out);
		
    // Video result generation.
		lk_result->GetCacheOnMatrix();
		lk_vf.AddFrame(lk_result->GetMatrix());
	}
	

  /*
	HornSchunck hs; // Horn Schunck algorithm.
	VideoFactory hs_vf(dir + "hs-flow.avi", width, height,
                     vcapture.get(CV_CAP_PROP_FPS));

	double* vx, *vy;
	std::cout << "\n\nStarting process.\n";
	int fps = (int) vcapture.get(CV_CAP_PROP_FPS);
	Frame* hs_result = new Frame(false);
	
	for (int i = 0; i < 30; ++i) {
		char fileName[30];
		sprintf(fileName, "salida_hsown%d.txt", i);
		FILE *out = fopen(fileName, "w");
		vcapture >> capture;
		if (capture.empty()) break;
    std::cout << "\nCaptured frame " << i << ".\n";

		std::cout << "Processing frame " << i << "...\n";
    std::clock_t start_time = std::clock();

		if (i == 0) {
			hs_result->SetMatrix(&capture);
			hs_result->Rescale(width, height);
			hs_result->GetMatrixOnCache();
		}

    // Frame treatment for better performance.
		Frame* frame = new Frame(&capture);
		frame->Rescale(width, height);
		frame->GetMatrixOnCache();
		hs.AddFrame(frame);

		hs.CalculateFlow(&vx, &vy);
    // Print processing time for this frame.
    std::cout << "Finished processing frame " << i << ".\n";
    std::clock_t ptime = (std::clock() - start_time) / (double) (CLOCKS_PER_SEC / 1000);
    std::cout << "Processing time: " << ptime << " ms.\n";

    double* ptr_vx = vx, *ptr_vy = vy;
		for (int x = 0; x < height; ++x) {
			for (int y = 0; y < width; ++y, ++ptr_vx, ++ptr_vy) {
				double X = *ptr_vx, Y = *ptr_vy;

				int hor_color = 0;
        // Color scheme for the horizontal movement.
				double intensity = std::min(std::abs(Y) / kIntensity, 1.0);
				for (int k = 0; k <= 16; k += 8) {
					int color = (Y < 0)? (kLeft >> k) & 255: (kRight >> k) & 255;
					hor_color |= static_cast<int>(color * intensity) << k;
				}

				int ver_color = 0;
        // Color scheme for the vertical movement.
				intensity = std::min(std::abs(X) / kIntensity, 1.0);
				for (int k = 0; k <= 16; k += 8) {
					int color = (X < 0)? (kUp >> k) & 255: (kDown >> k) & 255;
					ver_color |= static_cast<int>(color * intensity) << k;
				}

        // Set the pixel for video result.
				hs_result->SetPixel(x, y, hor_color | ver_color);
				fprintf(out, "(%.2lf,%.2lf)", Y, X);
			}
			fprintf(out, "\n");
		}
		fclose(out);

    
		
    // Video result generation.
		hs_result->GetCacheOnMatrix();
		hs_vf.AddFrame(hs_result->GetMatrix());
	}
	*/

	return 0;
}