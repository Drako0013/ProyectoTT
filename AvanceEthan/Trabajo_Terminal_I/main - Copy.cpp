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

    //cv::namedWindow( "LK Demo", 1 );
    //cv::setMouseCallback( "LK Demo", onMouse, 0 );

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
            // automatic initialization
            //goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, cv::Mat(), 3, 0, 0.04);
            //cornerSubPix(gray, points[1], subPixWinSize, cv::Size(-1,-1), termcrit);
            //addRemovePt = false;
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

			/*
			for( i = k = 0; i < points[1].size(); i++ ) {
                if( addRemovePt ) {
                    if( norm(point - points[1][i]) <= 10000 ) {
                        addRemovePt = false;
                        continue;
                    }
                }
				if( !status[i] )
                    continue;
				points[1][k++] = points[1][i];
                //circle( image, points[1][i], 3, cv::Scalar(0,255,0), -1, 8);
            }
            points[1].resize(k);
			*/
        }

		/*
        if( addRemovePt && points[1].size() < (size_t)MAX_COUNT )
        {
            std::vector<cv::Point2f> tmp;
            tmp.push_back(point);
            cornerSubPix( gray, tmp, winSize, cvSize(-1,-1), termcrit);
            points[1].push_back(tmp[0]);
            addRemovePt = false;
        }
		*/
        needToInit = false;
        //std::swap(points[1], points[0]);
        cv::swap(prevGray, gray);
    }
	

	/*
=======
int main(int argc, char** argv) {
  // Video file must be specified.
>>>>>>> a271a22c667e66c3cde4522f956d5f7073e758c0
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
<<<<<<< HEAD
	std::string dir = std::string(argv[1]);
	int width = 1920;
	int height = 1080;
	int orig_width;
	int orig_height;
	*/
	/*
		TestGenerator::GenerateTest("C:\\Users\\Adonais\\Desktop\\test.avi", 320, 240, 5);
		return 0;
	*/
	/*
	LucasKanade lk;
	VideoFactory lk_vf(dir + "-lk-flow.avi", width, height, vcapture.get(CV_CAP_PROP_FPS));
=======
	std::string dir(argv[2]);
	int width = 480, height = 360;
	
	/*
  LucasKanade lk; // Lucas Kanade algorithm.
	VideoFactory lk_vf(dir + "lk-flow.avi", width, height,
                     vcapture.get(CV_CAP_PROP_FPS));

>>>>>>> a271a22c667e66c3cde4522f956d5f7073e758c0
	cv::Mat vx, vy;
	std::cout << "\n\nStarting process.\n";
	int fps = (int) vcapture.get(CV_CAP_PROP_FPS);
	Frame* lk_result = new Frame(false);
<<<<<<< HEAD
	//int fileNumber = 7;
	//char* number = new char[3];
	for (int i = 0; i <= 30; ++i) {
		//sprintf(number, "%02d", fileNumber);
		//std::string fileName = std::string( argv[1] ) + std::string(number) + ".flo";
		//std::string imageName = std::string( argv[1] ) + std::string(number) + ".png";
		std::cout << "Processing frame " << i << ".\n";
		//capture = cv::imread(imageName);
=======
	
	for (int i = 0; i < 30; ++i) {
>>>>>>> a271a22c667e66c3cde4522f956d5f7073e758c0
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
			}
		}
		
    // Video result generation.
		lk_result->GetCacheOnMatrix();
		lk_vf.AddFrame(lk_result->GetMatrix());
	}
	*/
<<<<<<< HEAD
	
	/*
	HornSchunck hs;

	VideoFactory hs_vf(dir + "-hs-flow.avi", width, height, vcapture.get(CV_CAP_PROP_FPS));
=======
>>>>>>> a271a22c667e66c3cde4522f956d5f7073e758c0

	HornSchunck hs; // Horn Schunck algorithm.
	VideoFactory hs_vf(dir + "hs-flow.avi", width, height,
                     vcapture.get(CV_CAP_PROP_FPS));
  
	double* vx, *vy;
	std::cout << "\n\nStarting process.\n";
	int fps = (int) vcapture.get(CV_CAP_PROP_FPS);
	Frame* hs_result = new Frame(false);

	for (int i = 0; i < 30; ++i) {
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
			}
		}
    delete [] vx;
    delete [] vy;
    
    // Video result generation.
	  hs_result->GetCacheOnMatrix();
	  hs_vf.AddFrame(hs_result->GetMatrix());
	}
  

<<<<<<< HEAD
	hs_result->GetCacheOnMatrix();
	hs_vf.AddFrame(hs_result->GetMatrix());
	}
	printf("%d %d\n", i, fps); 
	printf("Ancho * alto = %d %d\n", orig_width, orig_height);
	*/

=======
>>>>>>> a271a22c667e66c3cde4522f956d5f7073e758c0
	/*
	SimpleFlow hs;

	VideoFactory hs_vf(dir + "-sf-flow.avi", width, height, vcapture.get(CV_CAP_PROP_FPS));

	int fps = (int)vcapture.get(CV_CAP_PROP_FPS);
	int i = 0;

	cv::Mat u, v;
	Frame* hs_result = new Frame(false);
	//std::cout << "\n\nStarting process.\n";
	for (i = 0; i < 10 * fps; ++i) {
		//std::cout << "Processing frame " << i << ".\n";

		vcapture >> capture;
		if (capture.empty()) break;

		if (!i) {
			hs_result->SetMatrix(&capture);
			hs_result->Rescale(width, height);
			hs_result->GetMatrixOnCache();
		}

		Frame* frame = new Frame(&capture);

		orig_width = frame->Columns();
	orig_height = frame->Rows();

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
	}
	printf("%d %d\n", i, fps); 
	printf("Ancho * alto = %d %d\n", orig_width, orig_height);
	*/
	return 0;
}
