#ifndef HORN_SCHUNCK_H_
#define HORN_SCHUNCK_H_

#include <vector>
#include <algorithm>

#include "opencv2/imgproc/imgproc.hpp"

#include "frame.h"

class HornSchunck {
 public:
  // Frame control methods
	cv::Mat AddFrame(Frame*);
	void RemoveFrame();

  // Flow calculation method
	void CalculateFlow(double**, double**);

 private:
  // Algorithm constants
  static const double kFlowAlpha;
  static const int kFlowIterations;
	static const int kAvgKernel[3][3];
  static const int kKernelBegin;
  static const int kKernelEnd;
  static const int kFrameSize;
  
	double* LocalAverage(double*);

  // Gradient methods
	void GradientEstimations(double**, double**, double**);

  // Member variables
	std::vector<Frame*> frames;
};

#endif

