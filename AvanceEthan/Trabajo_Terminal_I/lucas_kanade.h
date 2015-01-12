#ifndef LUCASKANADE_H_
#define LUCASKANADE_H_

#include <vector>
#include <algorithm>

#include "opencv2/imgproc/imgproc.hpp"

#include "frame.h"

class LucasKanade {
 public:
  // Frame control methods
  cv::Mat AddFrame(Frame*);
  void RemoveFrame();

  // Flow calculation method
  void CalculateFlow(cv::Mat&, cv::Mat&);

 private:
  // Algorithm constants 
  static const double kAlpha;
  static const int kSpatialSmoothSize;
  static const int kGradientBegin;
  static const int kGradientEnd;
  static const int kGradient[];
  static const double kKernel[5][5];

  // Smoothing methods
  void SmoothFrame(int);
  void GradientSmoothing(double**, double**, double**);
  
  // Gradient methods
  double* GradientEstimationAtX();
  double* GradientEstimationAtY();
  double* GradientEstimationAtT();
  
  // Member variables
  std::vector<Frame*> frames;
};

#endif