#ifndef LUCASKANADE_H_
#define LUCASKANADE_H_

#include <vector>
#include <algorithm>

#include "opencv2/imgproc/imgproc.hpp"

#include "frame.h"

class LucasKanade {
 public:
  cv::Mat AddFrame(Frame*);
  void RemoveFrame();
  cv::Mat GradientEstimationAtX();
  cv::Mat GradientEstimationAtY();
  cv::Mat GradientEstimationAtT();
  void CalculateFlow(cv::Mat&, cv::Mat&);

 private:
  static const double kAlpha;
  static const int kSpatialSmoothSize;
  static const int kGradientBegin;
  static const int kGradientEnd;
  static const int kGradient[];
  static const double kKernel[5][5];

  void SmoothFrame(int);
  void GradientSmoothing(cv::Mat&, cv::Mat&, cv::Mat&);
  
  std::vector<Frame*> frames_;
};

#endif