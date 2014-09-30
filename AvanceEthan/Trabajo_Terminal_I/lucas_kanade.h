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
   cv::Mat GradientSmoothing(cv::Mat &orig);

 private:
  static const double kAlpha;
  static const int kSpatialSmoothSize;
  static const int kGradientKernelSize;
  static const double gradient[];
  static const double kernel[5][5];

  std::vector<Frame> frames_;

  void SmoothFrame(int);
};

#endif