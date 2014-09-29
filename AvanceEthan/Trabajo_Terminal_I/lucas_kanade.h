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

 private:
  static const double kAlpha;
  static const int kSpatialSmoothSize;
  static const int kGradientKernelSize;

  void SmoothFrame(int);

  std::vector<Frame> frames_;
};

#endif