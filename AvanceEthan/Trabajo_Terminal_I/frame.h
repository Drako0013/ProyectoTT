#ifndef FRAME_H_
#define FRAME_H_

#include <vector>

#include "opencv2/imgproc/imgproc.hpp"

class Frame {
 public:
  Frame(bool = false);
  Frame(cv::Mat*, bool = false);
  ~Frame();

  int Rows() const;
  int Columns() const;
  int GetPixel(int, int) const;
  void SetPixel(int, int, int);
  cv::Mat GetMatrix() const;
  void SetMatrix(cv::Mat*);

 private:
  cv::Mat matrix_;
  bool grayscale_;
};

#endif