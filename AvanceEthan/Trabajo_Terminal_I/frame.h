#ifndef FRAME_H_
#define FRAME_H_

#include <vector>

#include "opencv2/imgproc/imgproc.hpp"

class Frame {
 public:
  // Constructors
  Frame(bool = false);
  Frame(cv::Mat*, bool = false);

  // Destructor
  ~Frame();

  // Size methods
  int Rows() const;
  int Columns() const;
  void Rescale(int, int);

  // Pixel methods
  int GetPixel(int, int) const;
  void SetPixel(int, int, int);

  // Matrix methods
  cv::Mat GetMatrix() const;
  void SetMatrix(cv::Mat*);

 private:
  // Member variables
  cv::Mat matrix_;
  bool grayscale_;
};

#endif