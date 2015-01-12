#ifndef FRAME_H_
#define FRAME_H_

#include <vector>

#include "opencv2/imgproc/imgproc.hpp"

class Frame {
 public:
  // Constructors
  Frame(bool = true);
  Frame(cv::Mat*, bool = true);

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

  // Cache methods
  void GetMatrixOnCache();
  void GetCacheOnMatrix();

 private:
  void ResizeCache(int, int);

  // Member variables
  cv::Mat matrix;
  int* matrix_cache;
  bool grayscale, cached;
};

#endif