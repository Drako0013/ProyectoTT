#include "frame.h"

Frame::Frame(bool grayscale)
  : matrix_(), grayscale_(grayscale) {}

Frame::Frame(cv::Mat* matrix, bool grayscale)
  : grayscale_(grayscale) {
  SetMatrix(matrix);
}

Frame::~Frame() {}

int Frame::Rows() const {
  return matrix_.rows;
}

int Frame::Columns() const {
  return matrix_.cols;
}

void Frame::Rescale(int width, int height) {
  cv::Mat matrix_copy;
  matrix_.copyTo(matrix_copy);
  cv::resize(matrix_copy, matrix_,
             cv::Size(width, height));
}

int Frame::GetPixel(int x, int y) const {
  if (grayscale_) return matrix_.at<uchar>(x, y);

  // Casting RGB vector to simple integer
  cv::Vec3b v = matrix_.at<cv::Vec3b>(x, y);
  return (v[2] << 16) + (v[1] << 8) + v[0];
}

void Frame::SetPixel(int x, int y, int pixel) {
  if (grayscale_) {
    // Grayscale for non-colored images
    matrix_.at<uchar>(x, y) = pixel;
  } else {
    // Color space for colored images is RGB
    matrix_.at<cv::Vec3b>(x, y)[0] = pixel & 255;
    matrix_.at<cv::Vec3b>(x, y)[1] = (pixel >> 8) & 255;
    matrix_.at<cv::Vec3b>(x, y)[2] = (pixel >> 16) & 255;
  }
}

cv::Mat Frame::GetMatrix() const {
  return matrix_;
}

// Assuming BGR as default color space of Mat
void Frame::SetMatrix(cv::Mat* matrix) {
  if (grayscale_) {
    // Transformation to grayscale Mat
    cv::cvtColor(*matrix, matrix_, CV_BGR2GRAY);
  } else {
    // Transformation to RGB standard Mat
    cv::cvtColor(*matrix, matrix_, CV_BGR2RGB);
  }
}