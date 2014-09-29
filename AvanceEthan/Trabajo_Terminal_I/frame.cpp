#include "frame.h"

Frame::Frame(bool grayscale) : matrix_(), grayscale_(grayscale) {}

Frame::Frame(cv::Mat* matrix, bool grayscale) : grayscale_(grayscale) {
  SetMatrix(matrix);
}

Frame::~Frame() {
  matrix_.~Mat();
}

int Frame::Rows() const{
  return matrix_.rows;
}

int Frame::Columns() const{
  return matrix_.cols;
}

int Frame::GetPixel(int x, int y) const{
  if (grayscale_) return matrix_.at<uchar>(x, y);
  cv::Vec3b v = matrix_.at<cv::Vec3b>(x, y);
  return (v[2] << 16) + (v[1] << 8) + v[0];
}

void Frame::SetPixel(int x, int y, int rgb) {
  if (!grayscale_) {
    matrix_.at<cv::Vec3b>(x, y)[0] = rgb & 255;
    matrix_.at<cv::Vec3b>(x, y)[1] = (rgb >> 8) & 255;
    matrix_.at<cv::Vec3b>(x, y)[2] = (rgb >> 16) & 255;
  } else {
    matrix_.at<uchar>(x, y) = rgb;
  }
}

cv::Mat Frame::GetMatrix() const{
  return matrix_;
}

void Frame::SetMatrix(cv::Mat* matrix) {
  matrix_.~Mat();
  if (grayscale_) {
    cv::cvtColor(*matrix, matrix_, CV_RGB2GRAY);
  } else {
    matrix->copyTo(matrix_);
  }
}