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

cv::Mat Frame::reduceImageSize(int desWidth, int desHeight){
	cv::Mat res = cv::Mat(desHeight, desWidth, CV_64F);
	double size, sum;
	int blockHeight = this->Rows() / desHeight;
	int blockWidth = this->Columns() / desWidth;

	for(int i = 0, ir = 0; i < this->Rows() && ir < desHeight; i = i + blockHeight, ir++){
		for(int j = 0, jr = 0; j < this->Columns() && jr < desWidth; j = j + blockWidth, jr++){
			sum = 0.0;
			size = 0.0;
			for(int ii = i; ii < i + blockHeight; ii++){
				for(int jj = j; jj < j + blockWidth; jj++){
					sum += (double)this->GetPixel(ii, jj);
					size += 1.0;
				}
			}
			res.at<double>(ir, jr) = sum / size;
		}
	}
	return res;
}