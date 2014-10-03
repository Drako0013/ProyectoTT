#include "lucas_kanade.h"

const double LucasKanade::kAlpha = 0.75;
const int LucasKanade::kSpatialSmoothSize = 5;

const int LucasKanade::kGradientEnd = 2;
const int LucasKanade::kGradientBegin = -2;
const int LucasKanade::kGradient[5] = {-1, 8, 0, -8, 1};

const double LucasKanade::kKernel[5][5] = {
  {0.00390625, 0.015625, 0.0234375, 0.015625,	0.00390625},
  {0.01562500, 0.062500, 0.0937500, 0.062500,	0.01562500},
  {0.02343750, 0.093750, 0.1406250, 0.093750,	0.02343750},
  {0.01562500, 0.062500, 0.0937500, 0.062500,	0.01562500},
  {0.00390625, 0.015625, 0.0234375, 0.015625,	0.00390625}
};

cv::Mat LucasKanade::AddFrame(Frame* frame) {
  frames_.push_back(*frame);
  if (kGradientEnd - kGradientBegin + 1 < frames_.size()) RemoveFrame();
  SmoothFrame(frames_.size() - 1);
  return frames_.back().GetMatrix();
}

void LucasKanade::RemoveFrame() {
  if (frames_.size() == 0) return;
  frames_.erase(frames_.begin());
}

void LucasKanade::SmoothFrame(int index) {
  Frame* frame = &frames_[index];

  // x-Spatial Smoothing
  int* pixels = new int[kSpatialSmoothSize];
  for (int i = 0; i < frame->Rows(); ++i) {
    double pix_sum = 0;
    std::fill(pixels, pixels + kSpatialSmoothSize, 0);
    for (int j = 0; j < frame->Columns(); ++j) {
      int this_pix = frame->GetPixel(i, j);

      pix_sum += this_pix - pixels[j % kSpatialSmoothSize];
      pixels[j % kSpatialSmoothSize] = this_pix;

      this_pix = pix_sum / std::min(kSpatialSmoothSize, j + 1);
      frame->SetPixel(i, j, this_pix);
    }
  }

  // y-Spatial Smoothing
  for (int i = 0; i < frame->Columns(); ++i) {
    double pix_sum = 0;
    std::fill(pixels, pixels + kSpatialSmoothSize, 0);
    for (int j = 0; j < frame->Rows(); ++j) {
      int this_pix = frame->GetPixel(j, i);

      pix_sum += this_pix - pixels[j % kSpatialSmoothSize];
      pixels[j % kSpatialSmoothSize] = this_pix;

      this_pix = pix_sum / std::min(kSpatialSmoothSize, j + 1);
      frame->SetPixel(j, i, this_pix);
    }
  }
  delete pixels;

  // Temporal Smoothing
  if (index > 0) {
    double kalpha = 1.0 - kAlpha;
    Frame* prev = &frames_[index - 1];

    for (int i = 0; i < frame->Rows(); ++i) {
      for (int j = 0; j < frame->Columns(); ++j) {
        int prev_pix = prev->GetPixel(i, j);
        int this_pix = frame->GetPixel(i, j);
        frame->SetPixel(i, j, kalpha * prev_pix + kAlpha * this_pix);
      }
    }
  }
}

cv::Mat LucasKanade::GradientEstimationAtX() {
  Frame* frame = &frames_[frames_.size() / 2];
  cv::Mat Ix = cv::Mat(frame->Rows(), frame->Columns(), CV_64F);
  
  for (int i = 0; i < frame->Rows(); ++i) {
    for (int j = 0; j < frame->Columns(); ++j) {
      int pix_sum = 0;
      for (int k = kGradientBegin; k <= kGradientEnd; ++k)
        if (0 <= j + k && j + k < frame->Columns())
          pix_sum += frame->GetPixel(i, j + k) * kGradient[k - kGradientBegin];
      Ix.at<double>(i, j) = static_cast<double>(pix_sum) / 12.0;
    }
  }
  return Ix;
}

cv::Mat LucasKanade::GradientEstimationAtY() {
  Frame* frame = &frames_[frames_.size() / 2];
  cv::Mat Iy = cv::Mat(frame->Rows(), frame->Columns(), CV_64F);
  
  for (int i = 0; i < frame->Rows(); ++i) {
    for (int j = 0; j < frame->Columns(); ++j) {
      int pix_sum = 0;
      for (int k = kGradientBegin; k <= kGradientEnd; ++k)
        if (0 <= i + k && i + k < frame->Rows())
          pix_sum += frame->GetPixel(i + k, j) * kGradient[k - kGradientBegin];
      Iy.at<double>(i, j) = static_cast<double>(pix_sum) / 12.0;
    }
  }
  return Iy;
}

cv::Mat LucasKanade::GradientEstimationAtT() {
  int index = frames_.size() / 2;
  cv::Mat It = cv::Mat(frames_[index].Rows(), frames_[index].Columns(), CV_64F);
  if (frames_.size() < kGradientEnd - kGradientBegin + 1) return It;

  for (int i = 0; i < frames_[index].Rows(); ++i) {
    for (int j = 0; j < frames_[index].Columns(); ++j) {
      int pix_sum = 0;
      for (int k = kGradientBegin; k <= kGradientEnd; ++k)
        if (0 <= index + k && index + k < frames_.size())
          pix_sum += frames_[index + k].GetPixel(i, j) * kGradient[k - kGradientBegin];
      It.at<double>(i, j) = static_cast<double>(pix_sum) / 12.0;
    }
  }
  return It;
}

// SPRINT: OPTIMIZE THE FOLLOWING

void LucasKanade::GradientSmoothing(cv::Mat &gradX, cv::Mat &gradY, cv::Mat &gradT){
	cv::Mat gradEX = this->GradientEstimationAtX();
	cv::Mat gradEY = this->GradientEstimationAtY();
	cv::Mat gradET = this->GradientEstimationAtT();
	gradEX.copyTo(gradX);
	gradEY.copyTo(gradY);
	gradET.copyTo(gradT);
	const int rows = gradEX.rows;
	const int cols = gradEX.cols;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			double pix_sum_x, pix_sum_y, pix_sum_t;
			pix_sum_x = pix_sum_y = pix_sum_t = 0;
			for (int k = -2; k <= 2; k++) {
				for (int l = -2; l <= 2; l++) {
					if (i + k >= 0 && i + k < rows && j + l >= 0 && j + l < cols) {
						pix_sum_x += gradEX.at<double>(i + k, j + l) * kKernel[k + 2][l + 2];
						pix_sum_y += gradEY.at<double>(i + k, j + l) * kKernel[k + 2][l + 2];
						pix_sum_t += gradET.at<double>(i + k, j + l) * kKernel[k + 2][l + 2];
					}
				}
			}
			gradX.at<double>(i, j) = pix_sum_x;
			gradY.at<double>(i, j) = pix_sum_y;
			gradT.at<double>(i, j) = pix_sum_t;
		}
	}
}

void LucasKanade::CalculateFlow(cv::Mat &velX, cv::Mat &velY) {
	cv::Mat gradX, gradY, gradT;
	this->GradientSmoothing(gradX, gradY, gradT);
	const int rows = gradX.rows, cols = gradX.cols;
	velX = cv::Mat(rows, cols, CV_64F);
	velY = cv::Mat(rows, cols, CV_64F);
	int i, j;
	double ix, iy, it, ixx, ixy, iyy, ixt, iyt;
	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			ix = (double)gradX.at<double>(i, j);
			iy = (double)gradY.at<double>(i, j);
			it = (double)gradT.at<double>(i, j);
			ixx = ix * ix;
			ixy = ix * iy;
			iyy = iy * iy;
			ixt = ix * it;
			iyt = iy * it;
			double a[2][2] = { { ixx, ixy }, { ixy, iyy } };
			double b[2][1] = { { ixt }, { iyt } };
			cv::Mat velVector = cv::Mat(2, 2, CV_64F, a).inv() * cv::Mat(2, 1, CV_64F, b);
			velX.at<double>(i, j) = velVector.at<double>(0, 0);
			velY.at<double>(i, j) = velVector.at<double>(1, 0);
		}
	}
}