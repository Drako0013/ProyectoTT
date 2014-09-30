#include "lucas_kanade.h"

const double LucasKanade::kAlpha = 0.75;
const int LucasKanade::kSpatialSmoothSize = 5;
const int LucasKanade::kGradientKernelSize = 5;
const double LucasKanade::gradient[] = {-1.0f, 8.0f, 0.0f, -8.0f, 1.0f};
const double LucasKanade::kernel[5][5] = {
	{0.00390625f,	0.015625f,	0.0234375f,	0.015625f,	0.00390625f},
	{0.015625f,		0.0625f,	0.09375f,	0.0625f,	0.015625f},
	{0.0234375f,	0.09375f,	0.140625f,	0.09375f,	0.0234375f},
	{0.015625f,		0.0625f,	0.09375f,	0.0625f,	0.015625f},
	{0.00390625f,	0.015625f,	0.0234375f,	0.015625f,	0.00390625f}
};

cv::Mat LucasKanade::AddFrame(Frame* frame) {
  frames_.push_back(*frame);
  if (frames_.size() > kGradientKernelSize) RemoveFrame();
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
    for (int j = 0; j < frame->Columns(); ++j) {
      int this_pix = frame->GetPixel(i, j);

      pix_sum += this_pix;
      if(kSpatialSmoothSize <= j)
        pix_sum -= pixels[j % kSpatialSmoothSize];
      pixels[j % kSpatialSmoothSize] = this_pix;

      this_pix = pix_sum / std::min(kSpatialSmoothSize, j + 1);
      frame->SetPixel(i, j, this_pix);
    }
  }

  // y-Spatial Smoothing
  for (int i = 0; i < frame->Columns(); ++i) {
    double pix_sum = 0;
    for (int j = 0; j < frame->Rows(); ++j) {
      int this_pix = frame->GetPixel(j, i);

      pix_sum += this_pix;
      if(kSpatialSmoothSize <= j)
        pix_sum -= pixels[j % kSpatialSmoothSize];
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

cv::Mat LucasKanade::GradientEstimationAtX(){
	Frame* frame = &frames_.back();
	cv::Mat frame_x;
	frame->GetMatrix().copyTo(frame_x);

	// Ix estimation
	for (int i = 0; i < frame->Rows(); ++i) {
		for (int j = 0; j < frame->Columns(); ++j) {
			double pix_sum_x = 0;
			for(int k = -2; k <= 2; k++){
				if( (i + k) >= 0 && (i + k) < frame->Rows())
					pix_sum_x += frame->GetPixel(i + k, j) * gradient[k + 2];
			}
			frame_x.at<uchar>(i, j) = pix_sum_x / 12.0;
		}
	}

	return frame_x;
}

cv::Mat LucasKanade::GradientEstimationAtY(){
	Frame* frame = &frames_.back();
	cv::Mat frame_y;
	frame->GetMatrix().copyTo(frame_y);

	// Iy estimation
	for (int i = 0; i < frame->Rows(); ++i) {
		for (int j = 0; j < frame->Columns(); ++j) {
			double pix_sum_y = 0;
			for(int k = -2; k <= 2; k++){
				if(j + k >= 0 && j + k < frame->Columns())
					pix_sum_y += frame->GetPixel(i, j + k) * gradient[k + 2];
			}
			frame_y.at<uchar>(i, j) = pix_sum_y / 12.0;
		}
	}
	return frame_y;
}

cv::Mat LucasKanade::GradientEstimationAtT(){
	Frame* frame = &frames_.back();
	cv::Mat frame_t;
	frame->GetMatrix().copyTo(frame_t);

	// It estimation
	if(frames_.size() > 2){
		for(int i = 0; i < frame->Rows(); i++){
			for(int j = 0; j < frame->Columns(); j++){
				double pix_sum_t = 0;
				for(int k = 0; k < 5; k++){
					if( frames_.size() > k ){
						Frame* frame = &frames_[k];
						pix_sum_t += frame->GetPixel(i, j) * gradient[k];
					}
				}
				frame_t.at<uchar>(i, j) = pix_sum_t / 12.0;
			}
		}
	}
	return frame_t;
}

cv::Mat LucasKanade::GradientSmoothing(cv::Mat &orig){
	int width = orig.cols;
	int height = orig.rows;
	cv::Mat dest;
	orig.copyTo(dest);
	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			dest.at<uchar>(i, j) = 0;
			for(int k = 0; k < 5; k++){
				for(int l = 0; l < 5; l++){
					if( (i + k) < height && (j + l) < width ){
						dest.at<uchar>(i, j) += orig.at<uchar>(i + k, j + l) * kernel[k][l];
					}
				}
			}
		}
	}
	return dest;
}