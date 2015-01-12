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
  frames.push_back(frame);
  int total_frames = kGradientEnd - kGradientBegin + 1;
  if (total_frames < frames.size()) RemoveFrame();
  SmoothFrame(frames.size() - 1);
  return frames.back()->GetMatrix();
}

void LucasKanade::RemoveFrame() {
  if (frames.size() == 0) return;
  Frame* frame_to_delete = frames[0];
  frames.erase(frames.begin());
  delete frame_to_delete;
}

void LucasKanade::CalculateFlow(cv::Mat& vel_x, cv::Mat& vel_y) {
	double* grad_x, *grad_y, *grad_t;
	GradientSmoothing(&grad_x, &grad_y, &grad_t);

	cv::Matx<double, 2, 2> a;
	cv::Matx<double, 2, 1> b, vel_vector;
	double ix, iy, it, ixx, ixy, iyy, ixt, iyt;
  int rows = frames[frames.size() / 2]->Rows();
  int cols = frames[frames.size() / 2]->Columns();

	vel_x = cv::Mat(rows, cols, CV_64F);
	vel_y = cv::Mat(rows, cols, CV_64F);

	for (int i = 0; i < rows; ++i) {
    double* ptr_x = vel_x.ptr<double>(i);
    double* ptr_y = vel_y.ptr<double>(i);
		for (int j = 0; j < cols; ++j, ++ptr_x, ++ptr_y) {
			ix = grad_x[i * cols + j];
			iy = grad_y[i * cols + j];
			it = grad_t[i * cols + j];

      // Calculates gradient values
			ixx = ix * ix; ixy = ix * iy; iyy = iy * iy;
			ixt = ix * it; iyt = iy * it;

      // [Ix2 IxIy][IxIy Iy2]
			a(0, 0) = ixx; a(0, 1) = ixy;
			a(1, 0) = ixy; a(1, 1) = iyy;
      
      // -[IxIt IyIt]
			b(0, 0) = -ixt;
      b(1, 0) = -iyt;

      // Solve linear equation
			vel_vector = a.inv() * b;
			*ptr_x = vel_vector(0, 0);
			*ptr_y = vel_vector(1, 0);
		}
	}
  delete [] grad_x;
  delete [] grad_y;
  delete [] grad_t;
}

void LucasKanade::SmoothFrame(int index) {
  Frame* frame = frames[index];
  int rows = frame->Rows();
  int cols = frame->Columns();

  // x-Spatial Smoothing
  int* pixels = new int[kSpatialSmoothSize];
  for (int i = 0; i < rows; ++i) {
    int pix_sum = 0, this_pix;
    std::fill(pixels, pixels + kSpatialSmoothSize, 0);
    for (int j = 0; j < cols; ++j) {
      this_pix = frame->GetPixel(i, j);
      pix_sum += this_pix - pixels[j % kSpatialSmoothSize];
      pixels[j % kSpatialSmoothSize] = this_pix;

      this_pix = static_cast<double>(pix_sum) / std::min(kSpatialSmoothSize, j + 1);
      frame->SetPixel(i, j, this_pix);
    }
  }

  // y-Spatial Smoothing
  for (int i = 0; i < frame->Columns(); ++i) {
    int pix_sum = 0, this_pix;
    std::fill(pixels, pixels + kSpatialSmoothSize, 0);
    for (int j = 0; j < frame->Rows(); ++j) {
      this_pix = frame->GetPixel(j, i);
      pix_sum += this_pix - pixels[j % kSpatialSmoothSize];
      pixels[j % kSpatialSmoothSize] = this_pix;

      this_pix = static_cast<double>(pix_sum) / std::min(kSpatialSmoothSize, j + 1);
      frame->SetPixel(j, i, this_pix);
    }
  }
  delete[] pixels;

  // Temporal Smoothing
  if (index > 0) {
    double kalpha = 1.0 - kAlpha;
    Frame* prev = frames[index - 1];
    for (int i = 0; i < frame->Rows(); ++i) {
      for (int j = 0; j < frame->Columns(); ++j) {
        int prev_pix = prev->GetPixel(i, j);
        int this_pix = frame->GetPixel(i, j);
        frame->SetPixel(i, j, kalpha * prev_pix + kAlpha * this_pix);
      }
    }
  }
}

void LucasKanade::GradientSmoothing(double** grad_x, double** grad_y, double** grad_t){
	double* grad_ex = GradientEstimationAtX();
	double* grad_ey = GradientEstimationAtY();
	double* grad_et = GradientEstimationAtT();
  int rows = frames[frames.size() / 2]->Rows();
  int cols = frames[frames.size() / 2]->Columns();
  
  *grad_x = new double[rows * cols];
  *grad_y = new double[rows * cols];
  *grad_t = new double[rows * cols];

  double* ptr_x = *grad_x;
  double* ptr_y = *grad_y;
  double* ptr_t = *grad_t;

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j, ++ptr_x, ++ptr_y, ++ptr_t) {
			double pix_sum_x, pix_sum_y, pix_sum_t;
			pix_sum_x = pix_sum_y = pix_sum_t = 0;
			for (int k = kGradientBegin; k <= kGradientEnd; ++k) {
				for (int l = kGradientBegin; l <= kGradientEnd; ++l) {
					if (i + k >= 0 && i + k < rows && j + l >= 0 && j + l < cols) {
						pix_sum_x += grad_ex[(i + k) * cols + j + l] * kKernel[k + 2][l + 2];
						pix_sum_y += grad_ey[(i + k) * cols + j + l] * kKernel[k + 2][l + 2];
						pix_sum_t += grad_et[(i + k) * cols + j + l] * kKernel[k + 2][l + 2];
					}
				}
			}
			*ptr_x = pix_sum_x;
			*ptr_y = pix_sum_y;
			*ptr_t = pix_sum_t;
		}
	}
  delete [] grad_ex;
  delete [] grad_ey;
  delete [] grad_et;
}

double* LucasKanade::GradientEstimationAtX() {
  Frame* frame = frames[frames.size() / 2];
  int rows = frame->Rows();
  int cols = frame->Columns();
  double* ix = new double[rows * cols];

  double* ptr = ix;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j, ++ptr) {
      int pix_sum = 0;
      for (int k = kGradientBegin; k <= kGradientEnd; ++k) {
        if (j + k < 0 || cols <= j + k) continue;
        pix_sum += frame->GetPixel(i, j + k) * kGradient[k - kGradientBegin];
      }
      *ptr = static_cast<double>(pix_sum) / 12.0;
    }
  }
  return ix;
}

double* LucasKanade::GradientEstimationAtY() {
  Frame* frame = frames[frames.size() / 2];
  int rows = frame->Rows();
  int cols = frame->Columns();
  double* iy = new double[rows * cols];
  
  double* ptr = iy;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j, ++ptr) {
      int pix_sum = 0;
      for (int k = kGradientBegin; k <= kGradientEnd; ++k) {
        if (i + k < 0 || rows <= i + k) continue;
        pix_sum += frame->GetPixel(i + k, j) * kGradient[k - kGradientBegin];
      }
      *ptr = static_cast<double>(pix_sum) / 12.0;
    }
  }
  return iy;
}

double* LucasKanade::GradientEstimationAtT() {
  int index = frames.size() / 2;
  int rows = frames[index]->Rows();
  int cols = frames[index]->Columns();
  double* it = new double[rows * cols];
  std::fill(it, it + rows * cols, 0.0);

  if (frames.size() < kGradientEnd - kGradientBegin + 1) return it;

  double* ptr = it;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j, ++ptr) {
      int pix_sum = 0;
      for (int k = kGradientBegin; k <= kGradientEnd; ++k)
        if (0 <= index + k && index + k < frames.size())
          pix_sum += frames[index + k]->GetPixel(i, j) * kGradient[k - kGradientBegin];
      *ptr = static_cast<double>(pix_sum) / 12.0;
    }
  }
  return it;
}