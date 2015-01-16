#include "horn_schunck.h"

const int HornSchunck::kFlowIterations = 5;
const double HornSchunck::kFlowAlpha = 13.7;
const int HornSchunck::kAvgKernel[3][3] = {{1,  2, 1},
                                           {2, -1, 2},
                                           {1,  2, 1}};
const int HornSchunck::kKernelBegin = -1;
const int HornSchunck::kKernelEnd = 1;

cv::Mat HornSchunck::AddFrame(Frame* frame) {
	frames.push_back(frame);
  int kernel_size = kKernelEnd - kKernelBegin + 1;
	if (frames.size() > kernel_size) RemoveFrame();
	return frames.back()->GetMatrix();
}

void HornSchunck::RemoveFrame() {
  if (frames.size() == 0) return;
  Frame* frame_to_delete = frames[0];
  frames.erase(frames.begin());
  delete frame_to_delete;
}

void HornSchunck::CalculateFlow(double** u, double** v) {
	double* ix, *iy, *it;
  int rows = frames.back()->Rows();
  int cols = frames.back()->Columns();
  GradientEstimations(&ix, &iy, &it);

	*u = new double[rows * cols];
	*v = new double[rows * cols];
	double* up = new double[rows * cols];
	double* vp = new double[rows * cols];

  double* ptr_up = up, *ptr_vp = vp;
	for (int i = 0; i < rows * cols; ++i)
    *(ptr_up++) = 0, *(ptr_vp++) = 0;

  for (int k = 0; k < kFlowIterations; ++k) {
    ptr_up = up, ptr_vp = vp;
    double* ptr_u = *u, *ptr_v = *v;
    double* ptr_ix = ix, *ptr_iy = iy, *ptr_it = it;
		for (int i = 0; i < rows * cols; ++i, ++ptr_up, ++ptr_vp, ++ptr_u,
                                     ++ptr_v, ++ptr_ix, ++ptr_iy, ++ptr_it) {
      double cup = *ptr_up, cvp = *ptr_vp;
			double cix = *ptr_ix, ciy = *ptr_iy, cit = *ptr_it;
			double flow = (cix * cup + ciy * cvp + cit) /
                    (kFlowAlpha * kFlowAlpha + cix * cix + ciy * ciy);
			*ptr_u = cup - cix * flow, *ptr_v = cvp - ciy * flow;
		}
    delete [] up;
    delete [] vp;
		up = LocalAverage(*u);
		vp = LocalAverage(*v);
	}
  delete [] ix;
  delete [] iy;
  delete [] it;
  delete [] up;
  delete [] vp;
}

double* HornSchunck::LocalAverage(double* M) {
  int rows = frames.back()->Rows();
  int cols = frames.back()->Columns();
	double* avg = new double[rows * cols];

  double* ptr = avg;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j, ++ptr) {
			double pix_sum = -M[i * cols + j] * 12;
			for (int k = kKernelBegin; k <= kKernelEnd; ++k)
				for (int l = kKernelBegin; l <= kKernelEnd; ++l)
					if (i + k >= 0 && i + k < rows && j + l >= 0 && j + l < cols)
						pix_sum += kAvgKernel[k - kKernelBegin][l - kKernelBegin] *
                       M[(i + k) * cols + j + l];
			*ptr = pix_sum / 12.0;
		}
	}
	return avg;
}

void HornSchunck::GradientEstimations(double** ix, double** iy, double** it) {
	Frame* frame_f = frames.back(), *frame_c;
  if (frames.size() == 1) frame_c = frame_f;
  else frame_c = frames[frames.size() - 2];

  int rows = frame_c->Rows();
  int cols = frame_c->Columns();
  *ix = new double[rows * cols];
  *iy = new double[rows * cols];
  *it = new double[rows * cols];

  double* ptr_x = *ix;
  double* ptr_y = *iy;
  double* ptr_t = *it;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j, ++ptr_x, ++ptr_y, ++ptr_t) {
			int pix_sum_x = -frame_c->GetPixel(i, j) - frame_f->GetPixel(i, j);
			int pix_sum_y = pix_sum_x, pix_sum_t = pix_sum_x;

      int add = frame_c->GetPixel(i, j + 1) + frame_f->GetPixel(i, j + 1);
		  pix_sum_x += add, pix_sum_y -= add, pix_sum_t += add;
      add = frame_c->GetPixel(i + 1, j + 1) + frame_f->GetPixel(i + 1, j + 1);
		  pix_sum_x += add, pix_sum_y += add, pix_sum_t += add;
      add = frame_c->GetPixel(i + 1, j) + frame_f->GetPixel(i + 1, j);
			pix_sum_x -= add, pix_sum_y += add, pix_sum_t += add;

			*ptr_x = static_cast<double>(pix_sum_x) / 4.0;
			*ptr_y = static_cast<double>(pix_sum_y) / 4.0;
			*ptr_t = static_cast<double>(pix_sum_t) / 4.0;
		}
	}
}