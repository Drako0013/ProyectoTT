#include "horn_schunck.h"

const int HornSchunck::kFlowIterations = 3;
const double HornSchunck::kFlowAlpha = 7.13;
const int HornSchunck::kAvgKernel[3][3] = { { 1, 2, 1 },
                                            { 2, 0, 2 },
                                            { 1, 2, 1 } };
const int HornSchunck::kKernelBegin = -1;
const int HornSchunck::kKernelEnd = 1;
const int HornSchunck::kFrameSize = 2;

cv::Mat HornSchunck::AddFrame(Frame* frame) {
	frames.push_back(frame);
	if (frames.size() > kFrameSize) RemoveFrame();
	return frames.back()->GetMatrix();
}

void HornSchunck::RemoveFrame() {
	if (frames.size() == 0) return;
	Frame* frame_to_delete = frames.front();
	frames.erase(frames.begin());
	delete frame_to_delete;
}

void HornSchunck::CalculateFlow(double** u, double** v) {
	int rows = frames.back()->Rows();
	int cols = frames.back()->Columns();
	*u = new double[rows * cols];
	*v = new double[rows * cols];
	std::fill(*u, *u + rows * cols, 0.0);
	std::fill(*v, *v + rows * cols, 0.0);

	if (frames.size() == 1) return;

	double* ix, *iy, *it;
	GradientEstimations(&ix, &iy, &it);

	// Horn-Schunck iterations
	for (int k = 0; k < kFlowIterations; ++k) {
		// Local average for each iteration
		double* up = LocalAverage(*u);
		double* vp = LocalAverage(*v);

		double* ptr_u = *u, *ptr_v = *v;
		double* ptr_up = up, *ptr_vp = vp;
		double* ptr_ix = ix, *ptr_iy = iy, *ptr_it = it;
		for (int i = 0; i < rows * cols; ++i, ++ptr_u, ++ptr_v) {
			double cup = *(ptr_up++), cvp = *(ptr_vp++);
			double cix = *(ptr_ix++), ciy = *(ptr_iy++), cit = *(ptr_it++);
			// Calculating flow by brightness intensity
			double flow = (cix * cup + ciy * cvp + cit)
				/ (kFlowAlpha * kFlowAlpha + cix * cix + ciy * ciy);
			*ptr_u = cup - cix * flow, *ptr_v = cvp - ciy * flow;
		}
		delete[] up;
		delete[] vp;
	}
	delete[] ix;
	delete[] iy;
	delete[] it;
}

double* HornSchunck::LocalAverage(double* M) {
	int rows = frames.back()->Rows();
	int cols = frames.back()->Columns();
	double* avg = new double[rows * cols];

	double* ptr = avg;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j, ++ptr) {
			double pix_sum = 0;
			for (int k = kKernelBegin; k <= kKernelEnd; ++k) {
				for (int l = kKernelBegin; l <= kKernelEnd; ++l) {
					// Out-of-bounds conditions
					if (i + k < 0 || rows <= i + k) continue;
					if (j + l < 0 || cols <= j + l) continue;
					pix_sum += kAvgKernel[k - kKernelBegin][l - kKernelBegin]
						* M[(i + k) * cols + j + l];
				}
			}
			*ptr = pix_sum / 12.0;
		}
	}
	return avg;
}

void HornSchunck::GradientEstimations(double** ix, double** iy, double** it) {
	Frame* frame_f = frames.back();
	Frame* frame_c = frames.front();

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
			// Cache required values for gradient
			int fc_00 = frame_c->GetPixel(i, j);
			int ff_00 = frame_f->GetPixel(i, j);
			int fc_01 = frame_c->GetPixel(i, j + 1);
			int ff_01 = frame_f->GetPixel(i, j + 1);
			int fc_10 = frame_c->GetPixel(i + 1, j);
			int ff_10 = frame_f->GetPixel(i + 1, j);
			int fc_11 = frame_c->GetPixel(i + 1, j + 1);
			int ff_11 = frame_f->GetPixel(i + 1, j + 1);

			// Formulas for gradient estimation
			int pix_sum_y = fc_01 - ff_00 + fc_11 + ff_11
				+ ff_01 - fc_10 - ff_10 - fc_00;
			int pix_sum_x = fc_10 - ff_00 + fc_11 + ff_11
				- ff_01 - fc_01 + ff_10 - fc_00;
			int pix_sum_t = ff_00 - fc_10 - fc_11 + ff_11
				+ ff_01 - fc_01 + ff_10 - fc_00;

			*ptr_x = static_cast<double>(pix_sum_x) / 4.0;
			*ptr_y = static_cast<double>(pix_sum_y) / 4.0;
			*ptr_t = static_cast<double>(pix_sum_t) / 4.0;
		}
	}
}