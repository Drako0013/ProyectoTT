#include "horn_schunck.h"


const int HornSchunck::avgKernel[3][3] = {
		{ 1, 2, 1 },
		{ 2, -1, 2 },
		{ 1, 2, 1 }
};

HornSchunck::HornSchunck()
{
}


HornSchunck::~HornSchunck()
{
}

cv::Mat HornSchunck::AddFrame(Frame* frame) {
	frames_.push_back(*frame);
	if (frames_.size() > 3) RemoveFrame();
	return frames_.back().GetMatrix();
}

void HornSchunck::RemoveFrame() {
	if (frames_.size() == 0) return;
	frames_.erase(frames_.begin());
}

cv::Mat HornSchunck::LocalAverage(cv::Mat &M) {
	// Expected Doubles matrix!!
	cv::Mat A(M.rows, M.cols, CV_64F);
	for (int i = 0; i < M.rows; i++) {
		for (int j = 0; j < M.cols; j++) {
			double pix_sum = -M.at<double>(i, j) * 12.0;
			for (int k = -1; k <= 1; k++) {
				for (int l = -1; l <= 1; l++) {
					if (i + k >= 0 && i + k < M.rows && j + l >= 0 && j + l < M.cols) {
						pix_sum += M.at<double>(i + k, j + l) * avgKernel[k + 1][l + 1];
					}
				}
			}
			A.at<double>(i, j) = pix_sum / 12.0;
		}
	}
	return A;
}

// TODO: Find a (alpha); find suitable value for k (# of iterations)
//a = 100, k = 100
void HornSchunck::CalculateFlow(cv::Mat &U, cv::Mat &V) {
	cv::Mat Ix = GradientEstimationAtX();
	cv::Mat Iy = GradientEstimationAtY();
	cv::Mat It = GradientEstimationAtT();
	const int rows = Ix.rows, cols = Ix.cols;
	const double a = 100.0;
	U = cv::Mat(rows, cols, CV_64F);
	V = cv::Mat(rows, cols, CV_64F);
	cv::Mat Up = cv::Mat(rows, cols, CV_64F);
	cv::Mat Vp = cv::Mat(rows, cols, CV_64F);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			Up.at<double>(i, j) = 0;
			Vp.at<double>(i, j) = 0;
		}
	}
	for (int k = 0; k < 40; k++) {
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				double cIx = Ix.at<double>(i, j);
				double cIy = Iy.at<double>(i, j);
				double cIt = It.at<double>(i, j);
				double cUp = Up.at<double>(i, j);
				double cVp = Up.at<double>(i, j);
				double f = (cIx * cUp + cIy * cVp + cIt) / (a * a + cIx * cIx + cIy * cIy);
				U.at<double>(i, j) = cUp - cIx * f;
				V.at<double>(i, j) = cVp - cIy * f;
			}
		}
		Up = LocalAverage(U);
		Vp = LocalAverage(V);
	}
}


cv::Mat HornSchunck::GradientEstimationAtX() {
	// TODO: frameF should represent next frame, frameC = current frame
	Frame* frameF = &frames_[frames_.size() - 1];
	Frame* frameC = &frames_[frames_.size() - frames_.size() >= 1 ? 1 : 0];
	cv::Mat Ix = cv::Mat(frameC->Rows(), frameC->Columns(), CV_64F);
	for (int i = 0; i < frameC->Rows(); ++i) {
		for (int j = 0; j < frameC->Columns(); ++j) {
			double pix_sum = -frameC->GetPixel(i, j) - frameF->GetPixel(i, j);
			if (j < frameC->Columns() - 1) {
				pix_sum += frameC->GetPixel(i, j + 1) + frameF->GetPixel(i, j + 1);
			}
			if (i < frameC->Rows() - 1 && j < frameC->Columns() - 1) {
				pix_sum += frameC->GetPixel(i + 1, j + 1) + frameF->GetPixel(i + 1, j + 1);
			}
			if (i < frameC->Rows() - 1) {
				pix_sum -= frameC->GetPixel(i + 1, j) + frameF->GetPixel(i + 1, j);
			}
			Ix.at<double>(i, j) = pix_sum / 4.0;
		}
	}
	return Ix;
}

cv::Mat HornSchunck::GradientEstimationAtY() {
	// TODO: frameF should represent next frame, frameC = current frame
	Frame* frameF = &frames_[frames_.size() - 1];
	Frame* frameC = &frames_[frames_.size() - frames_.size() >= 1 ? 1 : 0];
	cv::Mat Iy = cv::Mat(frameC->Rows(), frameC->Columns(), CV_64F);
	for (int i = 0; i < frameC->Rows(); ++i) {
		for (int j = 0; j < frameC->Columns(); ++j) {
			double pix_sum = -frameC->GetPixel(i, j) - frameF->GetPixel(i, j);
			if (j < frameC->Columns() - 1) {
				pix_sum -= frameC->GetPixel(i, j + 1) + frameF->GetPixel(i, j + 1);
			}
			if (i < frameC->Rows() - 1 && j < frameC->Columns() - 1) {
				pix_sum += frameC->GetPixel(i + 1, j + 1) + frameF->GetPixel(i + 1, j + 1);
			}
			if (i < frameC->Rows() - 1) {
				pix_sum += frameC->GetPixel(i + 1, j) + frameF->GetPixel(i + 1, j);
			}
			Iy.at<double>(i, j) = pix_sum / 4.0;
		}
	}
	return Iy;
}

cv::Mat HornSchunck::GradientEstimationAtT() {
	// TODO: frameF should represent next frame, frameC = current frame
	Frame* frameF = &frames_[frames_.size() - 1];
	Frame* frameC = &frames_[frames_.size() - frames_.size() >= 1 ? 1 : 0];
	cv::Mat Iy = cv::Mat(frameC->Rows(), frameC->Columns(), CV_64F);
	for (int i = 0; i < frameC->Rows(); ++i) {
		for (int j = 0; j < frameC->Columns(); ++j) {
			double pix_sum = frameF->GetPixel(i, j) - frameC->GetPixel(i, j);
			if (j < frameC->Columns() - 1) {
				pix_sum += frameF->GetPixel(i, j + 1) - frameC->GetPixel(i, j + 1);
			}
			if (i < frameC->Rows() - 1 && j < frameC->Columns() - 1) {
				pix_sum += frameF->GetPixel(i + 1, j + 1) - frameC->GetPixel(i + 1, j + 1);
			}
			if (i < frameC->Rows() - 1) {
				pix_sum += frameF->GetPixel(i + 1, j) - frameC->GetPixel(i + 1, j);
			}
			Iy.at<double>(i, j) = pix_sum / 4.0;
		}
	}
	return Iy;
}