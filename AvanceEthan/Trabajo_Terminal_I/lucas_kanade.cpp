#include "lucas_kanade.h"

const double LucasKanade::kAlpha = 0.75;
const int LucasKanade::kSpatialSmoothSize = 5;
const int LucasKanade::kGradientKernelSize = 5;

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