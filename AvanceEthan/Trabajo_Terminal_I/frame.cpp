#include "frame.h"

Frame::Frame(bool grayscale_)
  : matrix(), grayscale(grayscale_),
    cached(false), matrix_cache(NULL) {}

Frame::Frame(cv::Mat* matrix, bool grayscale_)
  : cached(false), matrix_cache(NULL),
    grayscale(grayscale_) {
  SetMatrix(matrix);
}

Frame Frame::Copy() {
	Frame f(grayscale);
	f.matrix = this->matrix;
	// Should we copy also the cache?
	return f;
}

Frame::~Frame() {
	//TODO: Implement via std::shared_ptr
	//if (cached) delete [] matrix_cache;
}

int Frame::Rows() const {
  return matrix.rows;
}

int Frame::Columns() const {
  return matrix.cols;
}

void Frame::Rescale(int width, int height) {
  // Do not resize with negative values
  if (width < 0 || height < 0) return;
  cv::Mat matrix_copy;
  matrix.copyTo(matrix_copy);
  cv::resize(matrix_copy, matrix,
             cv::Size(width, height));
  // Mark as not cached
  cached = false;
}

int Frame::GetPixel(int x, int y) const {
  // Out-of-bounds conditions
  if (x < 0 || Rows() <= x) return 0;
  if (y < 0 || Columns() <= y) return 0;
  // Fast getter with Mat cache
  if (cached) return matrix_cache[x * Columns() + y];
  // Return simple grayscale integer
  if (grayscale) return matrix.at<uchar>(x, y);
  // Casting RGB vector to simple integer
  cv::Vec3b v = matrix.at<cv::Vec3b>(x, y);
  return (v[2] << 16) | (v[1] << 8) | v[0];
}

void Frame::SetPixel(int x, int y, int pixel) {
  // Out-of-bounds conditions
  if (x < 0 || Rows() <= x) return;
  if (y < 0 || Columns() <= y) return;
  if (cached) {
    // Fast assignment with Mat cache
    matrix_cache[x * Columns() + y] = pixel;
  } else if (grayscale) {
    // Grayscale for non-colored images
    matrix.at<uchar>(x, y) = pixel;
  } else {
    // Color space for colored images is RGB
    matrix.at<cv::Vec3b>(x, y)[0] = pixel & 255;
    matrix.at<cv::Vec3b>(x, y)[1] = (pixel >> 8) & 255;
    matrix.at<cv::Vec3b>(x, y)[2] = (pixel >> 16) & 255;
  }
}

bool Frame::IsGrayscale() {
	return grayscale;
}

cv::Mat Frame::GetMatrix() const {
  return matrix;
}

// BGR as default color space of Mat
void Frame::SetMatrix(cv::Mat* matrix_) {
  if (grayscale) {
    // Transformation to grayscale Mat
    cv::cvtColor(*matrix_, matrix, CV_BGR2GRAY);
  } else {
    // Transformation to RGB standard Mat
    cv::cvtColor(*matrix_, matrix, CV_BGR2RGB);
  }
  // Mark as not cached
  cached = false;
}

void Frame::GetMatrixOnCache() {
  int rows = Rows();
  int cols = Columns();
  ResizeCache(rows, cols);
  // Fast caching with pointers
  int* pointer = matrix_cache;
  for (int i = 0; i < rows; ++i) {
    uchar* row = matrix.ptr<uchar>(i);
    for (int j = 0; j < cols; ++j, ++pointer) {
      if (grayscale) {
        // Simple grayscale
        *pointer = *(row++);
      } else {
        // RGB conversion to integer
        *pointer = *(row++);
        *pointer |= *(row++) << 8;
        *pointer |= *(row++) << 16;
      }
    }
  }
  // Mark as cached
  cached = true;
}

void Frame::DeleteCache() {
	if (cached) delete[] matrix_cache;
}

void Frame::GetCacheOnMatrix() {
  if (!cached) return;
  int rows = Rows();
  int cols = Columns();
  // Fast caching with pointers
  int* pointer = matrix_cache;
  for (int i = 0; i < rows; ++i) {
    uchar* row = matrix.ptr<uchar>(i);
    for (int j = 0; j < cols; ++j, ++pointer) {
      if (grayscale) {
        // Simple grayscale
        *(row++) = *pointer;
      } else {
        // Integer conversion to RGB
        *(row++) = *pointer & 255;
        *(row++) = (*pointer >> 8) & 255;
        *(row++) = (*pointer >> 16) & 255;
      }
    }
  }
}

void Frame::ResizeCache(int rows, int columns) {
  if (cached) delete [] matrix_cache;
  matrix_cache = new int[rows * columns];
}