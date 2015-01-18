#include "simple_flow.h"

const double SimpleFlow::rd = 5.5;
const double SimpleFlow::rc = 0.08;
const double SimpleFlow::occlusion_limit = 20.0; //this value has to be defined correctly
const double SimpleFlow::threshold = 0.25;
const int SimpleFlow::NeighborhoodSize = 4;
const int SimpleFlow::Layers = 5;

cv::Mat SimpleFlow::AddFrame(Frame* frame) {
	frames.push_back(frame);
	if (frames.size() > 2) RemoveFrame();
	else FillDistanceWeightMatrix();
	return frames.back()->GetMatrix();
}

void SimpleFlow::RemoveFrame() {
	if (frames.size() == 0) return;
	Frame* frame_to_delete = frames[0];
	frames.erase(frames.begin());
	delete frame_to_delete;
}

int SimpleFlow::GetEnergy(Frame &f1, int x1, int y1, Frame &f2, int x2, int y2){
	//int dif = f1.GetPixel(x1, y1) - f2.GetPixel(x2, y2);
	//return (dif * dif);
	int color1 = f1.GetPixel(x1, y1);
	int color2 = f2.GetPixel(x2, y2);
	int redDif = (color1 >> 16) - (color2 >> 16);
	int greenDif = ((color1 >> 8) % (1 << 8) - (color2 >> 8) % (1 << 8));
	int blueDif = color1 % (1 << 8) - color2 % (1 << 8);
	return (redDif * redDif + greenDif * greenDif + blueDif * blueDif);
}

double SimpleFlow::GetWr(int *energyArray, int energySize){
	double mean = 0.0;
	int mini = energyArray[0];
	for (int i = 0; i < energySize; i++){
		if (energyArray[i] < mini)
		{
			mini = energyArray[i];
		}
		mean = mean + (double)energyArray[i];
	}
	mean = mean / (double)energySize;
	return mean - (double)mini;
}

void SimpleFlow::FillDistanceWeightMatrix(){
	distanceWeight = new double*[2 * NeighborhoodSize + 1];
	for (int i = 0; i < 2 * NeighborhoodSize + 1; i++)
		distanceWeight[i] = new double[2 * NeighborhoodSize + 1];
	for (int i = 0; i < 2 * NeighborhoodSize + 1; i++){
		for (int j = 0; j < 2 * NeighborhoodSize + 1; j++){
			double dx = NeighborhoodSize - j;
			double dy = NeighborhoodSize - i;
			double norm = (dx * dx) + (dy * dy);
			distanceWeight[i][j] = std::exp(-norm / (2 * rd));
		}
	}
}

double SimpleFlow::GetWd(int x0, int y0, int x, int y){
	/*
	int difX = x0 - x;
	int difY = y0 - y;
	int norm = (difX * difX) + (difY * difY);
	return std::exp(-norm / (2 * rd));
	*/
	return distanceWeight[std::abs(y - y0)][std::abs(x - x0)];
}

double SimpleFlow::GetWc(Frame& f1, int x0, int y0, int x, int y){
	//double norm = (double)( f1.GetPixel(x0, y0) - f1.GetPixel(x, y) );
	int color1 = f1.GetPixel(x0, y0);
	int color2 = f1.GetPixel(x, y);
	int redDif = (color1 >> 16) - (color2 >> 16);
	int greenDif = ((color1 >> 8) % (1 << 8) - (color2 >> 8) % (1 << 8));
	int blueDif = color1 % (1 << 8) - color2 % (1 << 8);
	double norm = (double)(redDif * redDif + greenDif * greenDif + blueDif * blueDif);
	return std::exp(-norm / (2 * rc));
}

double SimpleFlow::getSmoothness(Frame &f1, Frame &f2, int x0, int y0, int u, int v){
	const int n = SimpleFlow::NeighborhoodSize;
	int rows = f1.Rows();
	int cols = f1.Columns();


	double e = GetEnergy(f1, x0, y0, f2, x0 + u, y0 + v);
	double result = 0.0f;
	for (int i = -n; i <= n; ++i) {
		for (int j = -n; j <= n; ++j) {
			if (x0 + i >= 0 && x0 + i < rows && y0 + j >= 0 && y0 + j < cols) {
				result += distanceWeight[std::abs(i)][std::abs(j)] * GetWc(f1, x0, y0, x0 + i, y0 + j) * e;
			}
		}
	}
	return result;
}

void SimpleFlow::BuildPyramid(Frame &src, std::vector<Frame>& pyramid) {
	pyramid.push_back(src);
	for (int i = 1; i <= SimpleFlow::Layers; ++i) {
		Frame prev = pyramid[i - 1];
		if (prev.Rows() <= 1 || prev.Columns() <= 1) {
			break;
		}
		Frame next = prev.Copy();
		next.Rescale((prev.Rows() + 1) / 2, (prev.Columns() + 1) / 2);
		pyramid.push_back(next);
	}
}

void SimpleFlow::CrossBilateralFilter(cv::Mat &orig, Frame &edge, cv::Mat& confidence, cv::Mat& dest) {
	const int n = SimpleFlow::NeighborhoodSize;
	int rows = edge.Rows();
	int cols = edge.Columns();
	for (int x = 0; x < rows; ++x) {
		double* ptr_dest = dest.ptr<double>(x);
		double* ptr_orig = orig.ptr<double>(x);
		double* ptr_wr = confidence.ptr<double>(x);
		for (int y = 0; y < cols; ++y, ++ptr_dest, ++ptr_orig, ++ptr_wr) {
			double result = 0.0f;
			double wr = *ptr_wr;
			for (int i = -n; i <= n; ++i) {
				for (int j = -n; j <= n; ++j) {
					if (x + i >= 0 && x + i < rows && y + j >= 0 && y + j < cols) {
						result += (*ptr_orig) * distanceWeight[std::abs(i)][std::abs(j)] * GetWc(edge, x, y, x + i, y + j) * wr;
					}
				}
			}
			*ptr_dest = result;
		}
	}
}


cv::Mat SimpleFlow::UpscaleFlow(cv::Mat& flow, int new_cols, int new_rows, Frame &image, cv::Mat& confidence) {
	cv::Mat orig_flow = flow;
	CrossBilateralFilter(orig_flow, image, confidence, flow);
	cv::Mat new_flow;
	resize(flow, new_flow, cv::Size(new_rows, new_cols), 0, 0, cv::INTER_NEAREST);
	new_flow *= 2;
	return new_flow;
}

void SimpleFlow::BilateralFilter(cv::Mat flow_x, cv::Mat flow_y, Frame cur, Frame next, cv::Mat confidence, std::vector< std::vector<bool> >& isOccludedPixel) {
	const int n = SimpleFlow::NeighborhoodSize;
	int rows = flow_x.rows;
	int cols = flow_x.cols;
	for (int x = 0; x < rows; x++){
		double* ptr_x_f = flow_x.ptr<double>(x);
		double* ptr_y_f = flow_y.ptr<double>(x);
		double* ptr_wr = confidence.ptr<double>(x);
		for (int y = 0; y < cols; y++, ++ptr_x_f, ++ptr_y_f, ++ptr_wr){
			if (!(isOccludedPixel[x][y])){
				double wr = *ptr_wr;
				*ptr_x_f = 0.0;
				*ptr_y_f = 0.0;
				for (int u = -n; u <= n; ++u) {
					for (int v = -n; v <= n; ++v) {
						if (x + u >= 0 && x + u < rows && y + v >= 0 && y + v < cols) {
							*ptr_x_f += GetWd(x, y, x + u, y + v) * GetWc(cur, x, y, x + u, y + v) * wr;
							*ptr_y_f += GetWd(x, y, x + u, y + v) * GetWc(cur, x, y, x + u, y + v) * wr;
						}
					}
				}
			}
		}
	}
}

void SimpleFlow::CalcOcclusion(Frame& cur, Frame& next, cv::Mat& vel_x, cv::Mat& vel_y, cv::Mat& vel_x_inv, cv::Mat& vel_y_inv, std::vector< std::vector<bool> >& isOccludedPixel) {

	int rows = cur.Rows();
	int cols = cur.Columns();

	isOccludedPixel = std::vector< std::vector<bool> >(rows, std::vector<bool>(cols));

	for (int x = 0; x < rows; ++x) {
		double* ptr_x = vel_x.ptr<double>(x);
		double* ptr_y = vel_y.ptr<double>(x);

		double* ptr_x_inv = vel_x_inv.ptr<double>(x);
		double* ptr_y_inv = vel_y_inv.ptr<double>(x);

		for (int y = 0; y < cols; ++y, ++ptr_x, ++ptr_y, ++ptr_x_inv, ++ptr_y_inv) {
			double d_ptr_x = (*ptr_x - *ptr_x_inv);
			double d_ptr_y = (*ptr_y - *ptr_y_inv);
			double occlusion = sqrt((d_ptr_x * d_ptr_x) + (d_ptr_y * d_ptr_y));
			//determine occludiness of pixels
			isOccludedPixel[x][y] = (occlusion > occlusion_limit); //true if the pixel is occluded

		}
	}
}

void SimpleFlow::CalcStageFlow(Frame& cur, Frame& next, cv::Mat& vel_x, cv::Mat& vel_y, cv::Mat& irreg) {
	const int n = SimpleFlow::NeighborhoodSize;

	int rows = cur.Rows();
	int cols = cur.Columns();

	double me, e;
	//double E[n * 2 + 1][n * 2 + 1]; // Due to obvious limitations of arrays, E(u, v) is represented by E[u + n][v + n]

	vel_x = cv::Mat(rows, cols, CV_64F);
	vel_y = cv::Mat(rows, cols, CV_64F);
	for (int x = 0; x < rows; ++x) {
		double* ptr_x = vel_x.ptr<double>(x);
		double* ptr_y = vel_y.ptr<double>(x);
		int* ptr_irreg = irreg.ptr<int>(x);
		for (int y = 0; y < cols; ++y, ++ptr_x, ++ptr_y, ++ptr_irreg) {
			if (*ptr_irreg) {
				me = std::numeric_limits<double>::max();
				for (int u = -n; u <= n; ++u) {
					for (int v = -n; v <= n; ++v) {
						if (x + u < 0 || x + u >= rows || y + v < 0 || y + v >= cols) {
							//E[u + n][v + n] = std::numeric_limits<double>::max();
						}
						else {
							//E[u + n][v + n] = SimpleFlow::getSmoothness(cur, next, x, y, u + *ptr_x, v + *ptr_y);
							e = SimpleFlow::getSmoothness(cur, next, x, y, u + *ptr_x, v + *ptr_y);
							if (e < me) {
								me = e;
								*ptr_x = u;
								*ptr_y = v;
							}
						}
					}
				}
				/*
				// TODO: Sub-pixel estimation (low priority)
				for (int u = -n; u <= n; ++u) {
					for (int v = -n; v <= n; ++v) {
						if (E[u + n][v + n] < me) {
							me = E[u + n][v + n];
							*ptr_x = u;
							*ptr_y = v;
						}
					}
				}
				*/
			}
		}
	}
}

void SimpleFlow::CalcConfidence(Frame& cur, Frame& next, cv::Mat& confidence) {
	const int n = SimpleFlow::NeighborhoodSize;
	int energyArray[(2 * n + 1) * (2 * n + 1)], energySize;
	int rows = cur.Rows();
	int cols = next.Columns();
	confidence = cv::Mat(rows, cols, CV_64F);
	for (int x = 0; x < rows; ++x) {
		double* ptr_wr = confidence.ptr<double>(x);
		for (int y = 0; y < cols; ++y, ++ptr_wr){
			energySize = 0;
			for (int u = -n; u <= n; ++u) {
				for (int v = -n; v <= n; ++v) {
					if (x >= 0 && x < rows && y >= 0 && y < cols && x + u >= 0 && x + u < rows && y + v >= 0 && y + v < cols) {
						energyArray[energySize] = GetEnergy(cur, x, y, next, x + u, y + v);
						energySize++;
					}
				}
			}
			*ptr_wr = GetWr(energyArray, energySize);
		}
	}
}

static inline float interpolateVal(int w, int h, double v11, double v12, double v21, double v22, int r, int c) {
	   if (r == 0 && c == 0) { return v11; }
	   if (r == 0 && c == w) { return v12; }
	   if (r == h && c == 0) { return v21; }
	   if (r == h && c == w) { return v22; }
	   float qr = (double)r / h;
	   float pr = 1.0 - qr;
	   float qc = (double)c / w;
	   float pc = 1.0 - qc;
	   return v11 * pr * pc + v12 * pr * qc + v21 * qr * pc + v22 * qc * qr;
}

void SimpleFlow::CalcRegularFlow(cv::Mat& flow_x, cv::Mat& flow_y, cv::Mat& irreg) {
	const int n = SimpleFlow::NeighborhoodSize;
	int rows = flow_x.rows;
	int cols = flow_x.cols;
	for (int x = 0; x < rows; ++x) {
		double* ptr_x = flow_x.ptr<double>(x);
		double* ptr_y = flow_y.ptr<double>(x);
		int* ptr_irreg = irreg.ptr<int>(x);
		for (int y = 0; y < cols; ++y, ++ptr_x, ++ptr_y, ++ptr_irreg) {
			if (!(*ptr_irreg)) {
				int l = std::max(x - n, 0), r = std::min(x + n, rows - 1), u = std::max(y - n, 0), d = std::min(y + n, cols - 1);
				*ptr_x = interpolateVal(l - r, u - d, flow_x.at<double>(l, y), flow_x.at<double>(r, u), flow_x.at<double>(l, d), flow_x.at<double>(r, d), x - l, y - u);
				*ptr_y = interpolateVal(l - r, u - d, flow_y.at<double>(l, y), flow_y.at<double>(r, u), flow_y.at<double>(l, d), flow_y.at<double>(r, d), x - l, y - u);
			}
		}
	}
}

void SimpleFlow::CalculateFlow(cv::Mat& vel_x, cv::Mat& vel_y) {

	if (frames.size() < 2) {
		return;
	}

	cv::Mat flow_x, flow_y, flow_inv_x, flow_inv_y, confidence, confidence_inv, irreg, irreg_inv, new_irreg;

	std::vector<Frame> pyramid_cur, pyramid_next;

	BuildPyramid(*frames[0], pyramid_cur);
	BuildPyramid(*frames[1], pyramid_next);

	std::vector< std::vector<bool> > isOccluded(pyramid_cur.back().Rows(), std::vector<bool>(pyramid_cur.back().Columns()));
	std::vector< std::vector<bool> > isOccludedInv(pyramid_cur.back().Rows(), std::vector<bool>(pyramid_cur.back().Columns()));

	irreg = cv::Mat::ones(pyramid_cur.back().GetMatrix().size(), CV_64F);
	irreg_inv = cv::Mat::ones(pyramid_cur.back().GetMatrix().size(), CV_64F);

	CalcStageFlow(pyramid_cur.back(), pyramid_next.back(), flow_x, flow_y, irreg);
	CalcStageFlow(pyramid_next.back(), pyramid_cur.back(), flow_inv_x, flow_inv_y, irreg_inv);

	CalcOcclusion(pyramid_cur.back(), pyramid_next.back(), flow_x, flow_y, flow_inv_x, flow_inv_y, isOccluded);
	CalcOcclusion(pyramid_next.back(), pyramid_cur.back(), flow_inv_x, flow_inv_y, flow_x, flow_y, isOccludedInv);

	confidence = cv::Mat::ones(pyramid_cur.back().GetMatrix().size(), CV_64F);
	confidence_inv = cv::Mat::ones(pyramid_cur.back().GetMatrix().size(), CV_64F);

	for (int l = pyramid_cur.size() - 2; l >= 0; --l) {
		Frame cur = pyramid_cur[l];
		Frame next = pyramid_next[l];
		Frame p_cur = pyramid_cur[l + 1];
		Frame p_next = pyramid_next[l + 1];

		const int curr_rows = cur.Rows();
		const int curr_cols = cur.Columns();

		std::vector< std::vector<bool> > isOccluded(curr_rows, std::vector<bool>(curr_cols));
		std::vector< std::vector<bool> > isOccludedInv(curr_rows, std::vector<bool>(curr_cols));

		CalcIrregularityMatrix(flow_x, flow_y, irreg);
		CalcIrregularityMatrix(flow_inv_x, flow_inv_y, irreg_inv);

		flow_x = UpscaleFlow(flow_x, curr_rows, curr_cols, p_cur, confidence);
		flow_inv_x = UpscaleFlow(flow_x, curr_rows, curr_cols, p_next, confidence_inv);
		flow_y = UpscaleFlow(flow_y, curr_rows, curr_cols, p_cur, confidence);
		flow_inv_y = UpscaleFlow(flow_y, curr_rows, curr_cols, p_next, confidence_inv);

		resize(irreg, new_irreg, cv::Size(curr_cols, curr_rows), 0, 0, cv::INTER_NEAREST);
		irreg = new_irreg;
		resize(irreg_inv, new_irreg, cv::Size(curr_cols, curr_rows), 0, 0, cv::INTER_NEAREST);
		irreg_inv = new_irreg;

		CalcConfidence(cur, next, confidence);
		CalcConfidence(next, cur, confidence_inv);

		CalcStageFlow(cur, next, flow_x, flow_y, irreg);
		CalcStageFlow(next, cur, flow_inv_x, flow_inv_y, irreg_inv);

		CalcRegularFlow(flow_x, flow_y, irreg);
		CalcRegularFlow(flow_x, flow_y, irreg_inv);

		CalcOcclusion(cur, next, flow_x, flow_y, flow_inv_x, flow_inv_y, isOccluded);
		CalcOcclusion(next, cur, flow_inv_x, flow_inv_y, flow_x, flow_y, isOccludedInv);

	}

	/*
	BilateralFilter(flow_x, flow_y, cur, next, confidence, isOccluded);
	*/

	vel_x = flow_x;
	vel_y = flow_y;

}

void SimpleFlow::CalcIrregularityMatrix(cv::Mat& flow_x, cv::Mat& flow_y, cv::Mat& irreg_mat){
	const int n = SimpleFlow::NeighborhoodSize;
	int rows = flow_x.rows;
	int cols = flow_x.cols;
	//create matrix of bool
	irreg_mat = cv::Mat(rows, cols, CV_32S);
	double dif_u = 0.0;
	double dif_v = 0.0;
	for (int x = 0; x < rows; x++){
		int* ptr_irr = irreg_mat.ptr<int>(x);
		for (int y = 0; y < cols; y++, ++ptr_irr){
			*ptr_irr = false;
			for (int u = -n; u <= n; ++u) {
				for (int v = -n; v <= n; ++v) {
					if (x + u > 0 && x + u >= rows && y + v < 0 && y + v >= cols) {
						double* flow_x_ini_ptr = flow_x.ptr<double>(x) + y;
						double* flow_y_ini_ptr = flow_y.ptr<double>(x) + y;
						double* flow_x_ptr = flow_x.ptr<double>(x + u) + y + v;
						double* flow_y_ptr = flow_y.ptr<double>(x + u) + y + v;
						dif_u = *flow_x_ini_ptr - *flow_x_ptr;
						dif_v = *flow_y_ini_ptr - *flow_y_ptr;
						*ptr_irr |= (sqrt( dif_u * dif_u + dif_v * dif_v ) > SimpleFlow::threshold);
					}
				}
			}
			*ptr_irr = *ptr_irr ? std::numeric_limits<int>::max() : 0;
		}
	}
}