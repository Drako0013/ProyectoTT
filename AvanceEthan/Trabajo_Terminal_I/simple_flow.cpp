#include "simple_flow.h"

cv::Mat SimpleFlow::AddFrame(Frame* frame) {
	frames.push_back(frame);
	if (frames.size() > 2) RemoveFrame();
	return frames.back()->GetMatrix();
}

void SimpleFlow::RemoveFrame() {
	if (frames.size() == 0) return;
	Frame* frame_to_delete = frames[0];
	frames.erase(frames.begin());
	delete frame_to_delete;
}

int SimpleFlow::GetEnergy(Frame &f1, int x1, int y1, Frame &f2, int x2, int y2){
	int dif = f1.GetPixel(x1, y1) - f2.GetPixel(x2, y2);
	return (dif * dif); 
}

double SimpleFlow::GetWr(std::vector<int> &energyArray){
	double mean = 0.0;
	int mini = energyArray[0];
	for(unsigned int i = 0; i < energyArray.size(); i++){
		mini = std::min(mini, energyArray[i]);
		mean = mean + (double)energyArray[i];
	}
	mean = mean / (double)energyArray.size();
	return mean - (double)mini;
}


double SimpleFlow::GetWd(int x0, int y0, int x, int y){
	int difX = x0 - x;
	int difY = y0 - y;
	int norm = (difX * difX) + (difY * difY);
	return std::exp( -norm / (2 * rd) );
}

double SimpleFlow::GetWc(Frame &f1, int x0, int y0, int x, int y){
	double norm = (double)( f1.GetPixel(x0, y0) - f1.GetPixel(x, y) );
	return std::exp( -norm / (2 * rc) );
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
				result += GetWd(x0, y0, x0 + i, y0 + j) * GetWc(f1, x0, y0, x0 + i, y0 + j) * e;
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
		Frame next;
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
		double* ptr_wr = orig.ptr<double>(x);
		for (int y = 0; y < cols; ++y, ++ptr_dest, ++ptr_orig, ++ptr_wr) {
			double result = 0.0f;
			for (int i = -n; i <= n; ++i) {
				for (int j = -n; j <= n; ++j) {
					if (x + i >= 0 && x + i < rows && y + j >= 0 && y + j < cols) {
						result += (*ptr_orig) * GetWd(x, y, x + i, y + j) * GetWc(edge, x, y, x + i, y + j) * (*ptr_wr);
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
	resize(flow, new_flow, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_NEAREST);
	new_flow *= 2;
	return new_flow;
}

void SimpleFlow::CalcStageFlow(Frame& cur, Frame& next, cv::Mat& flow_x, cv::Mat& flow_y) {
	// TODO: Refactor code
}

void SimpleFlow::CalcMultiStageFlow(cv::Mat& vel_x, cv::Mat& vel_y) {
	cv::Mat flow_x, flow_y, flow_inv_x, flow_inv_y, confidence, confidence_inv;

	//For now...
	confidence = cv::Mat::ones(frames[0]->GetMatrix().size(), CV_64F);
	confidence_inv = cv::Mat::ones(frames[0]->GetMatrix().size(), CV_64F);

	std::vector<Frame> pyramid_cur, pyramid_next;

	BuildPyramid(*frames[0], pyramid_cur);
	BuildPyramid(*frames[1], pyramid_next);

	// Refactor SimpleFlow for one stage. We'll be using it a lot...
	CalcStageFlow(pyramid_cur.back(), pyramid_next.back(), flow_x, flow_y);
	CalcStageFlow(pyramid_next.back(), pyramid_cur.back(), flow_inv_x, flow_inv_y);

	for (int l = pyramid_cur.size() - 2; l >= 0; --l) {
		Frame cur = pyramid_cur[l];
		Frame next = pyramid_next[l];
		Frame p_cur = pyramid_cur[l + 1];
		Frame p_next = pyramid_next[l + 1];

		const int curr_rows = cur.Rows();
		const int curr_cols = cur.Columns();

		flow_x = UpscaleFlow(flow_x, curr_rows, curr_cols, p_cur, confidence);
		flow_inv_x = UpscaleFlow(flow_x, curr_rows, curr_cols, p_next, confidence_inv);
		flow_y = UpscaleFlow(flow_y, curr_rows, curr_cols, p_cur, confidence);
		flow_inv_y = UpscaleFlow(flow_y, curr_rows, curr_cols, p_next, confidence_inv);


		//CalcConfidence(cur, curr_to, flow, confidence, max_flow);
		CalcStageFlow(cur, next, flow_x, flow_y);

		//calcConfidence(curr_to, cur, flow_inv, confidence_inv, max_flow);
		CalcStageFlow(next, cur, flow_inv_x, flow_inv_y);

	}

	vel_x = flow_x;
	vel_y = flow_y;
}


void SimpleFlow::CalculateFlow(cv::Mat& vel_x, cv::Mat& vel_y) {

	// Following code needs to be refactored:

	const int n = SimpleFlow::NeighborhoodSize;

	double occlusion_limit = 1.0; //this value has to be defined

	int rows = frames[0]->Rows();
	int cols = frames[0]->Columns();

	double e;
	double E[n * 2 + 1][n * 2 + 1]; // Due to obvious limitations of arrays, E(u, v) is represented by E[u + n][v + n]
	double Einv[n * 2 + 1][n * 2 + 1];

	std::vector<int> energyArray;

	vel_x = cv::Mat(rows, cols, CV_64F);
	vel_y = cv::Mat(rows, cols, CV_64F);
	for (int x = 0; x < rows; ++x) {
		double* ptr_x = vel_x.ptr<double>(x);
		double* ptr_y = vel_y.ptr<double>(x);

		double* ptr_x_inv = vel_x.ptr<double>(x);
		double* ptr_y_inv = vel_y.ptr<double>(x);

		for (int y = 0; y < cols; ++y, ++ptr_x, ++ptr_y) {
			energyArray.clear();
			for (int u = -n; u <= n; ++u) {
				for (int v = -n; v <= n; ++v) {
					if (x + u < 0 || x + u >= rows || y + v < 0 || y + v >= cols) {
						E[u + n][v + n] = std::numeric_limits<double>::max();
					} else {
						energyArray.push_back(GetEnergy(*frames[0], x, y, *frames[1], x + u, y + v));
						E[u + n][v + n] = SimpleFlow::getSmoothness(*frames[0], *frames[1], x, y, u, v);
					}
				}
			}
			double me = std::numeric_limits<double>::max();
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

			// TODO: Bi-filter results; calculate occlusion (check)

			//OcclusionCalculation
			for (int u = -n; u <= n; ++u) {
				for (int v = -n; v <= n; ++v) {
					if (x + u < 0 || x + u >= rows || y + v < 0 || y + v >= cols) {
						Einv[u + n][v + n] = std::numeric_limits<double>::max();
					} else {
						Einv[u + n][v + n] = SimpleFlow::getSmoothness(*frames[1], *frames[0], x + u, y + v, u, v);
					}
				}
			}
			
			double me = std::numeric_limits<double>::max();
			// TODO: Sub-pixel estimation (low priority)
			for (int u = -n; u <= n; ++u) {
				for (int v = -n; v <= n; ++v) {
					if (Einv[u + n][v + n] < me) {
						me = Einv[u + n][v + n];
						*ptr_x_inv = u;
						*ptr_y_inv = v;
					}
				}
			}
			double d_ptr_x = (*ptr_x - *ptr_x_inv);
			double d_ptr_y = (*ptr_y - *ptr_y_inv);
			double occlusion = sqrt( (d_ptr_x * d_ptr_x) + (d_ptr_y * d_ptr_y) );

			/*
				We found it useful to further regularize
				the result by applying a bilateral filter on the flow vectors.
				For this operation, we discard the occluded pixels, and use
				the weights wd and wc with an additional weight wr that represents
				how reliable is our flow estimate at (x, y):

			*/

			if( occlusion < occlusion_limit ){
				int wr = SimpleFlow::GetWr(energyArray);

			}

			

		}
	}

	

}