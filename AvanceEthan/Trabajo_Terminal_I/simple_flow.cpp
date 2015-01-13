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

void SimpleFlow::CalculateFlow(cv::Mat& vel_x, cv::Mat& vel_y) {

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