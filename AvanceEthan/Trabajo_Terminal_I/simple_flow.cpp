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

void SimpleFlow::CalculateFlow(cv::Mat& vel_x, cv::Mat& vel_y) {

	const int n = SimpleFlow::NeighborhoodSize;

	int rows = frames[0]->Rows();
	int cols = frames[0]->Columns();

	double e;
	double E[n * 2 + 1][n * 2 + 1]; // Due to obvious limitations of arrays, E(u, v) is represented by E[u + n][v + n]

	vel_x = cv::Mat(rows, cols, CV_64F);
	vel_y = cv::Mat(rows, cols, CV_64F);
	for (int x = 0; x < rows; ++x) {
		double* ptr_x = vel_x.ptr<double>(x);
		double* ptr_y = vel_y.ptr<double>(x);
		for (int y = 0; y < cols; ++y, ++ptr_x, ++ptr_y) {
			for (int u = -n; u <= n; ++u) {
				for (int v = -n; v <= n; ++v) {
					if (x + u < 0 || x + u >= rows || y + v < 0 || y + v >= cols) {
						E[u + n][v + n] = std::numeric_limits<double>::max();
					} else {
						E[u + n][v + n] = 0.0;
						e = GetEnergy(*frames[0], x, y, *frames[1], u, v);
						for (int i = -n; i <= n; ++i) {
							for (int j = -n; j <= n; ++j) {
								if (x + i >= 0 && x + i < rows && y + j >= 0 && y + j < cols) {
									E[u + n][v + n] += GetWd(x, y, x + i, y + j) * GetWc(*frames[0], x, y, x + i, y + j) * e;
								}
							}
						}
					}
				}
			}
			double me = std::numeric_limits<double>::max();
			// TODO: Sub-pixel estimation (low priority)
			for (int u = -n; u <= n; ++u) {
				for (int v = -n; v <= n; ++v) {
					if (E[u][v] < me) {
						me = E[u][v];
						*ptr_x = u;
						*ptr_y = v;
					}
				}
			}
		}
	}

	// TODO: Bi-filter results; calculate occlusion

}