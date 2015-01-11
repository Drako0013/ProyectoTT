#ifndef SIMPLEFLOW_H_
#define SIMPLEFLOW_H_

#include <algorithm>
#include <cmath>

#include "opencv2/imgproc/imgproc.hpp"

#include "frame.h"

class SimpleFlow{
public:
	int GetEnergy(Frame &f1, int x1, int y1, Frame &f2, int x2, int y2);
	double GetWt(std::vector<int> &energyArray);
	double GetWd(int x0, int y0, int x, int y);
	double GetWc(Frame &f1, int x0, int y0, int x, int y);

private:
	int neighborhoodSize;
	double rd;
	double rc;
}


#endif