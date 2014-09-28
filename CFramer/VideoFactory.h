#include "opencv2/core/core.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

class VideoFactory{
	public:
		//Video original
		VideoFactory();
		void agregaFrame(cv::Mat frame);
		void saveOutput();
		

	private:
		cv::VideoCapture output;

};