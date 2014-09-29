#include "opencv2/core/core.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

class VideoFactory{
	public:
		//Video original
		VideoFactory(int widht, int height, int fps);
		VideoFactory(std::string &nombre, int width, int height, int fps);
		//~VideoFactory();
		void agregaFrame(cv::Mat &frame);
		

	private:
		std::string nombreArchivo;
		cv::VideoWriter output;

};