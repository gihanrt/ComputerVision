/*author gihan tharanga*/

#include <iostream>
#include <string>

#include "opencv2\core\core.hpp"
#include "opencv2\contrib\contrib.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2\opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	
	 
	Mat image = imread("C://Users/lsf-admin/Desktop/UN/boganbara/DSC_0508.jpg",CV_LOAD_IMAGE_COLOR);
	if (!image.data){
		cout << "coudnot open" << endl;
		return -1;
	}

	namedWindow("window", WINDOW_NORMAL);
	imshow("window", image);
	
	waitKey(10000);
	destroyWindow("window");
	system("pause");
	return 0;
}