#include <iostream>
#include <string>

#include "opencv2\core\core.hpp"
#include "opencv2\contrib\contrib.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2\opencv.hpp"

using namespace std;
using namespace cv;

int smoothingImage(){

	Mat image, dst;
	image = imread("E://Lenna.png",1);

	if (image.empty()){
		cout << "image is not loaded...." << endl;
		return -1;
	}

	cout << "Resolution is : "<< image.cols << " x " << image.rows << endl;
	namedWindow("src", 1);
	imshow("src", image);

	for (int i = 1; i < 5; i++)
	{
		string value = std::to_string(i);
		string name = value + " x " + value;
		blur(image, dst, Size(i, i), Point(-1, -1), 4);
		namedWindow(name,1);
		imshow(name, dst);
	}

	waitKey(0);
	destroyAllWindows();
	return 0;
}

int edgeDetectionsCanny(){
	VideoCapture capture;
	Mat original, gray, edge, detectEdges;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	RNG rng(12345);

	capture.open(0);

	if (!capture.isOpened()){
		cout << " camera opened error" << endl;
		return -1;
	}

	namedWindow("edge detection", 1);

	while (true)
	{
		capture >> original;

		if (!original.empty())
		{
			/*convert image to gray scale*/
			cvtColor(original, gray, CV_BGR2GRAY);
			/*blur image */
			blur(original, edge, Size(3, 3));
			/*detecting edges using canny edge detection*/
			Canny(edge, detectEdges, 10, 100, 3, true);

			findContours(detectEdges, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

			Mat drawing = Mat::zeros(detectEdges.size(), CV_8UC3);

			for (int i = 0; i< contours.size(); i++)
			{
				drawContours(original, contours, i, Scalar(0, 0, 255), 1, 8, hierarchy, 0, Point());
			}
			/*showing frames in window*/
			imshow("edge detection", original);
		}
		if (waitKey(30) >= 0) break;
	}
}