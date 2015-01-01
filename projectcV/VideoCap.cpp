/*@author gihan tharanga*/

#include "VideoCap.h"

#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "opencv2\objdetect\objdetect.hpp"
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;


int FaceDetector(string &classifier){

	//haarcascade_frontalface_alt2
	//string face = "C:/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml";

	CascadeClassifier face_cascade;
	string window = "Capture - face detection";

	if (!face_cascade.load(classifier)){
		cout << " Error loading file" << endl;
		return -1;
	}

	VideoCapture cap(0);
	//VideoCapture cap("C:/Users/lsf-admin/Pictures/Camera Roll/video000.mp4");

	if (!cap.isOpened())
	{
		cout << "exit" << endl;
		return -1;
	}

	//double fps = cap.get(CV_CAP_PROP_FPS);
	//cout << " Frames per seconds " << fps << endl;
	namedWindow(window, 1);
	long count = 0;

	string name = "gihan";
	while (true)
	{
		vector<Rect> faces;
		Mat frame;
		Mat graySacleFrame;
		Mat cropImg;

		cap >> frame;
		//cap.read(frame);
		count = count + 1;//count frames;

		if (!frame.empty()){

			//convert image to gray scale and equalize
			//cvtColor(frame, graySacleFrame, CV_BGR2GRAY);
			//equalizeHist(graySacleFrame, graySacleFrame);

			face_cascade.detectMultiScale(frame, faces, 1.1, 3, 0, cv::Size(190, 190), cv::Size(200, 200));

			cout << faces.size() << " faces detected" << endl;
			std::string frameset = std::to_string(count);
			std::string faceset = std::to_string(faces.size());

			int width = 0, height = 0;

			//region of interest
			cv::Rect roi;

			for (int i = 0; i < faces.size(); i++)
			{
				rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(255, 0, 255), 1, 8, 0);
				cout << faces[i].width << faces[i].height << endl;
				width = faces[i].width; height = faces[i].height;

				//select the roi
				roi.x = faces[i].x; roi.width = faces[i].width;
				roi.y = faces[i].y; roi.height = faces[i].height;

				//get the roi from orginal frame
				cropImg = frame(roi);
				//cv::imshow("ROI", cropImg);

				//cv::imwrite("E:/FDB/"+frameset+".jpg", cropImg);
			}

			std::string wi = std::to_string(width);
			std::string he = std::to_string(height);

			cv::putText(frame, "Frames: " + frameset, cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 255, 0), 1, CV_AA);
			cv::putText(frame, "Faces Detected: " + faceset, cvPoint(30, 60), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 255, 0), 1, CV_AA);
			cv::putText(frame, "Resolution " + wi + " x " + he, cvPoint(30, 90), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 255, 0), 1, CV_AA);
			//cv::putText(frame, "size : " +)

			cv::imshow(window, frame);
		}
		if (waitKey(30) >= 0) break;
	}
}

int videoCapturing()
{
	VideoCapture cap(0);

	if (!cap.isOpened())
	{
		cout << "camera opened" << endl;
		return -1;
	}

	Mat edges;
	namedWindow("edges", 1);
	for (;;)
	{
		Mat frame;
		cap >> frame; // get a new frame from camera
		if (!frame.empty()) {
			cvtColor(frame, edges, CV_HLS2BGR);
			GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
			Canny(edges, edges, 0, 30, 3);
			imshow("edges", edges);
		}
		if (waitKey(30) == 10) break;
	}

	return 0;
}

int videoCapOriginal()
{
	/*camera*/
	VideoCapture cap(0);

	/*initiallize*/
	if (!cap.isOpened())
	{
		cout << "exit" << endl;
		return -1;
	}

	/*create window for display video*/
	namedWindow("window", 1);

	while (true)
	{
		/*reads each frame and assign to mat*/
		Mat frame;
		cap.read(frame);

		if (!frame.empty()){
			/*add to window*/
			imshow("window", frame);
		}
		if (waitKey(30) >= 0) break;
	}
	return 0;
}