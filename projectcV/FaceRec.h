/*@author gihan tharanga*/

#include <iostream>
#include <string>

//include opencv core
#include "opencv2\core\core.hpp"
#include "opencv2\contrib\contrib.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2\opencv.hpp"

//file handling
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;

static Mat norm_0_255(InputArray _src) {
	Mat src = _src.getMat();
	// Create and return normalized image:
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';'){
	std::ifstream file(filename.c_str(), ifstream::in);

	if (!file){
		string error = "no valid input file";
		CV_Error(CV_StsBadArg, error);
	}

	string line, path, label;
	while (getline(file, line))
	{
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, label);
		if (!path.empty() && !label.empty()){
			images.push_back(imread(path, 0));
			labels.push_back(atoi(label.c_str()));
		}
	}
}

void eigenFaceTrainer(){
	vector<Mat> images;
	vector<int> labels;

	try{
		string filename = "E:/at.txt";
		read_csv(filename, images, labels);

		cout << "size of the images is " << images.size() << endl;
		cout << "size of the labes is " << labels.size() << endl;
		cout << "Training begins...." << endl;
	}
	catch (cv::Exception& e){
		cerr << " Error opening the file " << e.msg << endl;
		exit(1);
	}

	//create algorithm eigenface recognizer
	Ptr<FaceRecognizer>  model = createEigenFaceRecognizer();
	//train data
	model->train(images, labels);

	model->save("E:/FDB/yaml/eigenface.yml");

	cout << "Training finished...." << endl;
	////get eigenvalue of eigenface model
	//Mat eigenValue = model->getMat("eigenvalues");

	//////get eigenvectors display(eigenface)
	//Mat w = model->getMat("eigenvectors");

	//////get the sample mean from the training data
	//Mat mean = model->getMat("mean");

	//////save or display
	//imshow("mean", norm_0_255(mean.reshape(1,images[0].rows)));
	////imwrite(format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));

	////display or save eigenfaces
	//for (int i = 0; i < min(10, w.cols); i++)
	//{
	//	string msg = format("Eigenvalue #%d = %.5f", i, eigenValue.at<double>(i));
	//	cout << msg << endl;

	//	//get the eigenvector #i
	//	Mat ev = w.col(i).clone();

	//	// Reshape to original size & normalize to [0...255] for imshow.
	//	Mat grayscale = norm_0_255(ev.reshape(1, height));
	//	// Show the image & apply a Jet colormap for better sensing.
	//	Mat cgrayscale;
	//	applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
	//	//display or save
	//	imshow(format("eigenface_%d", i), cgrayscale);
	//	//imwrite(format("%s/eigenface_%d.png", output_folder.c_str(), i), norm_0_255(cgrayscale));
	//}

	////display or save image reconstruction
	//for (int num_components = min(w.cols, 10); num_components < min(w.cols, 300); num_components += 15)
	//{
	//	// slice the eigenvectors from the model
	//	Mat evs = Mat(w, Range::all(), Range(0, num_components));
	//	Mat projection = subspaceProject(evs, mean, images[0].reshape(1, 1));
	//	Mat reconstruction = subspaceReconstruct(evs, mean, projection);
	//	// Normalize the result:
	//	reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));
	//	// Display or save:
	//	imshow(format("eigenface_reconstruction_%d", num_components), reconstruction);
	//	//imwrite(format("%s/eigenface_reconstruction_%d.png", output_folder.c_str(), num_components), reconstruction);	
	//		
	//}
	waitKey(10000);
}

void fisherFaceTrainer(){

	vector<Mat> images;
	vector<int> labels;

	try{
		string filename = "E:/at.txt";
		read_csv(filename, images, labels);

		cout << "size of the images is " << images.size() << endl;
		cout << "size of the labes is " << labels.size() << endl;
		cout << "Training begins...." << endl;
	}
	catch (cv::Exception& e){
		cerr << " Error opening the file " << e.msg << endl;
		exit(1);
	}

	Ptr<FaceRecognizer> model = createFisherFaceRecognizer();

	model->train(images, labels);

	int height = images[0].rows;

	model->save("E:/FDB/yaml/fisherface.yml");

	cout << "Training finished...." << endl;

	Mat eigenvalues = model->getMat("eigenvalues");
	// And we can do the same to display the Eigenvectors (read Eigenfaces):
	Mat W = model->getMat("eigenvectors");
	// Get the sample mean from the training data
	Mat mean = model->getMat("mean");
	imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));
	//imwrite(format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));

	// Display or save the first, at most 16 Fisherfaces:
	/*for (int i = 0; i < min(16, W.cols); i++) {
	string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
	cout << msg << endl;
	// get eigenvector #i
	Mat ev = W.col(i).clone();
	// Reshape to original size & normalize to [0...255] for imshow.
	Mat grayscale = norm_0_255(ev.reshape(1, height));
	// Show the image & apply a Bone colormap for better sensing.
	Mat cgrayscale;
	applyColorMap(grayscale, cgrayscale, COLORMAP_BONE);
	// Display or save:
	//imshow(format("fisherface_%d", i), cgrayscale);
	//imwrite(format("%s/fisherface_%d.png", output_folder.c_str(), i), norm_0_255(cgrayscale));
	}

	// Display or save the image reconstruction at some predefined steps:
	for (int num_component = 0; num_component < min(16, W.cols); num_component++) {
	// Slice the Fisherface from the model:
	Mat ev = W.col(num_component);
	Mat projection = subspaceProject(ev, mean, images[0].reshape(1, 1));
	Mat reconstruction = subspaceReconstruct(ev, mean, projection);
	// Normalize the result:
	reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));
	// Display or save:
	imshow(format("fisherface_reconstruction_%d", num_component), reconstruction);
	//imwrite(format("%s/fisherface_reconstruction_%d.png", output_folder.c_str(), num_component), reconstruction);
	}*/

	waitKey(10000);
}

void LBPHFaceTrainer(){

	vector<Mat> images;
	vector<int> labels;

	try{
		string filename = "E:/at.txt";
		read_csv(filename, images, labels);

		cout << "size of the images is " << images.size() << endl;
		cout << "size of the labes is " << labels.size() << endl;
		cout << "Training begins...." << endl;
	}
	catch (cv::Exception& e){
		cerr << " Error opening the file " << e.msg << endl;
		exit(1);
	}

	//lbph face recognier model
	Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();

	//training images with relevant labels 
	model->train(images, labels);

	//save the data in yaml file
	model->save("E:/FDB/yaml/LBPHface.yml");

	cout << "training finished...." << endl;

	waitKey(10000);
}

//lbpcascades works in lbphrecognier as fast as haarcascades 
int  FaceRecognition(){

	//load pre-trained data sets
	Ptr<FaceRecognizer>  model = createLBPHFaceRecognizer();
	model->load("E:/FDB/yaml/LBPHface.yml");

	Mat testSample = imread("E:/db/s41/5.pgm", 0);

	int img_width = testSample.cols;
	int img_height = testSample.rows;


	//lbpcascades/lbpcascade_frontalface.xml
	string classifier = "C:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml";

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

	while (true)
	{
		vector<Rect> faces;
		Mat frame;
		Mat graySacleFrame;
		Mat original;

		cap >> frame;
		//cap.read(frame);
		count = count + 1;//count frames;

		if (!frame.empty()){

			//clone from original frame
			original = frame.clone();

			//convert image to gray scale and equalize
			cvtColor(original, graySacleFrame, CV_BGR2GRAY);
			//equalizeHist(graySacleFrame, graySacleFrame);

			//detect face in gray image
			face_cascade.detectMultiScale(graySacleFrame, faces, 1.1, 3, 0, cv::Size(90, 90));

			//number of faces detected
			cout << faces.size() << " faces detected" << endl;
			std::string frameset = std::to_string(count);
			std::string faceset = std::to_string(faces.size());

			int width = 0, height = 0;

			//region of interest
			//cv::Rect roi;

			//person name
			string Pname = "";

			for (int i = 0; i < faces.size(); i++)
			{
				//region of interest
				Rect face_i = faces[i];

				//crop the roi from grya image
				Mat face = graySacleFrame(face_i);

				//resizing the cropped image to suit to database image sizes
				Mat face_resized;
				cv::resize(face, face_resized, Size(img_width, img_height), 1.0, 1.0, INTER_CUBIC);

				//recognizing what faces detected
				int label = -1; double confidence = 0;
				model->predict(face_resized, label, confidence);

				cout << " confidencde " << confidence << endl;

				//drawing green rectagle in recognize face
				rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
				string text = "Detected";
				if (label == 40){
					//string text = format("Person is  = %d", label);
					Pname = "gihan";
				}
				else{
					Pname = "unknown";
				}


				int pos_x = std::max(face_i.tl().x - 10, 0);
				int pos_y = std::max(face_i.tl().y - 10, 0);

				//name the person who is in the image
				putText(original, text, Point(pos_x, pos_y), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
				//cv::imwrite("E:/FDB/"+frameset+".jpg", cropImg);

			}


			putText(original, "Frames: " + frameset, Point(30, 60), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
			putText(original, "Person: " + Pname, Point(30, 90), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
			//display to the winodw
			cv::imshow(window, original);

			//cout << "model infor " << model->getDouble("threshold") << endl;

		}
		if (waitKey(30) >= 0) break;
	}
}