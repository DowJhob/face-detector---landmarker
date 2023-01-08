//  Created by Linzaer on 2019/11/15.
//  Copyright © 2019 Linzaer. All rights reserved.

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <opencv2/face.hpp>
//#include <opencv2/face/facemarkLBF.hpp>


#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>




#include "drawLandmarks.hpp"
#include "landmark/landmarker.h"

//#include "landmark/Peppa_Pig_Face_Landmark/Peppa_Pig_Face_Landmark.h"



#include "UltraFace.hpp"

using namespace std;
//using namespace mirror;
string root_path = "D:/Users/Bimbo/Documents/GitHub/drowsy-detectors/face-detector---landmarker";

#define meanC 10

int num = 70;
chrono::steady_clock::time_point start;

double meanR = 0;
double meanL = 0;

double meanAccR = 0;
double meanAccL = 0;

int meanCounter = 0;

void calcMean(double R, double L)
{
	meanCounter++;

	meanAccR += R;
	meanAccL += L;

	if (meanCounter > meanC)
	{
		meanR = meanAccR / meanC;
		meanL = meanAccL / meanC;
		meanAccR = 0;
		meanAccL = 0;

		meanCounter = 0;
	}
}

double calculate_ear_ZQ(vector<cv::Point>& eye)
{
	double A, B, C;
	double temp_x[6], temp_y[6];
	for (int i = 0; i < 6; i++)
	{
		temp_x[i] = (double)eye[i].x;
		temp_y[i] = (double)eye[i].y;
	}

	A = temp_x[5] - temp_x[1];
	A *= A;
	double Ay = temp_y[5] - temp_y[1];
	Ay *= Ay;
	A = sqrt(A + Ay);

	B = temp_x[4] - temp_x[2];
	B *= B;
	double By = temp_y[4] - temp_y[2];
	By *= By;
	B = sqrt(B + By);

	C = temp_x[3] - temp_x[0];
	C *= C;
	double Cy = temp_y[3] - temp_y[0];
	Cy *= Cy;
	C = sqrt(C + Cy);

	return (A + B) / (2 * C);
}
double calculate_ear_pfld(vector<cv::Point>& eye)
{
	double A, B, AB, C;
	double temp_x[8], temp_y[8];
	for (int i = 0; i < 8; i++)
	{
		temp_x[i] = (double)eye[i].x;
		temp_y[i] = (double)eye[i].y;
	}

	A = temp_x[7] - temp_x[1];
	A *= A;
	double Ay = temp_y[7] - temp_y[1];
	Ay *= Ay;
	A = sqrt(A + Ay);

	B = temp_x[5] - temp_x[3];
	B *= B;
	double By = temp_y[5] - temp_y[3];
	By *= By;
	B = sqrt(B + By);


	AB = temp_x[6] - temp_x[2];
	AB *= AB;
	double ABy = temp_y[6] - temp_y[2];
	ABy *= ABy;
	AB = sqrt(AB + ABy);





	C = temp_x[4] - temp_x[0];
	C *= C;
	double Cy = temp_y[4] - temp_y[0];
	Cy *= Cy;
	C = sqrt(C + Cy);

	return (A + B + AB) / (3 * C);
}

void pfld_processing(std::vector<cv::Point2f>& keypoints)
{
	vector<cv::Point> left_eye(&keypoints[60], &keypoints[68]);
	vector<cv::Point> right_eye(&keypoints[68], &keypoints[76]);
	calcMean(calculate_ear_pfld(right_eye), calculate_ear_pfld(left_eye));
}

void zq_processing(std::vector<cv::Point2f>& keypoints)
{
	vector<cv::Point> left_eye(&keypoints[52], &keypoints[58]);
	vector<cv::Point> right_eye(&keypoints[58], &keypoints[64]);
	auto it = left_eye.begin();
	left_eye.insert(it + 2, keypoints[72]);
	auto it2 = left_eye.begin();
	left_eye.insert(it2 + 6, keypoints[73]);

	it = right_eye.begin();
	right_eye.insert(it + 2, keypoints[75]);
	it2 = right_eye.begin();
	right_eye.insert(it2 + 6, keypoints[76]);
	//calcMean(calculate_ear_ZQ(right_eye), calculate_ear_ZQ(left_eye));
	calcMean(calculate_ear_pfld(right_eye), calculate_ear_pfld(left_eye));
}

void peppa_processing(std::vector<cv::Point2f>& keypoints)
{
	vector<cv::Point> left_eye(&keypoints[60], &keypoints[68]);
	vector<cv::Point> right_eye(&keypoints[68], &keypoints[76]);
	calcMean(calculate_ear_pfld(right_eye), calculate_ear_pfld(left_eye));
}

void show(cv::Mat& frame)
{
	auto ear_ratioL = to_string(meanL);
	cv::putText(frame,
		"L " + ear_ratioL, cv::Point(20, 30),
		cv::FONT_HERSHEY_SIMPLEX,
		1, cv::Scalar(0, 0, 255));

	auto ear_ratioR = to_string(meanR);
	cv::putText(frame,
		"R " + ear_ratioR, cv::Point(20, 70),
		cv::FONT_HERSHEY_SIMPLEX,
		1, cv::Scalar(0, 255, 0));

	cv::putText(frame,
		to_string(num), cv::Point(20, 400),
		cv::FONT_HERSHEY_SIMPLEX,
		0.5, cv::Scalar(0, 255, 0));

	auto end = chrono::steady_clock::now();
	chrono::duration<double> elapsed = end - start;
	cout << "all time: " << elapsed.count() << " s" << endl;
	cv::imshow("UltraFace", frame);
}

int main(int argc, char** argv)
{
	//string mnn_path = root_path + "/data/models/version-slim/slim-320.mnn";
	//string mnn_path = root_path + "/data/models/slim-320-quant-ADMM-50.mnn";
	string mnn_path = root_path + "/data/models/version-RFB/RFB-320.mnn";
	//string mnn_path = root_path + "/data/models/RFB-320-quant-ADMM-32.mnn";
	//string mnn_path = root_path + "/data/models/RFB-320-quant-KL-5792.mnn";
	UltraFace ultraface(mnn_path, 320, 240, 1, 0.95); // config model input

	auto stream = cv::VideoCapture(0, cv::CAP_MSMF);






	//mirror::LandmarkerFactory* landmarker_factory_ = new mirror::ZQLandmarkerFactory();
	//mirror::LandmarkerFactory* landmarker_factory_ = new mirror::PFLDLandmarkerFactory();
	mirror::LandmarkerFactory* landmarker_factory_ = new mirror::Peppa_PigLandmarkerFactory();
	mirror::Landmarker* landmarker_ = landmarker_factory_->CreateLandmarker();
	string lm_path = root_path + "/data/models";
	if (landmarker_->Init(lm_path.data()) != 0) {
		std::cout << "Init face landmarker failed." << std::endl;
		return 10000;
	}



	if (!stream.isOpened())
	{
		std::cerr << "Can't connect to camera\n";
		return EXIT_FAILURE;
	}


	std::cerr << "CAP_PROP_FRAME_HEIGHT\n" << stream.get(cv::CAP_PROP_FRAME_HEIGHT);

	char key = 0;
	constexpr int64 kTimeoutNs = 1000;
	std::vector<int> ready_index;
	while (key != 27) // 27 is the ascii code for ESC key.
	{
		//if (cv::VideoCapture::waitAny({ stream }, ready_index, kTimeoutNs))
		{
			cv::Mat frame, gray, canny, res;
			//stream.retrieve(frame);
			stream.read(frame);

			start = chrono::steady_clock::now();
			vector<FaceInfo> face_info;
			ultraface.detect(frame, face_info);

			for (auto& face : face_info)
			{
				cv::Point pt1(face.x1, face.y1);
				cv::Point pt2(face.x2, face.y2);
				//cv::format("%.4f", face.score);

				auto s = to_string(face.score);

				cv::putText(frame,
					s, pt1,
					cv::FONT_HERSHEY_SIMPLEX,
					0.5, cv::Scalar(0, 255, 0));

				//cv::cvtColor(frame, gray, cv::COLOR_RGBA2GRAY);
					//cv::Canny(gray, canny, 10, 100, 3);

				std::vector<cv::Point2f> keypoints;

				cv::cvtColor(frame, res, cv::COLOR_RGB2BGR);
				landmarker_->ExtractKeypoints(res, cv::Rect(pt1, pt2), &keypoints);

				int num_keypoints = static_cast<int>(keypoints.size());
				for (int j = 0; j < 98; ++j) {
					cv::circle(frame, keypoints[j], 1, cv::Scalar(255, 0, 0), 1);
				}

				cv::circle(frame, keypoints[num], 3, cv::Scalar(0, 0, 255), 1);

				//zq_processing(keypoints);
				//pfld_processing(keypoints);
				peppa_processing(keypoints);

				vector<cv::Point> left_eye(&keypoints[60], &keypoints[68]);
				vector<cv::Point> right_eye(&keypoints[68], &keypoints[76]);


				polylines(frame, left_eye, true, cv::Scalar(0, 255, 255), 1, 16);

				////drawPolyline(frame, left_eye, 0, 7, true);
				//polylines(frame, left_eye, true, cv::Scalar(0, 0, 255), 2, 16);
				//polylines(frame, right_eye, true, cv::Scalar(0, 255, 0), 2, 16);
				////drawPolyline(frame, right_eye, 0, 5, true);


				cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
			}
			show(frame);



			key = cv::waitKey(1);
			if (key == 'a')
				num++;
			if (key == 'z')
				num--;;

			// cv::imwrite(result_name, frame);

		}
	}
	stream.release();
	return 0;
}
