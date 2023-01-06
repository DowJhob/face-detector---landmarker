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

void drawPolyline(cv::Mat& image, dlib::full_object_detection landmarks, int start, int end, bool isClosed = false) {
	std::vector<cv::Point> points;
	for (int i = start; i <= end; i++) {
		points.push_back(cv::Point(landmarks.part(i).x(), landmarks.part(i).y()));
	}
	cv::polylines(image, points, isClosed, cv::Scalar(0, 255, 255), 2, 16);
}
void drawPolyline(cv::Mat& image, vector<dlib::point>  landmarks, int start, int end, bool isClosed = false) {
	std::vector<cv::Point> points;
	for (int i = start; i <= end; i++) {
		points.push_back(cv::Point(landmarks[i].x(), landmarks[i].y()));
	}
	cv::polylines(image, points, isClosed, cv::Scalar(0, 255, 255), 2, 16);
}
void convert_rect_CV2DLIB(vector<Rect>& cv_rect, vector<dlib::rectangle>& dlib_rect, int pos)
{
	Rect temp_cv;
	dlib::rectangle temp_dlib;

	temp_cv = cv_rect[pos];
	temp_dlib.set_left((long)temp_cv.x);
	temp_dlib.set_top((long)temp_cv.y);
	temp_dlib.set_right((long)(temp_cv.x + temp_cv.width));
	temp_dlib.set_bottom((long)(temp_cv.y + temp_cv.height));
	dlib_rect.push_back(temp_dlib);
}
void calculate_ear(cv::Mat& m, vector<cv::Point>& eye, double* ear)
{
	double A, B, C;
	double temp_x[6], temp_y[6];
	vector<Point2f> f;
	for (int i = 0; i < 6; i++)
	{
		temp_x[i] = (double)eye[i].x;
		temp_y[i] = (double)eye[i].y;

		f.push_back(cv::Point(eye[i].x, eye[i].y));
	}

	A = (temp_x[5] - temp_x[1]) * (temp_x[5] - temp_x[1]);
	A = sqrt(A + ((temp_y[5] - temp_y[1]) * (temp_y[5] - temp_y[1])));

	B = (temp_x[4] - temp_x[2]) * (temp_x[4] - temp_x[2]);
	B = sqrt(B + ((temp_y[4] - temp_y[2]) * (temp_y[4] - temp_y[2])));

	C = (temp_x[3] - temp_x[0]) * (temp_x[3] - temp_x[0]);
	C = sqrt(C + ((temp_y[3] - temp_y[0]) * (temp_y[3] - temp_y[0])));

	*ear = (A + B) / (2 * C);

	//drawPolyline(m, f, true);
}
void calculate_ear_pfld(cv::Mat& m, vector<cv::Point>& eye, double* ear)
{
	double A, B, C;
	double temp_x[8], temp_y[8];
	vector<Point2f> f;
	for (int i = 0; i < 8; i++)
	{
		temp_x[i] = (double)eye[i].x;
		temp_y[i] = (double)eye[i].y;

		f.push_back(cv::Point(eye[i].x, eye[i].y));
	}

	A = (temp_x[5] - temp_x[1]) * (temp_x[5] - temp_x[1]);
	A = sqrt(A + ((temp_y[5] - temp_y[1]) * (temp_y[5] - temp_y[1])));

	B = (temp_x[5] - temp_x[3]) * (temp_x[5] - temp_x[3]);
	B = sqrt(B + ((temp_y[5] - temp_y[3]) * (temp_y[5] - temp_y[3])));

	C = (temp_x[4] - temp_x[0]) * (temp_x[4] - temp_x[0]);
	C = sqrt(C + ((temp_y[4] - temp_y[0]) * (temp_y[4] - temp_y[0])));

	*ear = (A + B) / (2 * C);

	//drawPolyline(m, f, true);
}

int main(int argc, char** argv)
{

	//std::string lbf_path("D:/Users/Bimbo/Documents/GitHub/drowsy-detectors/UF-MNN/lbfmodel.yaml");
	//std::string kaz_path("D:/Users/Bimbo/Documents/GitHub/drowsy-detectors/UF-MNN/face_landmark_model.dat");
	//cv::Ptr<cv::face::Facemark> facemark = cv::face::createFacemarkLBF();
	//cv::Ptr<cv::face::Facemark> facemark = cv::face::createFacemarkKazemi();
	//cv::Ptr<cv::face::Facemark> facemark = cv::face::createFacemarkAAM();
	//facemark->loadModel(lbf_path);
	//facemark->loadModel(kaz_path);


	//dlib::shape_predictor sp;
	////dlib::deserialize("D:/Users/Bimbo/Documents/GitHub/drowsy-detectors/UF-MNN/shape_predictor_68_face_landmarks.dat") >> sp;
	//dlib::deserialize("D:/Users/Bimbo/Documents/GitHub/drowsy-detectors/UF-MNN/shape_predictor_68_face_landmarks_GTX.dat") >> sp; 


	//string mnn_path = "D:/Users/Bimbo/Documents/GitHub/drowsy-detectors/UF-MNN/model/version-slim/slim-320.mnn";
	//string mnn_path = "D:/Users/Bimbo/Documents/GitHub/drowsy-detectors/UF-MNN/model/version-slim/slim-320-quant-ADMM-50.mnn";
	string mnn_path = root_path + "/model/version-RFB/RFB-320.mnn";
	//string mnn_path = "D:/Users/Bimbo/Documents/GitHub/drowsy-detectors/UF-MNN/model/version-RFB/RFB-320-quant-ADMM-32.mnn";
	//string mnn_path = "D:/Users/Bimbo/Documents/GitHub/drowsy-detectors/UF-MNN/model/version-RFB/RFB-320-quant-KL-5792.mnn";
	UltraFace ultraface(mnn_path, 320, 240, 1, 0.95); // config model input

	auto stream = cv::VideoCapture(0, cv::CAP_MSMF);






	////mirror::LandmarkerFactory* landmarker_factory_ = new mirror::ZQLandmarkerFactory();
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

	int num = 32;
	char key = 0;
	constexpr int64 kTimeoutNs = 1000;
	std::vector<int> ready_index;
	while (key != 27) // 27 is the ascii code for ESC key.
	{
		//if (cv::VideoCapture::waitAny({ stream }, ready_index, kTimeoutNs))
		{
			cv::Mat frame, gray, canny;
			//stream.retrieve(frame);
			stream.read(frame);


			auto start = chrono::steady_clock::now();
			vector<FaceInfo> face_info;
			ultraface.detect(frame, face_info);
			std::vector<cv::Rect> faces;

			for (auto& face : face_info)
			{
				cv::Point pt1(face.x1, face.y1);
				cv::Point pt2(face.x2, face.y2);
				faces.push_back(cv::Rect(pt1, pt2));
				cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
				//cv::format("%.4f", face.score);

				auto s = to_string(face.score);

				cv::putText(frame,
					s, pt1,
					cv::FONT_HERSHEY_SIMPLEX,
					0.5, cv::Scalar(0, 255, 0));

			//cv::cvtColor(frame, gray, cv::COLOR_BGR2RGB);
				//cv::Canny(gray, canny, 10, 100, 3);

				std::vector<cv::Point2f> keypoints;
				landmarker_->ExtractKeypoints(frame, cv::Rect(pt1, pt2), &keypoints);

				int num_keypoints = static_cast<int>(keypoints.size());
				for (int j = 0; j < 68; ++j) {
					cv::circle(frame, keypoints[j], 1, cv::Scalar(0, 0, 255), 1);
				}

				//cv::circle(frame, keypoints[num], 3, cv::Scalar(0, 0, 255), 1);

				double ear;
				//  ZQCNN
				//vector<cv::Point> left_eye(&keypoints[52], &keypoints[58]);
				//vector<cv::Point> right_eye(&keypoints[58], &keypoints[64]);
				// 
				//  PFLD
				vector<cv::Point> left_eye(&keypoints[60], &keypoints[68]);
				vector<cv::Point> right_eye(&keypoints[68], &keypoints[76]);



				//drawPolyline(frame, left_eye, 0, 7, true);
				polylines(frame, left_eye, true, COLOR, 2, 16);
				polylines(frame, right_eye, true, COLOR, 2, 16);
				//drawPolyline(frame, right_eye, 0, 5, true);

				cv::Point p(keypoints[0].x - 150, keypoints[0].y);
				//calculate_ear(frame, left_eye, &ear);
				calculate_ear_pfld(frame, left_eye, &ear);
				auto ear_ratio = to_string(ear);
				cv::putText(frame,
					ear_ratio, p,
					cv::FONT_HERSHEY_SIMPLEX,
					1, cv::Scalar(0, 0, 255));

				cv::Point pp(keypoints[32].x + 10, keypoints[32].y);
				//calculate_ear(frame, right_eye, &ear);
				calculate_ear_pfld(frame, right_eye, &ear);
				ear_ratio = to_string(ear);
				cv::putText(frame,
					ear_ratio, pp,
					cv::FONT_HERSHEY_SIMPLEX,
					1, cv::Scalar(0, 255, 0));


			}
			cv::putText(frame,
				to_string(num), cv::Point(20, 20),
				cv::FONT_HERSHEY_SIMPLEX,
				0.5, cv::Scalar(0, 255, 0));
			//	




			auto end = chrono::steady_clock::now();
			chrono::duration<double> elapsed = end - start;
			cout << "all time: " << elapsed.count() << " s" << endl;
			cv::imshow("UltraFace", frame);
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
