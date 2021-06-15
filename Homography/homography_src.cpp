/**************************************************
File: homography_src.cpp

Author: Jerry Cheng

Date: 2021/06/03

***************************************************/

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>

cv::Mat find_H_matrix(std::vector<cv::Point2f> src, std::vector<cv::Point2f> tgt) {

	std::cout << "calculating H matrix..." << std::endl;

	double Point_matrix[8][8];
	for (int i = 0; i < 4; i++) {
		double row1[8] = { src[i].x, src[i].y, 1,        0,        0, 0, -tgt[i].x * src[i].x, -tgt[i].x * src[i].y };
		double row2[8] = { 0       ,        0, 0, src[i].x, src[i].y, 1, -tgt[i].y * src[i].x, -tgt[i].y * src[i].y };

		/* save 4 point data to 8*8 Matrix */
		memcpy(Point_matrix[2 * i], row1, sizeof(row1));
		memcpy(Point_matrix[2 * i + 1], row2, sizeof(row2));
	}

	/* 8*8 Matrix */
	cv::Mat Point_data(8, 8, CV_64FC1, Point_matrix);
	cv::Mat Point_target = (cv::Mat_<double>(8, 1) << tgt[0].x, tgt[0].y, tgt[1].x, tgt[1].y, tgt[2].x, tgt[2].y, tgt[3].x, tgt[3].y);
	cv::Mat h8 = Point_data.inv() * Point_target;

	/* 3*3 H Matrix */
	double H8[9]; 
	for (int i = 0; i < 8; i++) {
		H8[i] = h8.at<double>(i, 0);
	}
	H8[8] = 1.0;
	cv::Mat H(3, 3, CV_64FC1, H8);

	return H.clone();
}

cv::Mat do_transform(cv::Mat src, cv::Mat H) {

	std::cout << "doing transform..." << std::endl;

	cv::Mat tgt(src.rows, src.cols, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat X, Xresult;

	/* Target-to-Source */
	for (int i = 0; i < tgt.rows; i++) {
		for (int j = 0; j < tgt.cols; j++) {
			Xresult = (cv::Mat_<double>(3, 1) << j, i, 1.0);
			X = H.inv()*Xresult;
			int x = cvRound(X.at<double>(0, 0) / X.at<double>(2, 0));    /* normalized */
			int y = cvRound(X.at<double>(1, 0) / X.at<double>(2, 0));    /* normalized */

			if (x < 0 || y < 0) {
				continue;
			}
			if (x > src.cols - 1 || y > src.rows - 1) {
				continue;
			}

			tgt.at<cv::Vec3b>(i, j) = src.at<cv::Vec3b>(y, x);
		}
	}

	return tgt.clone();
}

int main(int argc, char *argv[]) {

	cv::Mat img_ori = cv::imread("src_img.jpg");
	cv::Mat img_homo(img_ori.rows, img_ori.cols, CV_8UC3, cv::Scalar(0, 0, 0));

	/* Correspondence Points */
	std::vector<cv::Point2f> img_ori_points = {    /* 4 sets of points on source image */
		cv::Point2f(559, 529), 
		cv::Point2f(2041, 349), 
		cv::Point2f(573, 1733), 
		cv::Point2f(2053, 1887) }; 
	std::vector<cv::Point2f> img_homo_points = {    /* 4 sets of points on target image */ 
		cv::Point2f(0, 0), 
		cv::Point2f(1023, 0), 
		cv::Point2f(0, 767), 
		cv::Point2f(1023, 767) };   

	/* OpenCV's API, for comparison use */
	cv::Mat H_opencv = findHomography(img_ori_points, img_homo_points);

	double t = (double)cv::getTickCount();

	/* my self-implemented function for finding H */
	cv::Mat H = find_H_matrix(img_ori_points, img_homo_points);
	img_homo = do_transform(img_ori, H);

	t = (double)cv::getTickCount() - t;
	std::cout << "time:" << t/(cv::getTickFrequency()) << std::endl;

	std::cout << "Self-implemented H matrix:" << std::endl;
	std::cout << H << std::endl;
	std::cout << "OpenCV's API H matrix:" << std::endl;
	std::cout << H_opencv << std::endl;

	cv::namedWindow("Source image", cv::WINDOW_NORMAL);
	cv::resizeWindow("Source image", 640, 480);
	imshow("Source image", img_ori);

	cv::namedWindow("After transform", cv::WINDOW_NORMAL);
	cv::resizeWindow("After transform", 640, 480);
	imwrite("Transformed_img.jpg", img_homo);
	imshow("After transform", img_homo);

	char key = (char)cv::waitKey(0);
	if (key == 'q') {
		cv::destroyWindow("Source image");
		cv::destroyWindow("After transform");
	}

	return 0;
}