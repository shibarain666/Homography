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

using namespace cv;
using namespace std;

Mat find_H_matrix(vector<Point2f> src, vector<Point2f> dst) {
	double Point_matrix[8][8];
	for (int i = 0; i < 4; i++) {
		double row1[8] = { src[i].x, src[i].y, 1,        0,        0, 0, -dst[i].x * src[i].x, -dst[i].x * src[i].y };
		double row2[8] = { 0       ,        0, 0, src[i].x, src[i].y, 1, -dst[i].y * src[i].x, -dst[i].y * src[i].y };

		for (int j = 0; j < 8; j++) { /* save 4 point data to 8*8 Matrix */
			Point_matrix[2 * i][j] = row1[j];
			Point_matrix[2 * i + 1][j] = row2[j];
		}
	}

	/* 8*8 Matrix */
	Mat Point_data(8, 8, CV_64FC1, Point_matrix); 

	double _dst[8];
	for (int i = 0; i < 4; i++) {
		_dst[2 * i] = dst[i].x;
		_dst[2 * i + 1] = dst[i].y;
	}
	Mat Point_target(8, 1, CV_64FC1, _dst);

	Mat h8 = Point_data.inv() * Point_target;

	double H8[9]; /* 3*3 H Matrix */
	for (int i = 0; i < 8; i++) {
		H8[i] = h8.at<double>(i, 0);
	}
	H8[8] = 1;
	Mat H(3, 3, CV_64FC1, H8);
	return H.clone();
}

Mat do_transform(Mat src, Mat H) {

	Mat tgt(768, 1024, CV_8UC3, Scalar(0, 0, 0));

	/* Target-to-Source */
	for (int i = 0; i < tgt.size().height; i++) {
		for (int j = 0; j < tgt.size().width; j++) {
			Mat Xresult = (Mat_<double>(3, 1) << j, i, 1.0);
			Mat X = H.inv()*Xresult;
			int x = cvRound(X.at<double>(0, 0) / X.at<double>(2, 0));    /* normalized */
			int y = cvRound(X.at<double>(1, 0) / X.at<double>(2, 0));
			if (x < src.size().width && y < src.size().height && x >= 0 && y >= 0) {
				tgt.at<Vec3b>(i, j) = src.at<Vec3b>(y, x);
			}
		}
	}
	return tgt.clone();
}

int main() {

	Mat img_ori = imread("origin_pic.JPG");
	Mat img_gt = imread("GroundTruth.JPG");
	Mat img_homo(img_gt.rows, img_gt.cols, CV_8UC3, Scalar(0, 0, 0));

	/* Correspondence Points */
	vector<Point2f> img_ori_points = { Point2f(559, 529), Point2f(2041, 349), Point2f(573, 1733), Point2f(2053, 1887) }; /* 4 sets of points on source image */
	vector<Point2f> img_homo_points = { Point2f(0, 0), Point2f(1024, 0), Point2f(0, 768), Point2f(1024, 768) };          /* 4 sets of points on target image */

	/* OpenCV's API, for comparison use */
	Mat H_opencv = findHomography(img_ori_points, img_homo_points);
	/* my self-implemented function for finding H */
	Mat H = find_H_matrix(img_ori_points, img_homo_points);

	cout << "Self-implemented H matrix:" << endl;
	cout << H << endl;
	cout << "OpenCV H matrix:" << endl;
	cout << H_opencv << endl;
	cout << "Self-implemented inverse H matrix:" << endl;
	cout << H.inv() << endl;

	img_homo = do_transform(img_ori, H);

	imwrite("Homography.JPG", img_homo);
	imshow("After homography", img_homo);

	Mat img_diff = img_gt - img_homo;

	imwrite("Diff.BMP", img_diff);
	imshow("Diff", img_diff);

	waitKey(0);
	return 0;
}