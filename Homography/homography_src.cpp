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

		for (int j = 0; j < 8; j++) { //將4組點資料存成8*8矩陣
			Point_matrix[2 * i][j] = row1[j];
			Point_matrix[2 * i + 1][j] = row2[j];
		}
	}
	Mat Point_data(8, 8, CV_64FC1, Point_matrix); //到此以上為做出8*8矩陣

	double _dst[8];
	for (int i = 0; i < 4; i++) {
		_dst[2 * i] = dst[i].x;
		_dst[2 * i + 1] = dst[i].y;
	}
	Mat Point_target(8, 1, CV_64FC1, _dst);

	Mat h8 = Point_data.inv() * Point_target;

	double H8[9]; //3*3 H矩陣
	for (int i = 0; i < 8; i++) {
		H8[i] = h8.at<double>(i, 0);
	}
	H8[8] = 1;     //右下角為1
	Mat H(3, 3, CV_64FC1, H8);
	return H.clone();
}

Mat do_transform(Mat src, Mat H) {
	Mat img(768, 1024, CV_8UC3, Scalar(0, 0, 0)); //宣告一張新圖，用回推到原圖的方式避免掉空洞問題

	for (int i = 0; i < img.size().height; i++) { //跑高
		for (int j = 0; j < img.size().width; j++) { //跑寬
			double x_pixel[3] = { j, i, 1 };
			Mat Xresult(3, 1, CV_64FC1, x_pixel);
			Mat X = H.inv()*Xresult;
			double x = cvRound(X.at<double>(0, 0) / X.at<double>(2, 0));	//正規化
			double y = cvRound(X.at<double>(1, 0) / X.at<double>(2, 0));
			if (x < src.size().width && y < src.size().height && x >= 0 && y >= 0) {
				img.at<Vec3b>(i, j) = src.at<Vec3b>(y, x);
			}
		}
	}
	return img.clone();
}

int main() {

	Mat img_ori = imread("origin_pic.JPG");
	Mat img_gt = imread("GroundTruth.JPG");
	Mat img_homo(img_gt.rows, img_gt.cols, CV_8UC3, Scalar(0, 0, 0));

	/* Correspondence Points */
	vector<Point2f> img_ori_points = { Point2f(559, 529), Point2f(2041, 349), Point2f(573, 1733), Point2f(2053, 1887) }; //直接抓取原圖角落四個點
	vector<Point2f> img_homo_points = { Point2f(0, 0), Point2f(1024, 0), Point2f(0, 768), Point2f(1024, 768) };          //確保不共線

	/* OpenCV's API, for comparison use */
	Mat H_opencv = findHomography(img_ori_points, img_homo_points);
	/* my self-implemented function for finding H */
	Mat H = find_H_matrix(img_ori_points, img_homo_points);

	cout << "Self-implemented H matrix:" << endl;
	cout << H << endl;
	cout << "OpenCV H matrix:" << endl;
	cout << H_opencv << endl;

	img_homo = do_transform(img_ori, H);

	imwrite("Homography.JPG", img_homo);
	imshow("After homography", img_homo);

	Mat img_diff = img_gt - img_homo;

	imwrite("Diff.BMP", img_diff);
	imshow("Diff", img_diff);

	waitKey(0);
	return 0;
}