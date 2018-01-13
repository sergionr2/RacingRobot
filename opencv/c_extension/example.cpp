#include <pybind11/pybind11.h>

#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <string>
#include <iostream>
#include <vector>
#include <math.h>       /* atan */
#include <cmath>        // std::abs

#define MAX_WIDTH 320
#define PI 3.14159265

#include "ndarray_converter.h"

using namespace Eigen;
namespace py = pybind11;

float REF_ANGLE = - PI / 2;
float MAX_ANGLE= PI / 4;

// cv::Mat image_test = 255.0 * cv::Mat::ones(240, 320, CV_32FC3);
cv::Mat image_test(240, 320, CV_32FC3, cv::Scalar(155.0, 155.0, 155.0));
int im_width = image_test.cols;

cv::Mat centroids;

cv::Mat resized = cv::Mat::ones(20, 80, CV_32FC3);

cv::Mat x1 = cv::Mat::ones(3, 80*20*3, CV_32FC1);
// cv::Mat z1 = cv::Mat::ones(3, 20, CV_32F);
// cv::Mat x1 = cv::Mat::ones(3, 80*20*3, CV_32F);
// cv::Mat x1 = cv::Mat::ones(3, 80*20*3, CV_32F);
cv::Mat z;

cv::Mat w1 = cv::Mat::ones(80*20*3, 20, CV_32F).t();
// Eigen::Map<Matrix<float, 3, 1>> w1_e;

cv::Mat b1 = cv::Mat::ones(20, 1, CV_32F).t();

cv::Mat w2 = cv::Mat::ones(20, 4, CV_32F).t();
cv::Mat b2 = cv::Mat::ones(4, 1, CV_32F).t();

cv::Mat w3 = cv::Mat::ones(4, 1, CV_32F).t();
cv::Mat b3 = cv::Mat::ones(1, 1, CV_32F).t();


// cv::Mat mat = cv::Mat::zeros(80, 20, CV_32F);

cv::Mat preprocessImage(cv::Mat image);

cv::Rect R0(0, 150, MAX_WIDTH, 50);
cv::Rect R1(0, 125, MAX_WIDTH, 50);
cv::Rect R2(0, 100, MAX_WIDTH, 50);
cv::Rect R3(0, 75, MAX_WIDTH, 50);
cv::Rect R4(0, 50, MAX_WIDTH, 50);
std::vector<cv::Rect> REGIONS({R1, R2, R3});


void init()
{
  image_test = cv::imread("test_sun.jpg");
  image_test.convertTo(image_test, CV_32FC3);
  // Matrix<float, 1, 20> w1_e;
  // cv::cv2eigen(w1, w1_e);
}

void setWeights(cv::Mat w1_, cv::Mat b1_, cv::Mat w2_, cv::Mat b2_, cv::Mat w3_, cv::Mat b3_)
{
  w1 = w1_.clone().t();
  w2 = w2_.clone().t();
  w3 = w3_.clone().t();
  b1 = b1_.clone().t();
  b2 = b2_.clone().t();
  b3 = b3_.clone().t();
  std::vector<cv::Mat> m1 {b1, b1, b1};
  cv::vconcat(m1, b1);
  std::vector<cv::Mat> m2 {b2, b2, b2};
  cv::vconcat(m2, b2);
  std::vector<cv::Mat> m3 {b3, b3, b3};
  cv::vconcat(m3, b3);
  // init();
}

cv::Mat preprocessImage(cv::Mat image)
{
  cv::resize(image, image, resized.size(), 0, 0, cv::INTER_AREA);
  image = image.reshape(1, 1);
  image /= 255.0;
  image -= 0.5;
  image *= 2;
  return image;
}

void relu(cv::Mat &x)
{
  for (int i = 0; i < x.rows; i++) {
    for (int j = 0; j < x.cols; j++) {
      if (x.at<float>(i, j) < 0) {
        x.at<float>(i, j) = 0.0;
      }
    }
  }
}


// Efficient implementation:
// void relu(cv::Mat &x)
// {
//     // accept only char type matrices
//   CV_Assert(x.depth() == CV_32F);
//   int channels = x.channels();
//   int nRows = x.rows;
//   int nCols = x.cols * channels;
//   if (x.isContinuous())
//   {
//       nCols *= nRows;
//       nRows = 1;
//   }
//   int i, j;
//   float* p;
//   for( i = 0; i < nRows; ++i)
//   {
//       p = x.ptr<float>(i);
//       for (j = 0; j < nCols; ++j)
//       {
//         if (p[j] < 0)
//         {
//           p[j] = 0;
//         }
//       }
//   }
// }

//
// void forward2()
// {
//   Map<Matrix<float, Dynamic, Dynamic, RowMajor>> w1_Eigen( (float *) w1.ptr(), w1.rows, w1.cols);
//   Map<Matrix<float, Dynamic, Dynamic, RowMajor>> b1_Eigen( (float *) b1.ptr(), b1.rows, b1.cols);
//   Map<Matrix<float, Dynamic, Dynamic, RowMajor>> w2_Eigen( (float *) w2.ptr(), w2.rows, w2.cols);
//   Map<Matrix<float, Dynamic, Dynamic, RowMajor>> b2_Eigen( (float *) b2.ptr(), b2.rows, b2.cols);
//   Map<Matrix<float, Dynamic, Dynamic, RowMajor>> w3_Eigen( (float *) w3.ptr(), w3.rows, w3.cols);
//   Map<Matrix<float, Dynamic, Dynamic, RowMajor>> b3_Eigen( (float *) b3.ptr(), b3.rows, b3.cols);
//   relu2(b3_Eigen);
// }

cv::Mat forward(cv::Mat x)
{
  cv::gemm(x, w1, 1, b1, 1, z, cv::GEMM_2_T);
  relu(z);
  // cv::threshold(z, z, 0, 0, 3);
  cv::gemm(z, w2, 1, b2, 1, z, cv::GEMM_2_T);
  relu(z);
  cv::gemm(z, w3, 1, b3, 1, z, cv::GEMM_2_T);
  relu(z);
  return z;
}


// TODO: replace .at<> access
// https://docs.opencv.org/3.1.0/db/da5/tutorial_how_to_scan_images.html

std::tuple<float, cv::Mat> processImage(cv::Mat image)
{
  float turn_percent;
  cv::Mat centroids = cv::Mat::zeros(REGIONS.size(), 2, CV_16U);
  cv::Mat first_col = centroids.rowRange(0, REGIONS.size()).colRange(0, 1);
  std::vector<cv::Mat> matrices;
  cv::Mat batch;

  int i = 0;
  for (std::vector<cv::Rect>::iterator it = REGIONS.begin() ; it != REGIONS.end(); ++it)
  {
    cv::Mat roi(image, *it);
    cv::Mat out = preprocessImage(roi);
    matrices.push_back(out);
    // Add left margin
    centroids.at<uint16_t>(i, 0) = it->x;
    // Add top margin + set y_center to the middle heght
    centroids.at<uint16_t>(i, 1) = int(it->height / 2) + it->y;
    i++;
  }
  cv::vconcat(matrices, batch);
  cv::Mat pred = forward(batch) * im_width;

  pred.convertTo(pred, CV_16U);
  first_col += pred;

  cv::Mat x = first_col;
  cv::Mat y = centroids.rowRange(0, REGIONS.size()).colRange(1, 2);
  // Case x = cst, m = 0
  bool x_constant = true;
  for (size_t i = 0; i < REGIONS.size() - 1; i++)
  {
    if (centroids.at<uint16_t>(i, 0) != centroids.at<uint16_t>(i + 1, 0))
    {
      x_constant = false;
      break;
    }
  }

  cv::Mat coeff_mat;
  if (x_constant)
  {
    turn_percent = 0.0;
  }
  else
  {
    cv::Mat ones = cv::Mat::ones(REGIONS.size(), 1, CV_16U);
    // cv::Mat ones_1 = cv::Mat::ones(REGIONS.size(), 1, CV_16U);
    // ones.at<uint16_t>(1, 0) = 2;
    // cv::Mat two = 2 * cv::Mat::ones(REGIONS.size(), 1, CV_32F);
    // two.at<float>(1, 0) = 4;
    // A = np.vstack([y, np.ones(len(y))]).T
    // m, b = np.linalg.lstsq(A, x)[0]
    cv::Mat A;
    cv::hconcat(y, ones, A);
    A.convertTo(A, CV_32F);
    x.convertTo(x, CV_32F);

    cv::solve(A, x, coeff_mat, cv::DECOMP_SVD);
    float m = coeff_mat.at<float>(0, 0);
    float track_angle = atan(1 / m);
    float diff_angle = std::abs(REF_ANGLE) - std::abs(track_angle);
    // Estimation of the line curvature
    turn_percent = (diff_angle / MAX_ANGLE) * 100.0;
  }

  return std::make_tuple(turn_percent, centroids);
}


PYBIND11_MODULE(test_module, m) {
  NDArrayConverter::init_numpy();
  m.def("forward", forward, "");
  // m.def("init", init, "");
  m.def("processImage", processImage, "");
  m.def("setWeights", setWeights, "");
}
