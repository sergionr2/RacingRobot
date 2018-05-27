#include <string>
#include <iostream>
#include <vector>
#include <math.h> // atan
#include <cmath> // std::abs

#include <pybind11/pybind11.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define CONVERT_TO_INT true
#define MAX_WIDTH 320
#define PI 3.14159265
#define REF_ANGLE float(- PI / 2)
#define MAX_ANGLE float(PI / 4)
#define IM_WIDTH MAX_WIDTH
#define N_REGIONS 3

#include "ndarray_converter.h"

using namespace Eigen;
using EigenMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace py = pybind11;

bool initialized = false;


cv::Mat centroids;
cv::Mat resized = cv::Mat::ones(20, 80, CV_32FC3);

cv::Mat w1 = cv::Mat::ones(80*20*3, 20, CV_32F).t();
cv::Mat b1 = cv::Mat::ones(20, 1, CV_32F).t();

cv::Mat w2 = cv::Mat::ones(20, 4, CV_32F).t();
cv::Mat b2 = cv::Mat::ones(4, 1, CV_32F).t();

cv::Mat w3 = cv::Mat::ones(4, 1, CV_32F).t();
cv::Mat b3 = cv::Mat::ones(1, 1, CV_32F).t();

// Regions of interest (ROIs)
cv::Rect R0(0, 100, MAX_WIDTH, 100);
cv::Rect R1(0, 75, MAX_WIDTH, 100);
cv::Rect R2(0, 50, MAX_WIDTH, 100);
cv::Rect R3(0, 75, MAX_WIDTH, 50);
cv::Rect R4(0, 50, MAX_WIDTH, 50);
// ROI used
std::vector<cv::Rect> REGIONS({R0, R1, R2});

/**
 * Set the weights of the neural network
 * it should be called from python
 */
void setWeights(cv::Mat w1_, cv::Mat b1_, cv::Mat w2_, cv::Mat b2_, cv::Mat w3_, cv::Mat b3_)
{
  w1 = w1_.clone().t();
  w2 = w2_.clone().t();
  w3 = w3_.clone().t();
  b1 = b1_.clone().t();
  b2 = b2_.clone().t();
  b3 = b3_.clone().t();
  // Concatenate bias to avoid broadcast issue
  // (addition between a matrix and a vector)
  // TODO: replace vector creation with
  //  a loop that depends on the number of regions
  CV_Assert(REGIONS.size() == N_REGIONS);
  std::vector<cv::Mat> m1 {b1, b1, b1};
  cv::vconcat(m1, b1);
  std::vector<cv::Mat> m2 {b2, b2, b2};
  cv::vconcat(m2, b2);
  std::vector<cv::Mat> m3 {b3, b3, b3};
  cv::vconcat(m3, b3);
  initialized = true;
}

/**
 * Reshape and normalize an input image
 * @param  image CV_32FC3
 */
cv::Mat preprocessImage(cv::Mat image)
{
  // WARNING: resizing is a bottleneck in the computation
  // Best image quality: cv::INTER_AREA
  // Fastest:  cv::INTER_NEAREST
  // Between the two: cv::INTER_LINEAR (bilinear interpolation)
  cv::resize(image, image, resized.size(), 0, 0, cv::INTER_LINEAR);
  image = image.reshape(1, 1);
  image /= 255.0;
  image -= 0.5;
  image *= 2;
  return image;
}

/**
 * Rectifier activation
 */
float relu(float x)
{
    return std::max(float(0.0), x);
}

/**
 * Forward pass of the neural network
 * @param  x CV_32F
 * @return   CV_32F
 */
cv::Mat forward(cv::Mat x)
{
  CV_Assert(x.type() == CV_32F);

  // Convert to eigen matrices
  Map<EigenMat> w1_eigen(w1.ptr<float>(), w1.rows, w1.cols);
  Map<EigenMat> b1_eigen(b1.ptr<float>(), b1.rows, b1.cols);
  Map<EigenMat> w2_eigen(w2.ptr<float>(), w2.rows, w2.cols);
  Map<EigenMat> b2_eigen(b2.ptr<float>(), b2.rows, b2.cols);
  Map<EigenMat> w3_eigen(w3.ptr<float>(), w3.rows, w3.cols);
  Map<EigenMat> b3_eigen(b3.ptr<float>(), b3.rows, b3.cols);
  Map<EigenMat> x_eigen(x.ptr<float>(), x.rows, x.cols);

  // 1st layer
  EigenMat z1 = x_eigen * w1_eigen.transpose() + b1_eigen;
  // Apply ReLU activation elementwise
  z1 = z1.unaryExpr(&relu);
  // 2nd layer
  EigenMat z2 = z1 * w2_eigen.transpose() + b2_eigen;
  z2 = z2.unaryExpr(&relu);
  // Output layer
  EigenMat z3 = z2 * w3_eigen.transpose() + b3_eigen;
  z3 = z3.unaryExpr(&relu);

  // Convert back to OpenCV Matrix
  cv::Mat z3_mat;
  cv::eigen2cv(z3, z3_mat);
  return z3_mat;
}

std::tuple<float, cv::Mat> processImage(cv::Mat image)
{
  if (!initialized)
  {
    std::cout << "WARNING: Neural Network is not initialized!" << '\n';
    std::cout << "Please use setWeights() function" << '\n';
  }
  CV_Assert(image.type() == CV_32FC3);

  float turn_percent;
  std::vector<cv::Mat> matrices;
  cv::Mat batch;
  cv::Mat centroids = cv::Mat::zeros(REGIONS.size(), 2, CV_32F);
  cv::Mat first_col = centroids.rowRange(0, REGIONS.size()).colRange(0, 1);

  int i = 0;
  for (std::vector<cv::Rect>::iterator it = REGIONS.begin() ; it != REGIONS.end(); ++it)
  {
    // Extract each region of interest
    cv::Mat roi(image, *it);
    // Preprocess the image: scaling and normalization
    cv::Mat out = preprocessImage(roi);
    matrices.push_back(out);
    // Add left margin
    centroids.at<float>(i, 0) = it->x;
    // Add top margin + set y_center to the middle height
    centroids.at<float>(i, 1) = int(it->height / 2) + it->y;
    i++;
  }
  // Create a batch
  cv::vconcat(matrices, batch);
  // Predict where is the center of the line using the trained network
  // and scale the output
  first_col += forward(batch) * IM_WIDTH;

  // Linear Regression to fit a line | x = m*y + b
  // It estimates the line curve
  cv::Mat x = first_col;
  cv::Mat y = centroids.rowRange(0, REGIONS.size()).colRange(1, 2);
  // Case x = cst, m = 0
  bool x_constant = true;
  for (size_t i = 0; i < REGIONS.size() - 1; i++)
  {
    if (centroids.at<float>(i, 0) != centroids.at<float>(i + 1, 0))
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
    // Linear regression using least squares method
    // x = m*y + b -> y = 1/m * x - b/m if m != 0
    cv::Mat ones = cv::Mat::ones(REGIONS.size(), 1, CV_32F);
    cv::Mat A;
    cv::hconcat(y, ones, A);
    cv::solve(A, x, coeff_mat, cv::DECOMP_SVD);
    // Compute the angle between the reference and the fitted line
    float m = coeff_mat.at<float>(0, 0);
    float track_angle = atan(1 / m);
    float diff_angle = std::abs(REF_ANGLE) - std::abs(track_angle);
    // Estimation of the line curvature
    turn_percent = (diff_angle / MAX_ANGLE) * 100.0;
  }

  if (CONVERT_TO_INT)
  {
    centroids.convertTo(centroids, CV_16U);
  }

  return std::make_tuple(turn_percent, centroids);
}


PYBIND11_MODULE(fast_image_processing, m) {
  NDArrayConverter::init_numpy();
  m.def("forward", forward, "");
  m.def("processImage", processImage, "");
  m.def("setWeights", setWeights, "");
}
