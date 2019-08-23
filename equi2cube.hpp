#pragma once
#define _USE_MATH_DEFINES
#include <vector>
#include <algorithm>
#include <utility>
#include <set>
#include <functional>
#include <sstream>
#include <iostream>
#include <cmath>

#include "opencv2/opencv.hpp"

#include <omp.h>

class equi2cube
{
    public:
    void set_omp(int num_proc);
    cv::Mat get_back(const cv::Mat& im, int cube_size);
    cv::Mat get_front(const cv::Mat& im, int cube_size);
    cv::Mat get_left(const cv::Mat& im, int cube_size);
    cv::Mat get_right(const cv::Mat& im, int cube_size);
    cv::Mat get_top(const cv::Mat& im, int cube_size);
    cv::Mat get_bottom(const cv::Mat& im, int cube_size);
    cv::Mat get_all(const cv::Mat& im, int cube_size);
    cv::Mat cube2equi(const cv::Mat& im, int out_width, int out_height);
    private:
};