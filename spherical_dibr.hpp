#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <omp.h>
#include <sstream>

#include <opencv2/opencv.hpp>

class spherical_dibr
{
public:
    cv::Mat map_distance(cv::Mat& depth, double min_pixel, double max_pixel, double min_dist, double max_dist);
    cv::Mat remap_distance(cv::Mat& depth, double min_dist, double max_dist, double min_pixel, double max_pixel);
    cv::Vec3d rot2eular(cv::Mat rot_mat);
    cv::Mat eular2rot(cv::Vec3d theta);
    void render(cv::Mat& im, cv::Mat& depth_double
            , cv::Mat& rot_mat, cv::Vec3d t_vec);

    cv::Mat im_out_forward;
    cv::Mat im_out_inverse_median;
    cv::Mat im_out_inverse_closing;
    cv::Mat depth_out_forward;
    cv::Mat depth_out_median;
    cv::Mat depth_out_closing;

    int test(int argc, char** argv);
private:
    cv::Vec3d pixel2rad(const cv::Vec3d& in_vec, int width, int height);
    cv::Vec3d rad2cart(const cv::Vec3d& vec_rad);
    cv::Vec3d applyTR(const cv::Vec3d& vec_cartesian, const cv::Mat& rot_mat, const cv::Vec3d t_vec);
    cv::Vec3d applyRT(const cv::Vec3d& vec_cartesian, const cv::Mat& rot_mat, const cv::Vec3d t_vec);
    cv::Vec3d cart2rad(const cv::Vec3d& vec_cartesian_rot);
    cv::Vec3d rad2pixel(const cv::Vec3d& vec_rot, int width, int height);
    cv::Vec3d tr_pixel(const cv::Vec3d& in_vec, const cv::Vec3d& t_vec, const cv::Mat& rot_mat, int width, int height);
    cv::Vec3d rt_pixel(const cv::Vec3d& in_vec, const cv::Vec3d& t_vec, const cv::Mat& rot_mat, int width, int height);
    cv::Mat median_depth(cv::Mat& depth_double, int size);
    cv::Mat closing_depth(cv::Mat& depth_double, int size);
    void image_depth_forward_mapping(cv::Mat& im, cv::Mat& depth_double, cv::Mat& rot_mat, cv::Vec3d t_vec, cv::Mat& im_out, cv::Mat& depth_out_double);
    void image_depth_inverse_mapping(cv::Mat& im, cv::Mat& depth_out_double, cv::Mat& rot_mat_inv, cv::Vec3d t_vec_inv, cv::Mat& im_out);
    cv::Mat show_double_depth(cv::Mat& depth_double);
    cv::Mat show_float_depth(cv::Mat& depth_double);
    void save_log_image(std::string log_dir, cv::Mat& im_out_forward, cv::Mat& im_out_inverse_median, cv::Mat& im_out_inverse_closing, cv::Mat& depth_out_forward, cv::Mat& depth_out_median, cv::Mat& depth_out_closing);
};