#include "spherical_dibr.hpp"
#include "equi2cube.hpp"

#define RAD(x) M_PI*(x)/180.0
#define DEGREE(x) 180.0*(x)/M_PI


using namespace std;
using namespace cv;

// XYZ-eular rotation 
Mat spherical_dibr::eular2rot(Vec3d theta)
{
    // Calculate rotation about x axis
    Mat R_x = (Mat_<double>(3,3) <<
               1,       0,              0,
               0,       cos(theta[0]),   -sin(theta[0]),
               0,       sin(theta[0]),   cos(theta[0])
               );
     
    // Calculate rotation about y axis
    Mat R_y = (Mat_<double>(3,3) <<
               cos(theta[1]),    0,      sin(theta[1]),
               0,               1,      0,
               -sin(theta[1]),   0,      cos(theta[1])
               );
     
    // Calculate rotation about z axis
    Mat R_z = (Mat_<double>(3,3) <<
               cos(theta[2]),    -sin(theta[2]),      0,
               sin(theta[2]),    cos(theta[2]),       0,
               0,               0,                  1);
     
    // Combined rotation matrix
    Mat R = R_x * R_y * R_z;
     
    return R;
}

// Rotation matrix to rotation vector in XYZ-eular order
Vec3d spherical_dibr::rot2eular(Mat R)
{
    double sy = sqrt(R.at<double>(2,2) * R.at<double>(2,2) +  R.at<double>(1,2) * R.at<double>(1,2) );
 
    bool singular = sy < 1e-6; // If
 
    double x, y, z;
    if (!singular)
    {
        x = atan2(-R.at<double>(1,2) , R.at<double>(2,2));
        y = atan2(R.at<double>(0,2), sy);
        z = atan2(-R.at<double>(0,1), R.at<double>(0,0));
    }
    else
    {
        x = 0;
        y = atan2(R.at<double>(0,2), sy);
        z = atan2(-R.at<double>(0,1), R.at<double>(0,0));
    }
    return Vec3d(x, y, z);
}

Vec3d spherical_dibr::pixel2rad(const Vec3d& in_vec, int width, int height)
{
    return Vec3d(M_PI*in_vec[0]/height, 2*M_PI*in_vec[1]/width, in_vec[2]);
}

Vec3d spherical_dibr::rad2cart(const Vec3d& vec_rad)
{
    Vec3d vec_cartesian;
    vec_cartesian[0] = vec_rad[2]*sin(vec_rad[0])*cos(vec_rad[1]);
    vec_cartesian[1] = vec_rad[2]*sin(vec_rad[0])*sin(vec_rad[1]);
    vec_cartesian[2] = vec_rad[2]*cos(vec_rad[0]);
    return vec_cartesian;
}

Vec3d spherical_dibr::applyTR(const Vec3d& vec_cartesian, const Mat& rot_mat, const Vec3d t_vec)
{
    double* rot_mat_data = (double*)rot_mat.data;
    Vec3d vec_cartesian_tran;
    vec_cartesian_tran[0] = vec_cartesian[0] - t_vec[0];
    vec_cartesian_tran[1] = vec_cartesian[1] - t_vec[1];
    vec_cartesian_tran[2] = vec_cartesian[2] - t_vec[2];

    Vec3d vec_cartesian_rot;
    vec_cartesian_rot[0] = rot_mat_data[0]*vec_cartesian_tran[0] + rot_mat_data[1]*vec_cartesian_tran[1] + rot_mat_data[2]*vec_cartesian_tran[2];
    vec_cartesian_rot[1] = rot_mat_data[3]*vec_cartesian_tran[0] + rot_mat_data[4]*vec_cartesian_tran[1] + rot_mat_data[5]*vec_cartesian_tran[2];
    vec_cartesian_rot[2] = rot_mat_data[6]*vec_cartesian_tran[0] + rot_mat_data[7]*vec_cartesian_tran[1] + rot_mat_data[8]*vec_cartesian_tran[2];

    return vec_cartesian_rot;
}

Vec3d spherical_dibr::applyRT(const Vec3d& vec_cartesian, const Mat& rot_mat, const Vec3d t_vec)
{
    double* rot_mat_data = (double*)rot_mat.data;
    Vec3d vec_cartesian_rot;
    vec_cartesian_rot[0] = rot_mat_data[0]*vec_cartesian[0] + rot_mat_data[1]*vec_cartesian[1] + rot_mat_data[2]*vec_cartesian[2];
    vec_cartesian_rot[1] = rot_mat_data[3]*vec_cartesian[0] + rot_mat_data[4]*vec_cartesian[1] + rot_mat_data[5]*vec_cartesian[2];
    vec_cartesian_rot[2] = rot_mat_data[6]*vec_cartesian[0] + rot_mat_data[7]*vec_cartesian[1] + rot_mat_data[8]*vec_cartesian[2];

    Vec3d vec_cartesian_tran;
    vec_cartesian_tran[0] = vec_cartesian_rot[0] - t_vec[0];
    vec_cartesian_tran[1] = vec_cartesian_rot[1] - t_vec[1];
    vec_cartesian_tran[2] = vec_cartesian_rot[2] - t_vec[2];

    return vec_cartesian_tran;
}

Vec3d spherical_dibr::cart2rad(const Vec3d& vec_cartesian_rot)
{
    Vec3d vec_rot;
    vec_rot[2] = sqrt(vec_cartesian_rot[0]*vec_cartesian_rot[0] + vec_cartesian_rot[1]*vec_cartesian_rot[1] + vec_cartesian_rot[2]*vec_cartesian_rot[2]);

    vec_rot[0] = acos(vec_cartesian_rot[2]/vec_rot[2]);
    vec_rot[1] = atan2(vec_cartesian_rot[1], vec_cartesian_rot[0]);
    if(vec_rot[1] < 0)
        vec_rot[1] += M_PI*2;

    return vec_rot;
}

Vec3d spherical_dibr::rad2pixel(const Vec3d& vec_rot, int width, int height)
{
    Vec3d vec_pixel;
    vec_pixel[0] = height*vec_rot[0]/M_PI;
    vec_pixel[1] = width*vec_rot[1]/(2*M_PI);
    vec_pixel[2] = vec_rot[2];
    return vec_pixel;
}

// rotate and translate pixel, in_vec as input(row, col)
Vec3d spherical_dibr::rt_pixel(const Vec3d& in_vec, const Vec3d& t_vec, const Mat& rot_mat, int width, int height)
{
    Vec3d vec_rad = pixel2rad(in_vec, width, height);
    Vec3d vec_cartesian = rad2cart(vec_rad);
    Vec3d vec_cartesian_rot = applyRT(vec_cartesian, rot_mat, t_vec);
    Vec3d vec_rot = cart2rad(vec_cartesian_rot);
    Vec3d vec_pixel = rad2pixel(vec_rot, width, height);

    return vec_pixel;
}

// translate and rotate pixel, in_vec as input(row, col)
Vec3d spherical_dibr::tr_pixel(const Vec3d& in_vec, const Vec3d& t_vec, const Mat& rot_mat, int width, int height)
{
    Vec3d vec_rad = pixel2rad(in_vec, width, height);
    Vec3d vec_cartesian = rad2cart(vec_rad);
    Vec3d vec_cartesian_rot = applyTR(vec_cartesian, rot_mat, t_vec);
    Vec3d vec_rot = cart2rad(vec_cartesian_rot);
    Vec3d vec_pixel = rad2pixel(vec_rot, width, height);

    return vec_pixel;
}

Mat spherical_dibr::map_distance(Mat& depth, double min_pixel, double max_pixel, double min_dist, double max_dist)
{
    Mat depth_double(depth.rows, depth.cols, CV_64FC1);
    unsigned short* depth_data = (unsigned short*)depth.data;
    double* depth_double_data = (double*)depth_double.data;
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < depth.rows; i++)
    {
        for(int j = 0; j < depth.cols; j++)
        {
            depth_double_data[i*depth.cols + j] = (depth_data[i*depth.cols + j] - min_pixel)/(max_pixel - min_pixel);
            depth_double_data[i*depth.cols + j] = depth_double_data[i*depth.cols + j]*(max_dist - min_dist) + min_dist;
        }
    }

    return depth_double;
}

Mat spherical_dibr::remap_distance(Mat& depth, double min_dist, double max_dist, double min_pixel, double max_pixel)
{
    Mat depth_int16(depth.rows, depth.cols, CV_16UC1);
    double* depth_data = (double*)depth.data;
    unsigned short* depth_int16_data = (unsigned short*)depth_int16.data;
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < depth.rows; i++)
    {
        for(int j = 0; j < depth.cols; j++)
        {
            depth_int16_data[i*depth.cols + j] = (depth_data[i*depth.cols + j] - min_dist)/(max_dist - min_dist)*(max_pixel - min_pixel) + min_pixel;
        }
    }

    return depth_int16;
}

Mat spherical_dibr::median_depth(Mat& depth_double, int size)
{
    Mat depth_float, depth_double_median, depth_float_median;
    depth_double.convertTo(depth_float, CV_32FC1);
    medianBlur(depth_float, depth_float_median, size);
    depth_float_median.convertTo(depth_double_median, CV_64FC1);
    return depth_double_median;
}

Mat spherical_dibr::closing_depth(Mat& depth_double, int size)
{
    Mat depth_float, depth_double_median, depth_float_median;
    depth_double.convertTo(depth_float, CV_32FC1);
    Mat element(size, size, CV_32FC1, Scalar(1.0));
    morphologyEx(depth_float, depth_float_median, CV_MOP_CLOSE, element);
    depth_float_median.convertTo(depth_double_median, CV_64FC1);
    return depth_double_median;
}

void spherical_dibr::image_depth_forward_mapping(Mat& im, Mat& depth_double, Mat& rot_mat, Vec3d t_vec, Mat& im_out, Mat& depth_out_double, int map_opt)
{
    int im_width = im.cols;
    int im_height = im.rows;

	Mat srci(im_height, im_width, CV_32F);
	Mat srcj(im_height, im_width, CV_32F);
    float* srci_data = (float*)srci.data;
    float* srcj_data = (float*)srcj.data;

    im_out.create(im.rows, im.cols, im.type());
    depth_out_double.create(depth_double.rows, depth_double.cols, depth_double.type());
    Vec3w* im_data = (Vec3w*)im.data;
    Vec3w* im_out_data = (Vec3w*)im_out.data;
    double* depth_data = (double*)depth_double.data;
    double* depth_out_double_data = (double*)depth_out_double.data;
    
    if(map_opt == FORWARD_INVERSE)
    {
        #pragma omp parallel for collapse(2)
        for(int i = 0; i < im_height; i++)
        {
            for(int j = 0; j < im_width; j++)
            {
                // forward warping
                Vec3d vec_pixel = tr_pixel(Vec3d(i, j, depth_data[i*im.cols + j])
                                        , t_vec
                                        , rot_mat
                                        , im_width, im_height);
                int dist_i = vec_pixel[0];
                int dist_j = vec_pixel[1];

                srci_data[dist_i*im.cols + dist_j] = i;
                srcj_data[dist_i*im.cols + dist_j] = j;
                double dist_depth = vec_pixel[2];
                if((dist_i >= 0) && (dist_j >= 0) && (dist_i < im_height) && (dist_j < im_width))
                {
                    if(depth_out_double_data[dist_i*im.cols + dist_j] == 0)
                        depth_out_double_data[dist_i*im.cols + dist_j] = dist_depth;
                    else if(depth_out_double_data[dist_i*im.cols + dist_j] > dist_depth)
                        depth_out_double_data[dist_i*im.cols + dist_j] = dist_depth;
                }
            }
        }
        remap(im, im_out, srcj, srci, cv::INTER_LINEAR);
    }
    else if(map_opt == INVERSE_ONLY)
    {
        #pragma omp parallel for collapse(2)
        for(int i = 0; i < im_height; i++)
        {
            for(int j = 0; j < im_width; j++)
            {
                // forward warping
                Vec3d vec_pixel = tr_pixel(Vec3d(i, j, depth_data[i*im.cols + j])
                                        , t_vec
                                        , rot_mat
                                        , im_width, im_height);
                int dist_i = vec_pixel[0];
                int dist_j = vec_pixel[1];

                double dist_depth = vec_pixel[2];
                if((dist_i >= 0) && (dist_j >= 0) && (dist_i < im_height) && (dist_j < im_width))
                {
                    if(depth_out_double_data[dist_i*im.cols + dist_j] == 0)
                        depth_out_double_data[dist_i*im.cols + dist_j] = dist_depth;
                    else if(depth_out_double_data[dist_i*im.cols + dist_j] > dist_depth)
                        depth_out_double_data[dist_i*im.cols + dist_j] = dist_depth;
                }
            }
        }
    }
}

void spherical_dibr::image_depth_inverse_mapping(Mat& im, Mat& depth_out_double, Mat& rot_mat_inv, Vec3d t_vec_inv, Mat& im_out)
{
    int im_width = im.cols;
    int im_height = im.rows;
	
    Mat srci(im_height, im_width, CV_32F);
	Mat srcj(im_height, im_width, CV_32F);
    float* srci_data = (float*)srci.data;
    float* srcj_data = (float*)srcj.data;

    im_out.create(im.rows, im.cols, im.type());

    Vec3w* im_data = (Vec3w*)im.data;
    Vec3w* im_out_data = (Vec3w*)im_out.data;
    double* depth_out_double_data = (double*)depth_out_double.data;
    
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < im_height; i++)
    {
        for(int j = 0; j < im_width; j++)
        {
            // inverse warping
            Vec3d vec_pixel = rt_pixel(Vec3d(i, j, depth_out_double_data[i*im.cols + j])
                                       , t_vec_inv
                                       , rot_mat_inv
                                       , im_width, im_height);
            int origin_i = vec_pixel[0];
            int origin_j = vec_pixel[1];
            srci_data[i*im.cols + j] = origin_i;
            srcj_data[i*im.cols + j] = origin_j;
            double dist_depth = vec_pixel[2];
        }
    }
    remap(im, im_out, srcj, srci, cv::INTER_LINEAR);
}

Mat spherical_dibr::invert_depth(Mat& depth_double, double min_dist, double max_dist)
{
    Mat depth_inverted(depth_double.rows, depth_double.cols, depth_double.type());
    double* depth_double_data = (double*)depth_double.data;
    double* depth_inverted_data = (double*)depth_inverted.data;
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < depth_double.rows; i++)
    {
        for(int j = 0; j < depth_double.cols; j++)
        {
            if(depth_double_data[i*depth_double.cols + j] > 1e-6)
                depth_inverted_data[i*depth_double.cols + j] = max_dist - depth_double_data[i*depth_double.cols + j];
            else
                depth_inverted_data[i*depth_double.cols + j] = 0;
        }
    }

    return depth_inverted;
}

Mat spherical_dibr::revert_depth(Mat& depth_inverted, double min_dist, double max_dist)
{
    Mat depth_reverted(depth_inverted.rows, depth_inverted.cols, depth_inverted.type());
    double* depth_inverted_data = (double*)depth_inverted.data;
    double* depth_reverted_data = (double*)depth_reverted.data;
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < depth_inverted.rows; i++)
    {
        for(int j = 0; j < depth_inverted.cols; j++)
        {
            if(depth_inverted_data[i*depth_inverted.cols + j] > 1e-6)
                depth_reverted_data[i*depth_inverted.cols + j] = max_dist - depth_inverted_data[i*depth_inverted.cols + j];
            else
                depth_reverted_data[i*depth_inverted.cols + j] = 0;
        }
    }
    return depth_reverted;
}

Mat spherical_dibr::show_double_depth(Mat& depth_double)
{
    double min_pixel = 0, max_pixel = 65535;
    double min_dist, max_dist;
    minMaxLoc(depth_double, &min_dist, &max_dist);

    int im_height = depth_double.rows;
    int im_width = depth_double.cols;
    double* depth_double_data = (double*)depth_double.data;
    Mat depth(im_height, im_width, CV_16UC1);
    unsigned short* depth_data = (unsigned short*)depth.data;

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < im_height; i++)
    {
        for(int j = 0; j < im_width; j++)
        {
            depth_data[i*im_width + j] = (depth_double_data[i*im_width + j] - min_dist)*(max_pixel - min_pixel)/(max_dist - min_dist) + min_pixel;
        }
    }

    return depth;
}

Mat spherical_dibr::show_float_depth(Mat& depth_double)
{
    double min_pixel = 0, max_pixel = 65535;
    double min_dist, max_dist;
    minMaxLoc(depth_double, &min_dist, &max_dist);

    int im_height = depth_double.rows;
    int im_width = depth_double.cols;
    float* depth_double_data = (float*)depth_double.data;
    Mat depth(im_height, im_width, CV_16UC1);
    unsigned short* depth_data = (unsigned short*)depth.data;

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < im_height; i++)
    {
        for(int j = 0; j < im_width; j++)
        {
            depth_data[i*im_width + j] = (depth_double_data[i*im_width + j] - min_dist)*(max_pixel - min_pixel)/(max_dist - min_dist) + min_pixel;
        }
    }

    return depth;
}

void spherical_dibr::render(cv::Mat& im, cv::Mat& depth_double
                            , cv::Mat& rot_mat, cv::Vec3d t_vec
                            , camera_info& cam_info, camera_info& vt_cam_info
                            , int map_opt, int filt_opt)
{
    double min_dist = cam_info.depth_min;
    double max_dist = cam_info.depth_max;

    // Do forward mapping
    image_depth_forward_mapping(im, depth_double
                                , rot_mat, t_vec
                                , im_out_forward, depth_out_forward
                                , map_opt);
    Mat depth_out_forward_inverted = invert_depth(depth_out_forward, min_dist, max_dist);

    // Convert depth map to cube map
    equi2cube eq;
    eq.set_omp(omp_get_num_procs());
    depth_cube = eq.get_all(depth_out_forward_inverted, 600);

    // Filtering depth with median/morphological closing
    int element_size = 7;
    if(filt_opt == FILTER_MEDIAN)
    {
        depth_cube_median = median_depth(depth_cube, element_size);
        // Convert cubemap type depthmap to Equiractangular depthmap
        depth_out_median = eq.cube2equi(depth_cube_median, depth_double.cols, depth_double.rows);
        depth_out_median = invert_depth(depth_out_median, min_dist, max_dist);

        // Do inverse mapping
        Mat rot_mat_inv = rot_mat.t();
        Vec3d t_vec_inv = -t_vec;
        image_depth_inverse_mapping(im, depth_out_median, rot_mat_inv, t_vec_inv, im_out_inverse_median);
    }
    else if(filt_opt == FILTER_CLOSING)
    {
        depth_cube_closing = closing_depth(depth_cube, element_size);
        // Convert cubemap type depthmap to Equiractangular depthmap
        depth_out_closing = eq.cube2equi(depth_cube_closing, depth_double.cols, depth_double.rows);
        depth_out_closing = invert_depth(depth_out_closing, min_dist, max_dist);

        // Do inverse mapping
        Mat rot_mat_inv = rot_mat.t();
        Vec3d t_vec_inv = -t_vec;
        image_depth_inverse_mapping(im, depth_out_closing, rot_mat_inv, t_vec_inv, im_out_inverse_closing);
    }
}
