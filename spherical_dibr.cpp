#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <omp.h>

#define RAD(x) M_PI*(x)/180.0
#define DEGREE(x) 180.0*(x)/M_PI

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

void draw_progress(float progress)
{
    std::cout << "[";
    int bar_width = 70;
    int pos = bar_width * progress;
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

// From OpenCV example utils.hpp code
// Calculates rotation matrix given euler angles.
Mat eular2rot(Vec3f theta)
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
    Mat R = R_z * R_y * R_x;
     
    return R;
}

Vec3d pixel2rad(const Vec3d& in_vec, int width, int height)
{
    return Vec3d(M_PI*in_vec[0]/height, 2*M_PI*in_vec[1]/width, in_vec[2]);
}

Vec3d rad2cart(const Vec3d& vec_rad)
{
    Vec3d vec_cartesian;
    vec_cartesian[0] = vec_rad[2]*sin(vec_rad[0])*cos(vec_rad[1]);
    vec_cartesian[1] = vec_rad[2]*sin(vec_rad[0])*sin(vec_rad[1]);
    vec_cartesian[2] = vec_rad[2]*cos(vec_rad[0]);
    return vec_cartesian;
}

Vec3d applyRT(const Vec3d& vec_cartesian, const Mat& rot_mat, const Vec3d t_vec)
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

Vec3d cart2rad(const Vec3d& vec_cartesian_rot)
{
    Vec3d vec_rot;
    vec_rot[2] = sqrt(vec_cartesian_rot[0]*vec_cartesian_rot[0] + vec_cartesian_rot[1]*vec_cartesian_rot[1] + vec_cartesian_rot[2]*vec_cartesian_rot[2]);

    vec_rot[0] = acos(vec_cartesian_rot[2]/vec_rot[2]);
    vec_rot[1] = atan2(vec_cartesian_rot[1], vec_cartesian_rot[0]);
    if(vec_rot[1] < 0)
        vec_rot[1] += M_PI*2;

    return vec_rot;
}

Vec3d rad2pixel(const Vec3d& vec_rot, int width, int height)
{
    Vec3d vec_pixel;
    vec_pixel[0] = height*vec_rot[0]/M_PI;
    vec_pixel[1] = width*vec_rot[1]/(2*M_PI);
    vec_pixel[2] = vec_rot[2];
    return vec_pixel;
}

// rotate pixel, in_vec as input(row, col)
Vec3d rt_pixel(const Vec3d& in_vec, const Vec3d& t_vec, const Mat& rot_mat, int width, int height)
{
    Vec3d vec_rad = pixel2rad(in_vec, width, height);
    Vec3d vec_cartesian = rad2cart(vec_rad);
    Vec3d vec_cartesian_rot = applyRT(vec_cartesian, rot_mat, t_vec);
    Vec3d vec_rot = cart2rad(vec_cartesian_rot);
    Vec3d vec_pixel = rad2pixel(vec_rot, width, height);

    return vec_pixel;
}

Mat map_distance(Mat& depth, double min_pixel, double max_pixel, double min_dist, double max_dist)
{
    Mat depth_double(depth.rows, depth.cols, CV_64FC1);
    unsigned short* depth_data = (unsigned short*)depth.data;
    double* depth_double_data = (double*)depth_double.data;
    #pragma omp parallel for
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

int main(int argc, char** argv)
{
    if(argc != 9)
    {
        cout << "Usage : Equirectangular_rotate.out <Image file name> <Depth file name> <roll> <pitch> <yaw> <Tx> <Ty> <Tz>" << endl;
        cout << "<roll>, <pitch>, <yaw> is rotation angle, It should be 0~360" << endl;
        cout << "<Tx>, <Ty>, <Tz> is translation in meter" << endl;
        return 0;
    }
    Mat im = imread(argv[1], IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
    Mat depth = imread(argv[2], IMREAD_ANYDEPTH);
    if(im.data == NULL || depth.data == NULL)
    {
        cout << "Can't open image" << endl;
        return 0;
    }

    double im_width = im.cols;
    double im_height = im.rows;
    double im_size = im_width*im_height;
    Size im_shape(im_height, im_width);

    double min_pixel = 0, max_pixel = 65535;
    double min_dist = 0, max_dist = 6;
    Mat depth_double = map_distance(depth, min_pixel, max_pixel, min_dist, max_dist);

    cout << "width : " << im_width << ", height : " << im_height << endl;

    Mat2i im_pixel_rotate(im_height, im_width);
    Mat im_out(im.rows, im.cols, im.type());
    Mat depth_out_double(depth.rows, depth.cols, CV_64FC1);
    Mat depth_out(depth.rows, depth.cols, depth.type());

    Vec3w* im_data = (Vec3w*)im.data;
    Vec3w* im_out_data = (Vec3w*)im_out.data;
    double* depth_data = (double*)depth_double.data;
    double* depth_out_double_data = (double*)depth_out_double.data;
    unsigned short* depth_out_data = (unsigned short*)depth_out.data;

    Mat rot_mat = eular2rot(Vec3f(RAD(atof(argv[3])), RAD(atof(argv[4])), RAD(atof(argv[5]))));
    Vec3d t_vec(atof(argv[6]), atof(argv[7]), atof(argv[8]));

    #pragma omp parallel for
    for(int i = 0; i < static_cast<int>(im_height); i++)
    {
        for(int j = 0; j < static_cast<int>(im_width); j++)
        {
            // inverse warping
            Vec3d vec_pixel = rt_pixel(Vec3d(i, j, depth_data[i*im.cols + j])
                                       , t_vec
                                       , rot_mat
                                       , im_width, im_height);
            int dist_i = vec_pixel[0];
            int dist_j = vec_pixel[1];
            double dist_depth = vec_pixel[2];
            if((dist_i >= 0) && (dist_j >= 0) && (dist_i < im_height) && (dist_j < im_width))
            {
                im_out_data[dist_i*im.cols + dist_j] = im_data[i*im.cols + j];
                depth_out_double_data[dist_i*im.cols + dist_j] = (dist_depth - min_dist)/(max_dist - min_dist);
                depth_out_double_data[dist_i*im.cols + dist_j] = depth_out_double_data[dist_i*im.cols + dist_j]*(max_pixel - min_pixel) + min_pixel;
                depth_out_data[dist_i*im.cols + dist_j] = depth_out_double_data[dist_i*im.cols + dist_j];
            }
        }
        if(omp_get_thread_num() == 0)
            draw_progress((i*1.0f/(im_height/omp_get_num_threads())));
    }

    vector<int> param;
    param.push_back(CV_IMWRITE_PNG_COMPRESSION);
    param.push_back(0);

    String savename = argv[1];
    savename = "_" + savename;
    savename = argv[8] + savename;
    savename = "_" + savename;
    savename = argv[7] + savename;
    savename = "_" + savename;
    savename = argv[6] + savename;
    savename = "_" + savename;
    savename = argv[5] + savename;
    savename = "_" + savename;
    savename = argv[4] + savename;
    savename = "_" + savename;
    savename = argv[3] + savename;
    savename = "rt_" + savename;
    cout << "Save to " << savename << endl;
    imwrite(savename, im_out, param);

    String depthname = argv[1];
    depthname = "_" + depthname;
    depthname = argv[8] + depthname;
    depthname = "_" + depthname;
    depthname = argv[7] + depthname;
    depthname = "_" + depthname;
    depthname = argv[6] + depthname;
    depthname = "_" + depthname;
    depthname = argv[5] + depthname;
    depthname = "_" + depthname;
    depthname = argv[4] + depthname;
    depthname = "_" + depthname;
    depthname = argv[3] + depthname;
    depthname = "depth_" + depthname;
    cout << "Save to " << depthname << endl;
    imwrite(depthname, depth_out, param);

    return 0;
}