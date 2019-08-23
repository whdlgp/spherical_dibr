#include "spherical_dibr.hpp"

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

Vec3d spherical_dibr::applyRT(const Vec3d& vec_cartesian, const Mat& rot_mat, const Vec3d t_vec)
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

// rotate pixel, in_vec as input(row, col)
Vec3d spherical_dibr::rt_pixel(const Vec3d& in_vec, const Vec3d& t_vec, const Mat& rot_mat, int width, int height)
{
    Vec3d vec_rad = pixel2rad(in_vec, width, height);
    Vec3d vec_cartesian = rad2cart(vec_rad);
    Vec3d vec_cartesian_rot = applyRT(vec_cartesian, rot_mat, t_vec);
    Vec3d vec_rot = cart2rad(vec_cartesian_rot);
    Vec3d vec_pixel = rad2pixel(vec_rot, width, height);

    return vec_pixel;
}

Mat spherical_dibr::map_distance(Mat& depth, double min_pixel, double max_pixel, double min_dist, double max_dist)
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

Mat spherical_dibr::remap_distance(Mat& depth, double min_dist, double max_dist, double min_pixel, double max_pixel)
{
    Mat depth_int16(depth.rows, depth.cols, CV_16UC1);
    double* depth_data = (double*)depth.data;
    unsigned short* depth_int16_data = (unsigned short*)depth_int16.data;
    #pragma omp parallel for
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

void spherical_dibr::image_depth_forward_mapping(Mat& im, Mat& depth_double, Mat& rot_mat, Vec3d t_vec, Mat& im_out, Mat& depth_out_double)
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
    
    #pragma omp parallel for
    for(int i = 0; i < im_height; i++)
    {
        for(int j = 0; j < im_width; j++)
        {
            // forward warping
            Vec3d vec_pixel = rt_pixel(Vec3d(i, j, depth_data[i*im.cols + j])
                                       , t_vec
                                       , rot_mat
                                       , im_width, im_height);
            int dist_i = vec_pixel[0];
            int dist_j = vec_pixel[1];

            srci_data[dist_i*im.cols + dist_j] = i;
            srcj_data[dist_i*im.cols + dist_j] = j;
            double dist_depth = vec_pixel[2];
            if((dist_i >= 0) && (dist_j >= 0) && (dist_i < im_height) && (dist_j < im_width))
                depth_out_double_data[dist_i*im.cols + dist_j] = dist_depth;
        }
    }
    remap(im, im_out, srcj, srci, cv::INTER_LINEAR);
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
    
    #pragma omp parallel for
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

    #pragma omp parallel for
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

    #pragma omp parallel for
    for(int i = 0; i < im_height; i++)
    {
        for(int j = 0; j < im_width; j++)
        {
            depth_data[i*im_width + j] = (depth_double_data[i*im_width + j] - min_dist)*(max_pixel - min_pixel)/(max_dist - min_dist) + min_pixel;
        }
    }

    return depth;
}

void spherical_dibr::save_log_image(string log_dir, Mat& im_out_forward, Mat& im_out_inverse_median, Mat& im_out_inverse_closing, Mat& depth_out_forward, Mat& depth_out_median, Mat& depth_out_closing)
{
    string im_out_forward_name = log_dir + "im_out_forward.png";
    string im_out_inverse_median_name = log_dir + "im_out_inverse_median.png";
    string im_out_inverse_closing_name = log_dir + "im_out_inverse_closing.png";
    string depth_out_forward_name = log_dir + "depth_out_forward.png";
    string depth_out_median_name = log_dir + "depth_out_median.png";
    string depth_out_closing_name = log_dir + "depth_out_closing.png";

    vector<int> param;
    param.push_back(CV_IMWRITE_PNG_COMPRESSION);
    param.push_back(0);

    imwrite(im_out_forward_name, im_out_forward, param);
    imwrite(im_out_inverse_median_name, im_out_inverse_median, param);
    imwrite(im_out_inverse_closing_name, im_out_inverse_closing, param);
    imwrite(depth_out_forward_name, depth_out_forward, param);
    imwrite(depth_out_median_name, depth_out_median, param);
    imwrite(depth_out_closing_name, depth_out_closing, param);
}

void spherical_dibr::render(cv::Mat& im, cv::Mat& depth_double
            , cv::Mat& rot_mat, cv::Vec3d t_vec)
{
    // Do forward mapping
    Mat im_out;
    Mat depth_out_double;
    image_depth_forward_mapping(im, depth_double, rot_mat, t_vec, im_out, depth_out_double);

    // Filtering depth with median/morphological closing
    int element_size = 7;
    Mat depth_out_double_median = median_depth(depth_out_double, element_size);
    Mat depth_out_double_closing = closing_depth(depth_out_double, element_size);
    
    // Do inverse mapping
    Mat rot_mat_inv = rot_mat.t();
    Vec3d t_vec_inv = -t_vec;
    
    Mat im_out_inv_median, im_out_inv_closing;
    image_depth_inverse_mapping(im, depth_out_double_median, rot_mat_inv, t_vec_inv, im_out_inv_median);
    image_depth_inverse_mapping(im, depth_out_double_closing, rot_mat_inv, t_vec_inv, im_out_inv_closing);

    im_out_forward = im_out;
    im_out_inverse_median = im_out_inv_median;
    im_out_inverse_closing = im_out_inv_closing;
    depth_out_forward = depth_out_double;
    depth_out_median = depth_out_double_median;
    depth_out_closing = depth_out_double_closing;
}

int spherical_dibr::test(int argc, char** argv)
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

    // Position for virtual view point
    Mat rot_mat = eular2rot(Vec3f(RAD(atof(argv[3])), RAD(atof(argv[4])), RAD(atof(argv[5]))));
    Vec3d t_vec(atof(argv[6]), atof(argv[7]), atof(argv[8]));

    // Depth image range
    double min_pixel = 0, max_pixel = 65535;
    double min_dist = 0, max_dist = 6;
    Mat depth_double = map_distance(depth, min_pixel, max_pixel, min_dist, max_dist);

    double im_width = im.cols;
    double im_height = im.rows;
    cout << "width : " << im_width << ", height : " << im_height << endl;

    // Do Depth Image Based Rendering
    render(im, depth_double
            , rot_mat, t_vec);

    // Save log images
    string log_dir = "log/";
    save_log_image(log_dir, im_out_forward
                          , im_out_inverse_median
                          , im_out_inverse_closing
                          , depth_out_forward
                          , depth_out_median
                          , depth_out_closing);

    return 0;
}
