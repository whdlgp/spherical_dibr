#include "equi2cube.hpp"
using namespace std;
using namespace cv;

void equi2cube::set_omp(int num_proc)
{
    omp_set_num_threads(num_proc);
}

Mat equi2cube::get_back(const Mat& im, int cube_size)
{
    int im_width = im.cols;
    int im_height = im.rows;

    //Back 
    Mat face(cube_size, cube_size, CV_64FC1);
    double* face_data = (double*)face.data;
    double* im_data = (double*)im.data;
    #pragma omp parallel for
    for(int i = 0; i < cube_size; i++)
    {
        for(int j = 0; j < cube_size; j++)
        {
            // pixel coordinate to cartesian coordinate
            Vec3d vec_cart;
            vec_cart[0] = 1.0;
            vec_cart[1] = (2.0*j - cube_size)/cube_size;
            vec_cart[2] = (cube_size - 2.0*i)/cube_size;

            // to unit vector
            Vec3d vec_unit_cart;
            double vec_norm = sqrt(vec_cart[0]*vec_cart[0] + vec_cart[1]*vec_cart[1] + vec_cart[2]*vec_cart[2]);
            vec_unit_cart[0] = vec_cart[0]/vec_norm;
            vec_unit_cart[1] = vec_cart[1]/vec_norm;
            vec_unit_cart[2] = vec_cart[2]/vec_norm;

            // to radian
            Vec2d vec_rad;
            vec_rad[0] = acos(vec_unit_cart[2]);
            vec_rad[1] = atan2(vec_unit_cart[1], vec_unit_cart[0]);
            if(vec_rad[1] < 0)
                vec_rad[1] += M_PI*2;

            Vec2i vec_pixel;
            vec_pixel[0] = im_height*vec_rad[0]/M_PI;
            vec_pixel[1] = im_width*vec_rad[1]/(2*M_PI);
            
            face_data[i*cube_size + j] = im_data[vec_pixel[0]*im_width + vec_pixel[1]];
        }
    }

    return face;
}

Mat equi2cube::get_front(const Mat& im, int cube_size)
{
    int im_width = im.cols;
    int im_height = im.rows;

    //Front 
    Mat face(cube_size, cube_size, CV_64FC1);
    double* face_data = (double*)face.data;
    double* im_data = (double*)im.data;
    #pragma omp parallel for
    for(int i = 0; i < cube_size; i++)
    {
        for(int j = 0; j < cube_size; j++)
        {
            // pixel coordinate to cartesian coordinate
            Vec3d vec_cart;
            vec_cart[0] = -1.0;
            vec_cart[1] = (cube_size - 2.0*j)/cube_size;
            vec_cart[2] = (cube_size - 2.0*i)/cube_size;

            // to unit vector
            Vec3d vec_unit_cart;
            double vec_norm = sqrt(vec_cart[0]*vec_cart[0] + vec_cart[1]*vec_cart[1] + vec_cart[2]*vec_cart[2]);
            vec_unit_cart[0] = vec_cart[0]/vec_norm;
            vec_unit_cart[1] = vec_cart[1]/vec_norm;
            vec_unit_cart[2] = vec_cart[2]/vec_norm;

            // to radian
            Vec2d vec_rad;
            vec_rad[0] = acos(vec_unit_cart[2]);
            vec_rad[1] = atan2(vec_unit_cart[1], vec_unit_cart[0]);
            if(vec_rad[1] < 0)
                vec_rad[1] += M_PI*2;

            Vec2i vec_pixel;
            vec_pixel[0] = im_height*vec_rad[0]/M_PI;
            vec_pixel[1] = im_width*vec_rad[1]/(2*M_PI);
            
            face_data[i*cube_size + j] = im_data[vec_pixel[0]*im_width + vec_pixel[1]];
        }
    }

    return face;
}

Mat equi2cube::get_left(const Mat& im, int cube_size)
{
    int im_width = im.cols;
    int im_height = im.rows;

    //Left 
    Mat face(cube_size, cube_size, CV_64FC1);
    double* face_data = (double*)face.data;
    double* im_data = (double*)im.data;
    #pragma omp parallel for
    for(int i = 0; i < cube_size; i++)
    {
        for(int j = 0; j < cube_size; j++)
        {
            // pixel coordinate to cartesian coordinate
            Vec3d vec_cart;
            vec_cart[0] = (cube_size - 2.0*j)/cube_size;
            vec_cart[1] = 1.0;
            vec_cart[2] = (cube_size - 2.0*i)/cube_size;

            // to unit vector
            Vec3d vec_unit_cart;
            double vec_norm = sqrt(vec_cart[0]*vec_cart[0] + vec_cart[1]*vec_cart[1] + vec_cart[2]*vec_cart[2]);
            vec_unit_cart[0] = vec_cart[0]/vec_norm;
            vec_unit_cart[1] = vec_cart[1]/vec_norm;
            vec_unit_cart[2] = vec_cart[2]/vec_norm;

            // to radian
            Vec2d vec_rad;
            vec_rad[0] = acos(vec_unit_cart[2]);
            vec_rad[1] = atan2(vec_unit_cart[1], vec_unit_cart[0]);
            if(vec_rad[1] < 0)
                vec_rad[1] += M_PI*2;

            Vec2i vec_pixel;
            vec_pixel[0] = im_height*vec_rad[0]/M_PI;
            vec_pixel[1] = im_width*vec_rad[1]/(2*M_PI);
            
            face_data[i*cube_size + j] = im_data[vec_pixel[0]*im_width + vec_pixel[1]];
        }
    }

    return face;
}

Mat equi2cube::get_right(const Mat& im, int cube_size)
{
    int im_width = im.cols;
    int im_height = im.rows;

    //Right 
    Mat face(cube_size, cube_size, CV_64FC1);
    double* face_data = (double*)face.data;
    double* im_data = (double*)im.data;
    #pragma omp parallel for
    for(int i = 0; i < cube_size; i++)
    {
        for(int j = 0; j < cube_size; j++)
        {
            // pixel coordinate to cartesian coordinate
            Vec3d vec_cart;
            vec_cart[0] = (2.0*j - cube_size)/cube_size;
            vec_cart[1] = -1.0;
            vec_cart[2] = (cube_size - 2.0*i)/cube_size;

            // to unit vector
            Vec3d vec_unit_cart;
            double vec_norm = sqrt(vec_cart[0]*vec_cart[0] + vec_cart[1]*vec_cart[1] + vec_cart[2]*vec_cart[2]);
            vec_unit_cart[0] = vec_cart[0]/vec_norm;
            vec_unit_cart[1] = vec_cart[1]/vec_norm;
            vec_unit_cart[2] = vec_cart[2]/vec_norm;

            // to radian
            Vec2d vec_rad;
            vec_rad[0] = acos(vec_unit_cart[2]);
            vec_rad[1] = atan2(vec_unit_cart[1], vec_unit_cart[0]);
            if(vec_rad[1] < 0)
                vec_rad[1] += M_PI*2;

            Vec2i vec_pixel;
            vec_pixel[0] = im_height*vec_rad[0]/M_PI;
            vec_pixel[1] = im_width*vec_rad[1]/(2*M_PI);
            
            face_data[i*cube_size + j] = im_data[vec_pixel[0]*im_width + vec_pixel[1]];
        }
    }

    return face;
}

Mat equi2cube::get_top(const Mat& im, int cube_size)
{
    int im_width = im.cols;
    int im_height = im.rows;

    //Top 
    Mat face(cube_size, cube_size, CV_64FC1);
    double* face_data = (double*)face.data;
    double* im_data = (double*)im.data;
    #pragma omp parallel for
    for(int i = 0; i < cube_size; i++)
    {
        for(int j = 0; j < cube_size; j++)
        {
            // pixel coordinate to cartesian coordinate
            Vec3d vec_cart;
            vec_cart[0] = (cube_size - 2.0*i)/cube_size;
            vec_cart[1] = (cube_size - 2.0*j)/cube_size;
            vec_cart[2] = 1.0;

            // to unit vector
            Vec3d vec_unit_cart;
            double vec_norm = sqrt(vec_cart[0]*vec_cart[0] + vec_cart[1]*vec_cart[1] + vec_cart[2]*vec_cart[2]);
            vec_unit_cart[0] = vec_cart[0]/vec_norm;
            vec_unit_cart[1] = vec_cart[1]/vec_norm;
            vec_unit_cart[2] = vec_cart[2]/vec_norm;

            // to radian
            Vec2d vec_rad;
            vec_rad[0] = acos(vec_unit_cart[2]);
            vec_rad[1] = atan2(vec_unit_cart[1], vec_unit_cart[0]);
            if(vec_rad[1] < 0)
                vec_rad[1] += M_PI*2;

            Vec2i vec_pixel;
            vec_pixel[0] = im_height*vec_rad[0]/M_PI;
            vec_pixel[1] = im_width*vec_rad[1]/(2*M_PI);
            
            face_data[i*cube_size + j] = im_data[vec_pixel[0]*im_width + vec_pixel[1]];
        }
    }

    return face;
}

Mat equi2cube::get_bottom(const Mat& im, int cube_size)
{
    int im_width = im.cols;
    int im_height = im.rows;

    //Bottom 
    Mat face(cube_size, cube_size, CV_64FC1);
    double* face_data = (double*)face.data;
    double* im_data = (double*)im.data;
    #pragma omp parallel for
    for(int i = 0; i < cube_size; i++)
    {
        for(int j = 0; j < cube_size; j++)
        {
            // pixel coordinate to cartesian coordinate
            Vec3d vec_cart;
            vec_cart[0] = (2.0*i - cube_size)/cube_size;
            vec_cart[1] = (cube_size - 2.0*j)/cube_size;
            vec_cart[2] = -1.0;

            // to unit vector
            Vec3d vec_unit_cart;
            double vec_norm = sqrt(vec_cart[0]*vec_cart[0] + vec_cart[1]*vec_cart[1] + vec_cart[2]*vec_cart[2]);
            vec_unit_cart[0] = vec_cart[0]/vec_norm;
            vec_unit_cart[1] = vec_cart[1]/vec_norm;
            vec_unit_cart[2] = vec_cart[2]/vec_norm;

            // to radian
            Vec2d vec_rad;
            vec_rad[0] = acos(vec_unit_cart[2]);
            vec_rad[1] = atan2(vec_unit_cart[1], vec_unit_cart[0]);
            if(vec_rad[1] < 0)
                vec_rad[1] += M_PI*2;

            Vec2i vec_pixel;
            vec_pixel[0] = im_height*vec_rad[0]/M_PI;
            vec_pixel[1] = im_width*vec_rad[1]/(2*M_PI);
            
            face_data[i*cube_size + j] = im_data[vec_pixel[0]*im_width + vec_pixel[1]];
        }
    }

    return face;
}

Mat equi2cube::get_all(const Mat& im, int cube_size)
{
    Mat face_back = get_back(im, cube_size);
    Mat face_front = get_front(im, cube_size);
    Mat face_left = get_left(im, cube_size);
    Mat face_right = get_right(im, cube_size);
    Mat face_top = get_top(im, cube_size);
    Mat face_bottom = get_bottom(im, cube_size);

    Mat cubemap_all;
    vector<Mat> cube_arr;
    cube_arr.push_back(face_left);
    cube_arr.push_back(face_front);
    cube_arr.push_back(face_right);
    cube_arr.push_back(face_back);
    cube_arr.push_back(face_top);
    cube_arr.push_back(face_bottom);
    hconcat(cube_arr, cubemap_all);
    
    return cubemap_all;
}

Mat equi2cube::cube2equi(const Mat& im, int out_width, int out_height)
{
    int cube_size = im.rows;

    Mat out(out_height, out_width, CV_64FC1);
    double* out_data = (double*)out.data;
    double* im_data = (double*)im.data;
    #pragma omp parallel for
    for(int i = 0; i < out_height; i++)
    {
        for(int j = 0; j < out_width; j++)
        {
            Vec2d vec_rad(M_PI*i/out_height, 2*M_PI*j/out_width);
            Vec3d vec_cartesian;
            vec_cartesian[0] = sin(vec_rad[0])*cos(vec_rad[1]);
            vec_cartesian[1] = sin(vec_rad[0])*sin(vec_rad[1]);
            vec_cartesian[2] = cos(vec_rad[0]);

            Vec2i cube_idx;

            Vec3d vec_cartesian_x1, vec_cartesian_y1, vec_cartesian_z1;
            vec_cartesian_x1 = vec_cartesian/abs(vec_cartesian[0]);
            vec_cartesian_y1 = vec_cartesian/abs(vec_cartesian[1]);
            vec_cartesian_z1 = vec_cartesian/abs(vec_cartesian[2]);

            if(vec_cartesian_x1[0] > 0)
            { // check back
                if(abs(vec_cartesian_x1[1]) <= 1 && abs(vec_cartesian_x1[2]) <= 1)
                {
                    int cube_i, cube_j;
                    cube_i = cube_size*(1 - vec_cartesian_x1[2])/2;
                    cube_j = cube_size*(1 + vec_cartesian_x1[1])/2;
                    cube_idx[0] = cube_i;
                    cube_idx[1] = 3*cube_size+cube_j;
                }
            }
            else
            {
                if(abs(vec_cartesian_x1[1]) <= 1 && abs(vec_cartesian_x1[2]) <= 1)
                {
                    int cube_i, cube_j;
                    cube_i = cube_size*(1 - vec_cartesian_x1[2])/2;
                    cube_j = cube_size*(1 - vec_cartesian_x1[1])/2;
                    cube_idx[0] = cube_i;
                    cube_idx[1] = cube_size + cube_j;
                }
            }
            
            if(vec_cartesian_y1[1] > 0)
            { // check left
                if(abs(vec_cartesian_y1[0]) <= 1 && abs(vec_cartesian_y1[2]) <= 1)
                {
                    int cube_i, cube_j;
                    cube_i = cube_size*(1 - vec_cartesian_y1[2])/2;
                    cube_j = cube_size*(1 - vec_cartesian_y1[0])/2;
                    cube_idx[0] = cube_i;
                    cube_idx[1] = cube_j;
                }
            }
            else
            { // check right
                if(abs(vec_cartesian_y1[0]) <= 1 && abs(vec_cartesian_y1[2]) <= 1)
                {
                    int cube_i, cube_j;
                    cube_i = cube_size*(1 - vec_cartesian_y1[2])/2;
                    cube_j = cube_size*(1 + vec_cartesian_y1[0])/2;
                    cube_idx[0] = cube_i;
                    cube_idx[1] = 2*cube_size + cube_j;
                }
            }

            if(vec_cartesian_z1[2] > 0)
            { // check top
                if(abs(vec_cartesian_z1[0]) <= 1 && abs(vec_cartesian_z1[1]) <= 1)
                {
                    int cube_i, cube_j;
                    cube_i = cube_size*(1 - vec_cartesian_z1[0])/2;
                    cube_j = cube_size*(1 - vec_cartesian_z1[1])/2;
                    cube_idx[0] = cube_i;
                    cube_idx[1] = 4*cube_size + cube_j;
                }
            }
            else
            { // check bottom
                if(abs(vec_cartesian_z1[0]) <= 1 && abs(vec_cartesian_z1[1]) <= 1)
                {
                    int cube_i, cube_j;
                    cube_i = cube_size*(1 + vec_cartesian_z1[0])/2;
                    cube_j = cube_size*(1 - vec_cartesian_z1[1])/2;
                    cube_idx[0] = cube_i;
                    cube_idx[1] = 5*cube_size + cube_j;
                }
            }

            out_data[i*out_width + j] = im_data[cube_idx[0]*6*cube_size + cube_idx[1]];
        }
    }

    return out;
}
