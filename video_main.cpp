#include "srcs/spherical_dibr.hpp"
#include "srcs/debug_print.h"
#include "srcs/INIReader.h"

#include <iostream>
#include <sstream>
#include <vector>

using namespace std;
using namespace cv;

#define RAD(x) M_PI*(x)/180.0
#define DEGREE(x) 180.0*(x)/M_PI

vector<string> string_parse(string str, string tok)
{
    vector<string> token;
    size_t pos = 0;
    while ((pos = str.find(tok)) != std::string::npos) 
    {
        token.push_back(str.substr(0, pos));
        str.erase(0, pos + tok.length());
    }
    token.push_back(str);

    return token;
}

Vec3d string_to_vec(string str)
{
    Vec3d vec;
    vector<string> vec_str = string_parse(str, " ");
    vec[0] = stod(vec_str[0]);
    vec[1] = stod(vec_str[1]);
    vec[2] = stod(vec_str[2]);

    return vec;
}

Mat dibr(vector<Mat>& im, vector<Mat>& depth_double
        , vector<camera_info> cam_info, camera_info vt_cam_info
        , int cam_num, int filter_option, int render_option)
{
    vector<Mat> depth_map_result(cam_num);
    vector<Mat> img_result(cam_num);
    vector<double> cam_dist(cam_num);

    // Start rendering
    for(int i = 0; i < cam_num; i++)
    {
        START_TIME(render_one_image);
        spherical_dibr spd;
        cout << "image " << i << " now do rendering" << endl;

        // Calculate R|t to render virtual view point
        Mat cam_rot_mat = spd.eular2rot(Vec3f(RAD(cam_info[i].rot[0]), RAD(cam_info[i].rot[1]), RAD(cam_info[i].rot[2])));
        Mat rot_mat_inv = cam_rot_mat.t();
        Mat vt_rot_mat = spd.eular2rot(Vec3f(RAD(vt_cam_info.rot[0]), RAD(vt_cam_info.rot[1]), RAD(vt_cam_info.rot[2])));
        Mat r = vt_rot_mat*rot_mat_inv;
        Vec3d t_tmp = vt_cam_info.tran-cam_info[i].tran;
        Vec3d t;
        t[0] = cam_rot_mat.at<double>(0,0)*t_tmp[0] + cam_rot_mat.at<double>(0,1)*t_tmp[1] + cam_rot_mat.at<double>(0,2)*t_tmp[2];
        t[1] = cam_rot_mat.at<double>(1,0)*t_tmp[0] + cam_rot_mat.at<double>(1,1)*t_tmp[1] + cam_rot_mat.at<double>(1,2)*t_tmp[2];
        t[2] = cam_rot_mat.at<double>(2,0)*t_tmp[0] + cam_rot_mat.at<double>(2,1)*t_tmp[1] + cam_rot_mat.at<double>(2.2)*t_tmp[2];

        // Render virtual view point
        spd.render(im[i], depth_double[i]
                   , r, t
                   , cam_info[i], vt_cam_info
                   , render_option, filter_option);
        STOP_TIME(render_one_image);
        
        if(filter_option == spd.FILTER_MEDIAN)
        {
            depth_map_result[i] = spd.depth_out_median;
            img_result[i] = spd.im_out_inverse_median;
        }
        else if(filter_option == spd.FILTER_CLOSING)
        {
            depth_map_result[i] = spd.depth_out_closing;
            img_result[i] = spd.im_out_inverse_closing;
        }

        cam_dist[i] = sqrt(t[0]*t[0] + t[1]*t[1] + t[2]*t[2]);
    }

    int width = im[0].cols;
    int height = im[0].rows;
    Mat blended_img(height, width, CV_16UC3);
    vector<Vec3w*> im_data(cam_num);
    vector<double*> depth_data(cam_num);
    Vec3w* blended_data = (Vec3w*)blended_img.data;
    for(int i = 0; i < cam_num; i++)
    {
        im_data[i] = (Vec3w*)img_result[i].data;
        depth_data[i] = (double*)depth_map_result[i].data;
    }

    START_TIME(Blend_image);
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            Vec3d pixel_val = 0;
            double dist_sum = 0;
            double threshold = 0.01; // consider below than threshold are occluded area
            int valid_count = 0;
            for(int c = 0; c < cam_num; c++)
            {
                if(depth_data[c][i*width + j] > threshold)
                {
                    valid_count++;
                    dist_sum += 1/cam_dist[c];
                }
            }
            for(int c = 0; c < cam_num; c++)
            {
                if(depth_data[c][i*width + j] > threshold)
                {
                    if(valid_count > 1)
                    {
                        pixel_val[0] += (1/cam_dist[c]/dist_sum)*im_data[c][i*width + j][0];
                        pixel_val[1] += (1/cam_dist[c]/dist_sum)*im_data[c][i*width + j][1];
                        pixel_val[2] += (1/cam_dist[c]/dist_sum)*im_data[c][i*width + j][2];
                    }
                    else if(valid_count == 1)
                    {
                        pixel_val[0] += im_data[c][i*width + j][0];
                        pixel_val[1] += im_data[c][i*width + j][1];
                        pixel_val[2] += im_data[c][i*width + j][2];
                    }
                }
            }
            blended_data[i*width + j][0] = pixel_val[0];
            blended_data[i*width + j][1] = pixel_val[1];
            blended_data[i*width + j][2] = pixel_val[2];
        }
    }
    STOP_TIME(Blend_image);

    return blended_img;
}

Mat map_distance(Mat& depth, double min_pixel, double max_pixel, double min_dist, double max_dist)
{
    Mat depth_double(depth.rows, depth.cols, CV_64FC1);
    unsigned char* depth_data = (unsigned char*)depth.data;
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

int main(int argc, char *argv[])
{   
    if(argc  < 2)
    {
	   cout << "usage:" << argv[0]   <<  " camera_info_file" << endl;
	   return 0;
    }

    INIReader reader(argv[1]);

    if (reader.ParseError() < 0)
    {
        std::cout << "Can't load 'camera_info.ini'\n";
        return 1;
    }

    int num_of_camera = reader.GetInteger("option", "number", -1);
    std::cout << "Config loaded from 'camera_info.ini'\n"
              << "number of camera : " <<  num_of_camera << "\n" << endl;
    int filter_option = reader.GetInteger("option", "filteroption", -1);
    int render_option = reader.GetInteger("option", "renderoption", -1);
    string output_dir = reader.Get("option", "output", "UNKNOWN") + "/";

    // Read Camera information
    int cam_num = num_of_camera;
    vector<camera_info> cam_info(cam_num);
    for(int i = 0; i < cam_num; i++)
    {
        string cams = "camera";
        cams = cams + to_string(1+i);
        cout << "Read camera: " << cams << endl;

        cam_info[i].projection = reader.GetInteger(cams, "type", -1);
        cam_info[i].cam_name = reader.Get(cams, "imagename", "UNKNOWN");
        cam_info[i].depth_name = reader.Get(cams, "depthname", "UNKNOWN");
        cam_info[i].depth_min = reader.GetReal(cams, "depthmin", -1);
        cam_info[i].depth_max = reader.GetReal(cams, "depthmax", -1);
        cam_info[i].rot = string_to_vec(reader.Get(cams, "rotation", "UNKNOWN"));
        cam_info[i].tran = string_to_vec(reader.Get(cams, "translation", "UNKNOWN"));
    }

    // Read Virtual View Point information
    camera_info vt_cam_info;
    vt_cam_info.rot = string_to_vec(reader.Get("virtualview", "rotation", "UNKNOWN"));
    vt_cam_info.tran = string_to_vec(reader.Get("virtualview", "translation", "UNKNOWN"));
    vt_cam_info.depth_min = reader.GetReal("virtualview", "depthmin", -1);
    vt_cam_info.depth_max = reader.GetReal("virtualview", "depthmax", -1);
    vt_cam_info.cam_name = reader.Get("virtualview", "outputname", "UNKNOWN");
    vt_cam_info.fps = reader.GetInteger("virtualview", "fps", -1);

    vector<VideoCapture> rgb_capture(cam_num);
    vector<VideoCapture> depth_capture(cam_num);
    for(int i = 0; i < cam_num; i++)
    {
        rgb_capture[i] = VideoCapture(cam_info[i].cam_name);
        depth_capture[i] = VideoCapture(cam_info[i].depth_name);
    }

    VideoWriter vt_writer;
    bool is_empty = false;
    while(1)
    {
        vector<Mat> im(cam_num);
        vector<Mat> depth_double(cam_num);

        for(int i = 0; i < cam_num; i++)
        {
            //convert 8bit 3channel image and depth to:
            //16bit RGB image
            //double 1channel depth image
            Mat im_8bit, depth_8bit;
            rgb_capture[i] >> im_8bit;
            depth_capture[i] >> depth_8bit;

            if(im_8bit.empty() || depth_8bit.empty())
            {
                is_empty = true;
                break;
            }

            im_8bit.convertTo(im_8bit, CV_16UC3);
            im[i] = im_8bit*256;
            
            Mat depth_8bit_chan[3];
            Mat depth;
            split(depth_8bit, depth_8bit_chan);
            depth_8bit_chan[0].convertTo(depth, CV_64FC1);
            depth_double[i] = depth * (cam_info[i].depth_max-cam_info[i].depth_min)/255.0 + cam_info[i].depth_min;
        }
        if(is_empty)
            break;

        //do dibr, blending
        Mat blended_img = dibr(im, depth_double, cam_info, vt_cam_info, cam_num, filter_option, render_option);

        //write video
        bool is_first = true;
        if(is_first)
        {
            is_first = false;
            vt_writer.open(vt_cam_info.cam_name, VideoWriter::fourcc('M', 'J', 'P', 'G'), vt_cam_info.fps
                            , Size(blended_img.cols, blended_img.rows), true);
            if(!vt_writer.isOpened())
            {
                cout << "error while initialize videowriter" << endl;
                return 1;
            }
        }

        Mat save_frame = blended_img/256;
        save_frame.convertTo(save_frame, CV_8UC3);
        vt_writer.write(save_frame);
    }

    return 0;
}
