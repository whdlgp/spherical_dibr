#include "opencv2/core/utility.hpp"

#define DEBUG_PRINT

#ifdef DEBUG_PRINT
#define DEBUG_PRINT_ERR(x) (std::cerr << x << std::endl)
#define DEBUG_PRINT_OUT(x) (std::cout << x << std::endl)

#define START_TIME(timeval) int64 (timeval) = cv::getTickCount()
#define STOP_TIME(timeval) (timeval) = cv::getTickCount() - (timeval); \
                             DEBUG_PRINT_OUT(#timeval" execution time : " \
                              << (timeval)/(cv::getTickFrequency() * 1.0000) \
                              << " sec")
#else
#define DEBUG_PRINT_ERR(x)
#define DEBUG_PRINT_OUT(x)

#define START_TIME(timeval)
#define STOP_TIME(timeval)
#endif