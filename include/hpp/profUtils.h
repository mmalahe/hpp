/** @file profUtils.h
* @author Michael Malahe
* @brief
*/

#ifndef HPP_PROFUTILS_H
#define HPP_PROFUTILS_H

#include <iostream>
#include <ctime>
#include <hpp/config.h>

namespace hpp {

class Timer {
public:
    Timer();
    void reset();
    void start();
    void stop();
    double getDuration(){return duration;}
private:
    double duration = 0.0;
    bool running = false;
    timespec beg_, end_;
};

} //END NAMESPACE HPP

#endif /* HPP_PROFUTILS_H */