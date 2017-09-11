#include <hpp/profUtils.h>

namespace hpp {

// TIMER //

Timer::Timer() {
}

void Timer::reset() { 
    duration = 0.0;
    running = false;
}

void Timer::start() {
    if (running) {
        std::cerr << "Warning: already started timer" << std::endl;
    }
    else {
        clock_gettime(CLOCK_REALTIME, &beg_);
        running = true;
    }
}

void Timer::stop() {
    if (!running) {
        std::cerr << "Warning: already stopped timer" << std::endl;
    }
    else {
        clock_gettime(CLOCK_REALTIME, &end_);
        duration += end_.tv_sec-beg_.tv_sec + (end_.tv_nsec-beg_.tv_nsec)/1e9;
        running = false;
    }
}

} //END NAMESPACE HPP