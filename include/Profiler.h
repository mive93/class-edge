#ifndef PROFILER_H
#define PROFILER_H

#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <iomanip>
#include <chrono>

#include "tkDNN/utils.h"

#define MAX_MIN 9999999

namespace edge{

struct stats{
    std::chrono::time_point<std::chrono::system_clock> start;
    std::vector<double> diff;
    double min  = MAX_MIN;
    double max  = 0;
    double sum  = 0;
    int count   = 0;
};

class Profiler{
    std::string name;
    std::map<std::string,stats> timers;

    float getMin(){}
    float getMax(){}
    float getA(){}
    
public:
    Profiler(std::string profiler_name)  {
        name = profiler_name;
    }
    ~Profiler() {}
    void tick(std::string timer_name){
        timers[timer_name].start = std::chrono::high_resolution_clock::now();
    }
    void tock(std::string timer_name){
        if ( timers.find(timer_name) == timers.end() ) 
            FatalError("Timer never started (no tick associated)");
        auto end = std::chrono::high_resolution_clock::now();
        double diff = std::chrono::duration_cast<std::chrono::microseconds>(end-timers[timer_name].start).count();
        timers[timer_name].diff.push_back(diff);
        timers[timer_name].min = (diff < timers[timer_name].min) ? diff : timers[timer_name].min;
        timers[timer_name].max = (diff > timers[timer_name].max) ? diff : timers[timer_name].max;
        timers[timer_name].sum += diff;
    }
    void printStats(){
        int max_lenght = 0;
        for (auto const& t : timers)
            if(t.first.size() > max_lenght)
                max_lenght = t.first.size();
        max_lenght += 10;

        std::cout<<"######################### Profiler "<< name <<" #########################"<<std::endl;

        for (auto& t : timers)
        {
            accumulate( t.second.diff.begin(), t.second.diff.end(), 0.0) / t.second.diff.size(); 
            std::cout << t.first << std::fixed << std::setprecision(2)
                    << std::setfill(' ') << std::setw (max_lenght - t.first.size())
                    << "\t\tavg(ms): " << t.second.sum / float(t.second.diff.size()) / 1000
                    << "\tmin(ms): " << t.second.min / 1000
                    << "\tmax(ms): " << t.second.max / 1000
                    << std::endl ;

            t.second.min = MAX_MIN;
            t.second.max = 0;
            t.second.sum = 0;
            t.second.diff.clear();
        }

    }
};

}
#endif /*PROFILER_H*/
