#include "message.h"

uint8_t orientationToUint8(const float yaw)
{
    uint8_t orientation = uint8_t((int((yaw * 57.29 + 360)) % 360) * 17 / 24);
    return orientation;
}

uint8_t velocityToUint8(const float vel)
{
    uint8_t velocity = uint8_t(std::abs(vel * 3.6 * 2));
    return velocity;
} 

Categories classToCategory(const int cl)
{
    Categories cat;
    switch (cl)
    {
    case 0:
        cat = Categories::C_person;
        break;
    case 1:
        cat = Categories::C_car;
        break;
    case 2:
        cat = Categories::C_car;
        break;
    case 3:
        cat = Categories::C_bus;
        break;
    case 4:
        cat = Categories::C_motorbike;
        break;
    case 5:
        cat = Categories::C_bycicle;
        break;
    }
    return cat;
}

RoadUser getRoadUser(const double latitude, const double longitude, const float velocity, const float orientation, const int cl )
{
    RoadUser r;
    r.latitude = static_cast<float>(latitude);
    r.longitude = static_cast<float>(longitude);
    r.speed = velocityToUint8(velocity);
    r.orientation = orientationToUint8(orientation);
    r.category = classToCategory(cl);
    return r;
}

unsigned long long getTimeMs() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    unsigned long long t_stamp_ms = (unsigned long long)(tv.tv_sec) * 1000 + (unsigned long long)(tv.tv_usec) / 1000;
    return t_stamp_ms;
}
