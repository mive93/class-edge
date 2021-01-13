#ifndef MESSAGE_H
#define MESSAGE_H

#include <iostream>
#include <iomanip>
#include <cmath>

#include "communicator.hpp"
#include "objects.hpp"
#include "messages.hpp"

#include "data.h"

/*
 * Converts orientation from radians to quantized degrees.
 * The given yaw is converted first into degrees by mutiplying it by 
 * 57.29 ( = 180/pi). The the degrees are quantized between 0 and 255 by 
 * mutiplying it by 17 / 24 ( = 255/360).
 *
 * @param yaw in radians
 * @return orientation in quantized degrees
 */
uint8_t orientationToUint8(const float yaw);

/*
 *  Converts velocity from m/s to quantized km/h every 1/2 km/h   
 *  The given velocity ( required in m/s) is converted into km/h (by 
 *  multiplying it for 3.6). This value is multiplied by 2 to quantize it 
 *  every half kilometer. In an urban environment we consider a max 
 *  velocity of 127 km/h.We fit 127 on a byte with a multiplication by 2. 
 *  Each increment corresponds to a speed greater than 1/2 km/h.
 *
 *  @param vel velocity in m/s
 *  @return quantized velocity in km/(30min)
 */
uint8_t velocityToUint8(const float vel);

bool checkClass(const int cl, const edge::Dataset_t dataset);
Categories classToCategory(const int cl, const edge::Dataset_t dataset);
RoadUser getRoadUser(const std::vector<int> camera_id, const double latitude, const double longitude, const std::vector<int> object_id, const float velocity, const float orientation, const float precision, const int cl, edge::Dataset_t dataset );

unsigned long long getTimeMs();

#endif /*MESSAGE_H*/
