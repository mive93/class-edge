#include "BackGroundSuppression2.h"

bool compareTile(const edge::Tile &a, const edge::Tile &b) {
    return a.height > b.height;
}

bool compareBoxImage(const cv::Mat &a, const cv::Mat &b) {
    // std::cout<<"a "<<a.cols<<":"<<a.rows<<std::endl;
    // std::cout<<"b "<<b.cols<<":"<<b.rows<<std::endl;
    // return (a.cols*a.rows > b.cols*b.rows);
    return (a.rows > b.rows);
}

namespace edge {

BackGroundSuppression2::BackGroundSuppression2(const bool on_gpu, const int width, const int height,
                                               const int n_classes) : BackGroundSuppression(on_gpu){
    for(int c=0; c<n_classes; c++) {
        int offset = c*123457 % n_classes;
        float r = getColor(2, offset, n_classes);
        float g = getColor(1, offset, n_classes);
        float b = getColor(0, offset, n_classes);
        colors[c] = cv::Scalar(int(255.0*b), int(255.0*g), int(255.0*r));
    }
}

std::vector<cv::Mat> BackGroundSuppression2::getBatchingFromBackground(const cv::Mat& frame_in) {
    std::vector<std::vector<cv::Point>> contours;
    findContours(backsup.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    int th_blob_size = 100;
    backBBres = frame_in.clone();
    std::vector<cv::Mat> batch_info;
    int k=0;
    for (int i = 0; i < contours.size(); ++i)
    {
        // Remove small blobs
        if (contours[i].size() < th_blob_size)
        {
            continue;
        }
        
        cv::Rect box = cv::boundingRect(contours[i]);
        edge::Tile t = {k++, 0, box.x, box.y, box.width, box.height};
        original_tiles.push_back(t);
        batch_info.push_back(frame_in(box));
        rectangle(backBBres, box, cv::Scalar(0,255,0), 1);
    }
    return batch_info;
}

void BackGroundSuppression2::showBoxImage(std::vector<cv::Mat> &box_image) {
    for(auto im : box_image) {
        cv::imshow("crop", im);
        cv::waitKey(1);
        sleep(1);
    }
} 

float BackGroundSuppression2::overlap(const float p1, const float d1, const float p2, const float d2){
    float l1 = p1 - d1/2;
    float l2 = p2 - d2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = p1 + d1/2;
    float r2 = p2 + d2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float BackGroundSuppression2::boxesIntersection(const tk::dnn::box &b, const edge::Tile &t){
    float width = this->overlap(t.x, t.width, b.x, b.w);
    float height = this->overlap(t.y, t.height, b.y, b.h);
    if(width < 0 || height < 0) 
        return 0;
    float area = width*height;
    return area;
}

float BackGroundSuppression2::boxesUnion(const tk::dnn::box &b, const edge::Tile &t){
    float i = this->boxesIntersection(b, t);
    float u = t.width*t.height + b.w*b.h - i;
    return u;
}

float BackGroundSuppression2::IoU(const tk::dnn::box &b, const edge::Tile &t){
    float I = this->boxesIntersection(b, t);
    float U = this->boxesUnion(b, t);
    if (I == 0 || U == 0) 
        return 0;
    return I / U;
}

void BackGroundSuppression2::concatDetections(std::vector<tk::dnn::box>& detected) {
    std::vector<edge::Tile> new_tiles = cl.getTiles();
    int tile_id;
    float max_overlap;
    float aus;

    for(int i=0; i<detected.size(); i++) { 
        tile_id = 0;
        max_overlap = 0.0;
        for (int j=0; j<new_tiles.size(); j++) {
            aus = IoU(detected[i], new_tiles[j]);
            if(aus > max_overlap) {
                max_overlap = aus;
                tile_id = j;
            }
        }
        detected[i].x = detected[i].x - new_tiles.at(tile_id).x + original_tiles.at(new_tiles.at(tile_id).image_id).x;
        detected[i].y = detected[i].y - new_tiles.at(tile_id).y + original_tiles.at(new_tiles.at(tile_id).image_id).y;
    }
}

void BackGroundSuppression2::drawTiles(std::vector<cv::Mat>& original_frames, 
                std::vector<std::vector<tk::dnn::box>> batchDetected, std::vector<std::string> classesNames) {
    tk::dnn::box b;
    int x0, w, x1, y0, h, y1;
    int objClass;
    std::string det_class;
    std::vector<edge::Tile> new_tiles = cl.getTiles();
    int baseline = 0;
    float font_scale = 0.5;
    int thickness = 2;   
    int tile_id;
    float max_overlap;
    float aus;
    for(int bi=0; bi<original_frames.size(); ++bi){
        // draw dets
        for(int i=0; i<batchDetected[bi].size(); i++) { 
            b           = batchDetected[bi][i];
            tile_id = 0;
            max_overlap = 0.0;
            for (int j=0; j<new_tiles.size(); j++) {
                aus = IoU(b, new_tiles[j]);
                if(aus > max_overlap) {
                    max_overlap = aus;
                    tile_id = j;
                }
            }
            x0   		= b.x - new_tiles.at(tile_id).x + original_tiles.at(new_tiles.at(tile_id).image_id).x;
            x1   		= x0 + b.w;
            y0   		= b.y - new_tiles.at(tile_id).y + original_tiles.at(new_tiles.at(tile_id).image_id).y;
            y1   		= y0 + b.h;
            det_class 	= classesNames[b.cl];

            // draw rectangle
            cv::rectangle(original_frames[bi], cv::Point(x0, y0), cv::Point(x1, y1), colors[b.cl], 2); 
            // draw label
            cv::Size text_size = getTextSize(det_class, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
            cv::rectangle(original_frames[bi], cv::Point(x0, y0), cv::Point((x0 + text_size.width - 2), (y0 - text_size.height - 2)), colors[b.cl], -1);                      
            cv::putText(original_frames[bi], det_class, cv::Point(x0, (y0 - (baseline / 2))), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);
        }
    }
}

cv::Mat BackGroundSuppression2::update(const cv::Mat& frame_in) {
    getBackgroundSuppression(frame_in);
    batchinfo.clear();
    original_tiles.clear();
    batchinfo = getBatchingFromBackground(frame_in);
    // showBoxImage(batchinfo);
    // sort tiles

    std::sort( original_tiles.begin(), original_tiles.end(), compareTile);
    // sort the images
    // std::sort( batchinfo.begin(), batchinfo.end(), compareBoxImage );
    //create image: 320, 544
    cl.init(original_tiles, batchinfo);
    // collectTile(batchinfo, collage, tiles);
    frame_out = cl.getCollage().clone();
    
    return frame_out;
}
}