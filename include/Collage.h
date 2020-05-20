#ifndef COLLAGE_H
#define COLLAGE_H

#include <iostream>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

namespace edge {

struct Tile {
    int image_id;
    int sort_id;
    int x, y;
    int width, height;
};

class Collage {
private:
    cv::Mat *collage;
    std::vector<edge::Tile> tiles;
    int H;
    int W;
    int A;
    std::vector<int> filled_image;
    int new_x;
    int new_y;
    int max_x;
    int max_y;
    int min_x;
    int min_y;
    bool flag;
    bool first;
    int id;
    int elem_for_col;
    
public:
    Collage() {};
    ~Collage() {};
    void init(std::vector<edge::Tile> original_tiles, std::vector<cv::Mat> &src);
    void setNewDim(std::vector<cv::Mat> &box_image);
    void resize();
    bool searchSpace(const int w, const int h);
    void findNewPosition(const int w, const int h);
    void collectTile(std::vector<edge::Tile> original_tiles, std::vector<cv::Mat> &box_image);
    cv::Mat getCollage();
    std::vector<edge::Tile>& getTiles();
    
};
}

#endif /* COLLAGE_H */