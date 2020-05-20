#include "Collage.h"

namespace edge{
    
void Collage::init(std::vector<edge::Tile> original_tiles, std::vector<cv::Mat> &src) {
    // setNewDim(src);
    //320, 544
    H = 320;
    W = 544;
    A = 20;
    collage = new cv::Mat(H, W,CV_8UC3, cv::Scalar(0, 0, 0));
    new_x = 0;
    new_y = 0;
    //fill the filled_image with 0;
    tiles.clear();
    filled_image.clear();
    for (int i=0; i<H; i++) 
        filled_image.push_back(0);

    collectTile(original_tiles, src);
}

void Collage::setNewDim(std::vector<cv::Mat> &box_image) {
    //find upper bound of image and new dimension
    int x=0,y=0;
    for(auto im : box_image) {
        x += im.cols;
        y += im.rows;
    }
    std::cout<<"for "<<box_image.size()<<" image, we obtain "<<x<<"x"<<y<<" size"<<std::endl;
    std::cout<<"new dimension could be: "<<x/2<<"x"<<y/2<<std::endl;
}

void Collage::resize() {
    int new_H = H + int(H/100 * A);
    int new_W = W + int(W/100 * A);
    cv::Mat aus(new_H, new_W,CV_8UC3, cv::Scalar(0, 0, 0));
    (*collage).copyTo(aus(cv::Rect(0,0,W,H)));
    delete collage;
    for (int i=H; i<new_H; i++) 
        filled_image.push_back(0);
    H = new_H;
    W = new_W;
    collage = new cv::Mat(aus);
}

bool Collage::searchSpace(const int w, const int h) {
    // std::vector<std::vector<int>> splots;
    // for(int i =0; i<filled_image.size(); i++) {
    //     std::cout<<"r: "<<i<<" - "<<filled_image.at(i)<<": ";
    //     for(int j=0; j<filled_image.at(i); j++)
    //         std::cout<<"* ";
    //     std::cout<<"\n";
    // }
    int prov_y = filled_image.size()-1;
    int prov_x = filled_image.at(filled_image.size()-1);
    int count=0;
    for(int id=filled_image.size()-1; id>=0; id--) {
        //while you scroll upper and meet a value minor of prov_x, restart teh counting
        if(filled_image.at(id) < prov_x)
            count = 0;
        if(count >= h) {
            // go to upper as possible
            if(filled_image.at(id) == prov_x) {
                count ++;
                prov_y = id;
                continue;
            }
            break;
        }
        prov_y = id;
        if(filled_image.at(id) != prov_x) {
            if(filled_image.at(id) < prov_x)
                count = 0;
            prov_x = filled_image.at(id);
            
        }
        else {
            count ++;
        }
    }
    if(count >= h) {//there is a solution
        new_x = prov_x;
        new_y = prov_y;
        return true;
    }
    else
        return false;
        
}

void Collage::findNewPosition(const int w, const int h) {
    //search the new position
    while(w > W && h > H) {
        // std::cout<<"resize()\n";
        resize();
    }
    //search space
    while(!searchSpace(w,h) && new_x+w > W)
        resize();
    //update filled_image
    // std::cout<<"new_x, new_y fill: "<<new_x<<", "<<new_y<<std::endl;
    // std::cout<<"w, h fill: "<<w<<", "<<h<<std::endl;
    for(int i=new_y; i<new_y+h; i++) {
        // if(filled_image.at(i) != new_x)
            // std::cout<<"ERROR\n";
        filled_image.at(i) = new_x + w;
    }
}

void Collage::collectTile(std::vector<edge::Tile> original_tiles, std::vector<cv::Mat> &box_image) {
    //320, 544
    /* insert the images for column (x=0, y)*/
    for(int i = 0; i < box_image.size(); i++) {
        // std::cout<<"image: "<<i<<" - "<<original_tiles.at(i).image_id<<std::endl;
        cv::Mat s = box_image.at(original_tiles.at(i).image_id);
        // std::cout<<"s size: "<<s.cols<<" - "<<s.rows<<std::endl;
        findNewPosition(s.cols, s.rows);
        // std::cout<<"new_x, new_y = "<<new_x<<", "<<new_y<<std::endl;
        Tile t;
        t.image_id = original_tiles.at(i).image_id;
        t.sort_id = i;
        t.x = new_x;
        t.y = new_y;
        t.width = s.cols;
        t.height = s.rows;
        //insert image and update position
        tiles.push_back(t);
        s.copyTo((*collage)(cv::Rect(new_x,new_y,s.cols,s.rows)));
    }
}

cv::Mat Collage::getCollage() {
    return *collage;
}
std::vector<edge::Tile>& Collage::getTiles() {
    return tiles;
}
}