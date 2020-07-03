#include <string>
#include <sstream>
#include <istream>
#include "USPS.h"

std::istream& operator>>(std::istream& str, USPS& data){
    bool error = false;
    std::string line;
    USPS temp;
    if (std::getline( str, line )){
        std::stringstream iss(line);

        for (int i = 0; i < COL_SIZE; i++){
            std::getline(iss, temp.feature[i], ',');
        }

        if (!error)
            data.fill(temp);
        else
            str.setstate(std::ios::failbit);
        
    }
    return str;
}
// Fill data of current object with that held in the temp object.
void USPS::fill(USPS& other){
    for (int i = 0; i < COL_SIZE; i++){
        std::swap(feature[i], other.feature[i]);
    }
}