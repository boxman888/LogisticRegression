#ifndef USPS_H
#define USPS_H

#ifndef COL_SIZE
#define COL_SIZE 256 // The # of dims will be const over train and test data.
#endif

class USPS{
public:
    std::string feature[COL_SIZE];

    void fill(USPS& temp);
};

std::istream& operator>>(std::istream& str, USPS& data);
#endif
