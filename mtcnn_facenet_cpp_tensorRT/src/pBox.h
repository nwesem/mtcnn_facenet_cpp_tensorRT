#ifndef PBOX_H
#define PBOX_H
#include <stdlib.h>
#include <iostream>

using namespace std;
#define mydataFmt float


struct pBox
{
	mydataFmt *pdata;
	int width;
	int height;
	int channel;
};
struct Bbox
{
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    float area;
    bool exist;
    mydataFmt ppoint[10];
    mydataFmt regreCoord[4];
};

struct orderScore
{
    mydataFmt score;
    int oriOrder;
};
#endif