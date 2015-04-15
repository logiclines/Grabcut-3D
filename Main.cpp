#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "gcut.h"
#include <iostream>

using namespace std;
using namespace cv;
const Scalar RED = Scalar(0,0,255);
const Scalar PINK = Scalar(230,130,255);
const Scalar BLUE = Scalar(255,0,0);
const Scalar LIGHTBLUE = Scalar(255,255,160);
const Scalar GREEN = Scalar(0,255,0);

const int BGD_KEY = EVENT_FLAG_CTRLKEY;
const int FGD_KEY = EVENT_FLAG_SHIFTKEY;

static void menu(){
	cout << "<Menu>\n N: RGB image segmentation \n A: Display Depth image \n Z: Depth image segmentation \n S: Dispaly Normal image \n X: Normal image segmentation \n" << endl;
}
static void getBinMask( const Mat& comMask, Mat& binMask )
{
    if( binMask.empty() || binMask.rows!=comMask.rows || binMask.cols!=comMask.cols )
        binMask.create( comMask.size(), CV_8UC1 );
    binMask = comMask & 1;
}

class display
{
public:
    enum{ NOT_SET = 0, IN_PROCESS = 1, SET = 2 };
    static const int radius = 2;
    static const int thickness = -1;

    void reset();
    void setImageAndWinName( const Mat& _image, const string& _winName );
    Mat showImage() const;
	Mat showImage2() const;
    void mouseClick( int event, int x, int y, int flags, void* param );
    int nextIter();
	int nextIter2();
    int getIterCount() const { return iterCount; }
private:
    void setRectInMask();
    void setLblsInMask( int flags, Point p, bool isPr );

    const string* winName;
    const Mat* image;
	const Mat* dep;
    Mat mask;
    
    uchar rectState, lblsState, prLblsState;
    bool isInitialized;

    Rect rect;
    vector<Point> fgdPxls, bgdPxls, prFgdPxls, prBgdPxls;
    int iterCount;
};
Mat binMask;
void display::reset()
{
    if( !mask.empty() )
        mask.setTo(Scalar::all(GC_BGD));
    bgdPxls.clear(); fgdPxls.clear();
    prBgdPxls.clear();  prFgdPxls.clear();

    isInitialized = false;
    rectState = NOT_SET;
    lblsState = NOT_SET;
    prLblsState = NOT_SET;
    iterCount = 0;
}

void display::setImageAndWinName( const Mat& _image, const string& _winName  )
{
    if( _image.empty() || _winName.empty() )
        return;
    image = &_image;
    winName = &_winName;
    mask.create( image->size(), CV_8UC1);
    reset();
}

Mat display::showImage() const
{

    Mat res;

    if( !isInitialized )
        image->copyTo( res );
    else
    {
        getBinMask( mask, binMask );
        image->copyTo( res, binMask );
    }

    vector<Point>::const_iterator it;
    for( it = bgdPxls.begin(); it != bgdPxls.end(); ++it )
        circle( res, *it, radius, BLUE, thickness );
    for( it = fgdPxls.begin(); it != fgdPxls.end(); ++it )
        circle( res, *it, radius, GREEN, thickness );

    if( rectState == IN_PROCESS || rectState == SET )
        rectangle( res, Point( rect.x, rect.y ), Point(rect.x + rect.width, rect.y + rect.height ), RED, 2);

    imshow( *winName, res );
	return res;
}

Mat display::showImage2() const
{
    Mat res;
    if( !isInitialized )
        image->copyTo( res );
    else
    {
        getBinMask( mask, binMask );
        image->copyTo( res, binMask );
    }

        rectangle( res, Point( rect.x, rect.y ), Point(rect.x + rect.width, rect.y + rect.height ), RED, 2);

    imshow( *winName, res );
	return res;
}

void display::setRectInMask()
{
    CV_Assert( !mask.empty() );
    mask.setTo( GC_BGD );
    rect.x = max(0, rect.x);
    rect.y = max(0, rect.y);
    rect.width = min(rect.width, image->cols-rect.x);
    rect.height = min(rect.height, image->rows-rect.y);
    (mask(rect)).setTo( Scalar(GC_PR_FGD) );
}

void display::setLblsInMask( int flags, Point p, bool isPr )
{
    vector<Point> *bpxls, *fpxls;
    uchar bvalue, fvalue;
    if( !isPr )
    {
        bpxls = &bgdPxls;
        fpxls = &fgdPxls;
        bvalue = GC_BGD;
        fvalue = GC_FGD;
    }
    else
    {
        bpxls = &prBgdPxls;
        fpxls = &prFgdPxls;
        bvalue = GC_PR_BGD;
        fvalue = GC_PR_FGD;
    }
    if( flags & BGD_KEY )
    {
        bpxls->push_back(p);
        circle( mask, p, radius, bvalue, thickness );
    }
    if( flags & FGD_KEY )
    {
        fpxls->push_back(p);
        circle( mask, p, radius, fvalue, thickness );
    }
}

void display::mouseClick( int event, int x, int y, int flags, void* )
{
    switch( event )
    {
    case EVENT_LBUTTONDOWN: 
        {
            bool isb = (flags & BGD_KEY) != 0,
                 isf = (flags & FGD_KEY) != 0;
            if( rectState == NOT_SET && !isb && !isf )
            {
                rectState = IN_PROCESS;
                rect = Rect( x, y, 1, 1 );
            }
            if ( (isb || isf) && rectState == SET )
                lblsState = IN_PROCESS;
        }
        break;
    case EVENT_RBUTTONDOWN: 
        {
            bool isb = (flags & BGD_KEY) != 0,
                 isf = (flags & FGD_KEY) != 0;
            if ( (isb || isf) && rectState == SET )
                prLblsState = IN_PROCESS;
        }
        break;
    case EVENT_LBUTTONUP:
        if( rectState == IN_PROCESS )
        {
            rect = Rect( Point(rect.x, rect.y), Point(x,y) );
            rectState = SET;
            setRectInMask();
            CV_Assert( bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty() );
            showImage();
        }
        if( lblsState == IN_PROCESS )
        {
            setLblsInMask(flags, Point(x,y), false);
            lblsState = SET;
            showImage();
        }
        break;
    case EVENT_RBUTTONUP:
        if( prLblsState == IN_PROCESS )
        {
            setLblsInMask(flags, Point(x,y), true);
            prLblsState = SET;
            showImage();
        }
        break;
    case EVENT_MOUSEMOVE:
        if( rectState == IN_PROCESS )
        {
            rect = Rect( Point(rect.x, rect.y), Point(x,y) );
            CV_Assert( bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty() );
            showImage();
        }
        else if( lblsState == IN_PROCESS )
        {
            setLblsInMask(flags, Point(x,y), false);
            showImage();
        }
        else if( prLblsState == IN_PROCESS )
        {
            setLblsInMask(flags, Point(x,y), true);
            showImage();
        }
        break;
    }
}

int display::nextIter()
{
    if( isInitialized )
		gCut( *image, mask, rect);
    else
    {
        if( rectState != SET )
            return iterCount;

        if( lblsState == SET || prLblsState == SET )
			gCut( *image, mask, rect);
        else
			gCut( *image, mask, rect);

        isInitialized = true;
    }
    iterCount++;
    //bgdPxls.clear(); fgdPxls.clear();
   // prBgdPxls.clear(); prFgdPxls.clear();

    return iterCount;
}

int display::nextIter2()
{
    if( isInitialized )
        gCut( *image, mask, rect);
    else
    {
        if( lblsState == SET || prLblsState == SET )
            gCut( *image, mask, rect);
        else
            gCut( *image, mask, rect);

        isInitialized = true;
    }
    iterCount++;

    return iterCount;
}

display gcapp;

static void mouse_click( int event, int x, int y, int flags, void* param )
{
    gcapp.mouseClick( event, x, y, flags, param );
}


int loadDepthAsBinary( /* out: */ ::cv::Mat &img, /* in: */ std::string path, /* scale values: */ float alpha = 1.f )
{
    FILE *fp = fopen( path.c_str(), "rb" );
    if (!fp)
    {
        printf( "loadDepthAsBinary failed, when opening %s\n", path.c_str() );
        return false;
    }

    int w,h;
    fread( &h,sizeof(int),1,fp ); // height
    fread( &w,sizeof(int),1,fp ); // width
    float *p_depth = new float[h*w];

    fread( p_depth, sizeof(float), w*h, fp );

    img.create( h, w, CV_16UC1 );
    int p_depth_index = 0;
    for ( unsigned y = 0; y < img.rows; ++y  )
    {
        for ( unsigned x = 0; x < img.cols; ++x, ++p_depth_index )
        {
            img.at<ushort>(y,x) = p_depth[ p_depth_index ] * alpha;
        }
    }

    fclose( fp );
    if ( p_depth ) { delete [] p_depth; p_depth = NULL; }

    return EXIT_SUCCESS;
}

void main( int argc, char** argv )
{
	string filename = "C:/Users/asus/Documents/Visual Studio 2012/Projects/OpenCVTemplate/x64/Debug/img_0043.png";
	string filename2 = "C:/Users/asus/Documents/Visual Studio 2012/Projects/OpenCVTemplate/x64/Debug/img_0043_abs_smooth.png";
	string filename3 = "C:/Users/asus/Documents/Visual Studio 2012/Projects/OpenCVTemplate/x64/Debug/img_0043_smooth_normal_map.png";
	string filename4 = "C:/Users/asus/Documents/Visual Studio 2012/Projects/OpenCVTemplate/x64/Debug/img_0043_noisy_normal_map.png";
	
	Mat imageOriginal = imread( filename, 1 );
	Mat image;
	medianBlur(imageOriginal, image, 3);
	Mat imageD = imread( filename2, 1 );
	Mat imageN = imread( filename3, 1 );
	Mat imageNnotD = imread( filename4, 1 );

	const string winName = "RGB image";
    namedWindow( winName, WINDOW_AUTOSIZE );
    setMouseCallback( winName, mouse_click, 0 );

    gcapp.setImageAndWinName( image, winName );
    gcapp.showImage();

	menu();

	const string winName_a = "Depth image";
	const string winName_s = "Normal image";
	const string winName_d = "Original Normal image";
	Mat maskNow;

    for(;;)
    {
        int c = waitKey(0);
        switch( (char) c )
        {
		//reset
        case 'r':
            cout << endl;
            gcapp.reset();
            gcapp.showImage();
            break;
		//display depth image
		case 'a':			
			namedWindow( winName_a, WINDOW_AUTOSIZE );
			gcapp.setImageAndWinName( imageD, winName_a );
			maskNow = gcapp.showImage2();
			break;
		//display normal image
        case 's':			
			namedWindow( winName_s, WINDOW_AUTOSIZE );
			gcapp.setImageAndWinName( imageN, winName_s );
			maskNow = gcapp.showImage2();
			break;
		//display original normal image
        case 'd':			
			namedWindow( winName_d, WINDOW_AUTOSIZE );
			gcapp.setImageAndWinName( imageNnotD, winName_d );
			maskNow = gcapp.showImage2();
			break;
		//depth image segmentation
		case 'z':
		{	int iterCount2 = gcapp.getIterCount();
			cout << "Iteration times:" << iterCount2 << endl;
			int newIterCount2 = gcapp.nextIter2();
			if( newIterCount2 > iterCount2 )
            {
                maskNow = gcapp.showImage2();
				imwrite("C:/Users/asus/Documents/Visual Studio 2012/Projects/OpenCVTemplate/x64/Debug/maskD.png", maskNow);
                cout << "Iteration times:" << iterCount2 << endl;
            }
		}
			break;
		//normal image segmentation
		case 'x':
		{	int iterCount2 = gcapp.getIterCount();
			cout << "Iteration times:" << iterCount2 << endl;
			int newIterCount2 = gcapp.nextIter2();
			if( newIterCount2 > iterCount2 )
            {
                maskNow = gcapp.showImage2();
				imwrite("C:/Users/asus/Documents/Visual Studio 2012/Projects/OpenCVTemplate/x64/Debug/maskN.png", maskNow);
                cout << "Iteration times:" << iterCount2 << endl;
            }
		}
			break;
		//normal image segmentation
		case 'c':
		{	int iterCount2 = gcapp.getIterCount();
			cout << "Iteration times:" << iterCount2 << endl;
			int newIterCount2 = gcapp.nextIter2();
			if( newIterCount2 > iterCount2 )
            {
                maskNow = gcapp.showImage2();
				imwrite("C:/Users/asus/Documents/Visual Studio 2012/Projects/OpenCVTemplate/x64/Debug/maskNnotD.png", maskNow);
                cout << "Iteration times:" << iterCount2 << endl;
            }
		}
			break;
		//RGB image segmentation
        case 'n':
            int iterCount = gcapp.getIterCount();
            cout << "Iteration times:" << iterCount << endl;
            int newIterCount = gcapp.nextIter();
            if( newIterCount > iterCount )
            {
                maskNow = gcapp.showImage();
				imwrite("C:/Users/asus/Documents/Visual Studio 2012/Projects/OpenCVTemplate/x64/Debug/maskRGB.png", maskNow);
                cout << "Iteration times:" << iterCount << endl;
            }
            else
                cout << "Please draw the rectangle." << endl;
            break;
        }
    }

exit_main:
    destroyWindow( winName );
}
