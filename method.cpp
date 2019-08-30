#include <GenICam\System.h>
#include<GenICam\Camera.h>
#include<GenICam\GigE\GigECamera.h>
#include<GenICam\GigE\GigEInterface.h>
#include<Infra\PrintLog.h>
#include<Memory\SharedPtr.h>

#include<opencv2\opencv.hpp>
#include<vector>
#include<map>
#include<utility>
#include<heapapi.h>

using namespace Dahua::GenICam;
using namespace Dahua::Infra;


int minArea=10,maxArea = 150;
int thresholdNum = 100;
void sortPointGroup(std::vector<cv::Point> input, std::vector<cv::Point> &output);
void getPointGroups(std::vector<cv::Point>input, std::vector<std::vector<cv::Point>> &output, int minDistance = 10, int maxDistance = 150);
int getMaxPosition(std::vector<std::pair<int, double>> input);
int getMaxPair(std::vector<std::pair<int, double>> input, std::pair<int, double> &output);
void recognizeProcess(cv::Mat input, cv::Mat &output);
bool isApproxed(double a1, double a2);
bool getFitIndex(std::vector<cv::Point> input, std::vector<int> &output);
bool isTwoSequenceSimilar(std::vector<double> input1, std::vector<double> input2, int &_translateDistance);

double A2d = 3.1415926 /180;
//double degreeSequenceIndex[6][4] = 
//				{{60*A2d,90 * A2d,120 * A2d,90 * A2d },
//				{90 * A2d,150 * A2d,0 * A2d,120 * A2d },
//				{120 * A2d,120 * A2d,60 * A2d,180 * A2d },
//				{90 * A2d,120 * A2d,0 * A2d,150 * A2d },
//				{150 * A2d,150 * A2d,120 * A2d,180 * A2d },
//				{60 * A2d,60 * A2d,120 * A2d,120 * A2d }};


std::vector<std::vector<double>> degreeSequenceIndex =
{ { cos(60 * A2d),cos(90 * A2d),cos(120 * A2d),cos(90 * A2d) },
{ cos(90 * A2d),cos(150 * A2d),cos(0 * A2d),cos(120 * A2d) },
{ cos(120 * A2d),cos(120 * A2d),cos(60 * A2d),cos(180 * A2d) },
{ cos(90 * A2d),cos(120 * A2d),cos(0 * A2d),cos(150 * A2d) },
{ cos(150 * A2d),cos(150 * A2d),cos(120 * A2d),cos(180 * A2d) },
{ cos(60 * A2d),cos(60 * A2d),cos(120 * A2d),cos(120 * A2d) } };

std::vector<std::vector<int>> pointIndex = 
{
	{1,2,3,4},
	{5,6,7,8},
	{8,5,7,6},
	{9,10,11,12},
	{12,9,11,10},
	{15,16,13,14}
};
 

std::vector<std::vector<double>>sixPointDegreeSequenceIndex = 
{
	{cos(60 * A2d),cos(120 * A2d),cos(60 * A2d),cos(120 * A2d),cos(60 * A2d),cos(60 * A2d)},
	{cos(60 * A2d),cos(90 * A2d),cos(150 * A2d),cos(120 * A2d),cos(120 * A2d),cos(60 * A2d)}
};

std::vector<std::vector<int>> sixPointIndex=
{
	{17,18,19,20,21,22},
	{17,18,20,19,21,22}
};

std::vector<cv::Point3f> worldPoints = 
{
	cv::Point3f(-12,4,0),cv::Point3f(-11,3,0),cv::Point3f(-8,4,0),cv::Point3f(-11,5,0),
	cv::Point3f(-11,-5,0),cv::Point3f(-8,-4,0),cv::Point3f(-10,-4,0),cv::Point3f(-12,-4,0),
	cv::Point3f(11,3,0),cv::Point3f(12,4,0),cv::Point3f(10,4,0),cv::Point3f(8,4,0),
	cv::Point3f(9,-5,0),cv::Point3f(11,-3,0),cv::Point3f(9,-3,0),cv::Point3f(8,-4,0),
	cv::Point3f(-1,-1,0),cv::Point3f(1,-1,0),cv::Point3f(0,0,0),cv::Point3f(1,1,0),cv::Point3f(-1,1,0),cv::Point3f(-2,0,0)
};

int main()
{

	cv::VideoCapture capture;
	if (!capture.open(0))
	{
		return 0;
	}
	cv::Mat frame;
	cv::Mat Image = cv::imread("Image7.bmp");
	cv::Mat result;
	
	
	//recognizeProcess(Image, result);
	while (true)
	{
		capture >> frame;
		/*cv::imshow("frame", frame);*/
		recognizeProcess(frame, result);
		cv::imshow("origin", frame);
		//cv::imshow("gaussianImage", gaussianImage);
		cv::imshow("recognize", result);
		if (cv::waitKey(30) > 0)
			break;
	}


	return 0;
}


void recognizeProcess(cv::Mat input, cv::Mat &output)
{
	cv::Mat Image = input.clone();
	cv::Mat gaussianImage;
	cv::Mat threeChanels[3];
	cv::Mat binaryFrame;
	
	cv::GaussianBlur(Image, gaussianImage, cv::Size(3, 3), 2.0, 2.0);
	cv::split(gaussianImage, threeChanels);
	cv::threshold(threeChanels[0], binaryFrame, thresholdNum, 255, CV_THRESH_BINARY_INV);
	
	cv::imshow("binary frame", binaryFrame);
	/************************/
	/*寻找有中心黑点的圆*/
	cv::Mat labels, stats, centroid;
	int nccomps = cv::connectedComponentsWithStats(binaryFrame, labels, stats, centroid);

	int center = 0;
	std::vector<cv::Point> allCenterPoint;
	allCenterPoint.clear();
	for (int i = 0; i < nccomps; i++)
	{
		int area = stats.at<int>(i, cv::CC_STAT_AREA);

		if (area < maxArea && area >minArea)
		{
			int cx = stats.at<int>(i, cv::CC_STAT_LEFT) + stats.at<int>(i, cv::CC_STAT_WIDTH) / 2;
			int cy = stats.at<int>(i, cv::CC_STAT_TOP) + stats.at<int>(i, cv::CC_STAT_HEIGHT) / 2;
			allCenterPoint.push_back(cv::Point(cx, cy));
			center++;
		}
	}

	printf("all center point num is %d.\n", allCenterPoint.size());
	/*************************/
	/*将所有检测的的中心点进行分组*/
	std::vector<std::vector<cv::Point>> centerPointGroups;
	getPointGroups(allCenterPoint, centerPointGroups);

	/*删除非4或6的组*/
	std::vector<std::vector<cv::Point>>::iterator it;
	for (it = centerPointGroups.begin(); it != centerPointGroups.end();)
	{
		//printf("iterator\n");
		if (it->size() != 4 && it->size() != 6)
		{
			it = centerPointGroups.erase(it);
		}
		else
			it++;
	}

	/*顺时针存储点*/
	std::vector<std::vector<cv::Point>> sortedGroups;
	for (int i = 0; i < centerPointGroups.size(); i++)
	{
		std::vector<cv::Point> sortedGroup;
		sortPointGroup(centerPointGroups[i], sortedGroup);
		sortedGroups.push_back(sortedGroup);
		//printf("sortedGroup size is %d \n", sortedGroup.size());
	}

	//std::vector<cv::Point> sortedGroup;
	//sortPointGroup(centerPointGroups[2], sortedGroup);

	///*printf("cos222 value list size is %d \n", sortedGroup.size());*/
	//std::vector<std::vector<cv::Point>> groupTemp;
	//groupTemp.push_back(sortedGroup);
	//cv::drawContours(Image, groupTemp, -1, (255), 5);
	
	cv::drawContours(Image, sortedGroups, -1, (255), 5);
	/*output = Image;*/
	std::vector<std::vector<int>> groupsIndex;
	for (int i = 0; i < sortedGroups.size();)
	{
		//printf("\n\n");
		std::vector<int> indexOutput;
		auto fited = getFitIndex(sortedGroups[i], indexOutput);
		if (fited)
		{
			groupsIndex.push_back(indexOutput);
			i++;
			//printf("fited\n");
			/*printf("index is :");
			for (int i = 0; i < indexOutput.size(); i++)
			{
				printf("%d", indexOutput[i]);
			}*/
		}
		else
		{
			std::vector<std::vector<cv::Point>>::iterator it = sortedGroups.begin()+i;
			sortedGroups.erase(it);
		}
	}


	cv::drawContours(Image, sortedGroups, -1, cv::Scalar(0,0,255), 5);
	output = Image;
	if (sortedGroups.size() == 0)
	{
		return;
	}
	/*生成用来计算相机参数的两个序列*/
	std::vector<cv::Point2f> image_points;
	std::vector<cv::Point3f> object_points;
	for (int i = 0; i < sortedGroups.size(); i++) 
	{
		for (int j = 0; j < sortedGroups[i].size(); j++)
		{
			image_points.push_back(sortedGroups[i][j]);
			
		}
	}
	/*printf("image points are: \n");
	for (int i = 0; i < image_points.size(); i++)
	{
		printf(" %f, %f, \t", image_points[i].x, image_points[i].y);
	}*/
	for (int i = 0; i < groupsIndex.size(); i++)
	{
		for (int j = 0; j < groupsIndex[i].size(); j++)
		{
			object_points.push_back(worldPoints[(groupsIndex[i][j]) - 1]);
		}
	}

	//printf("row is %d, col is %d", Image.rows, Image.cols);
	std::vector<std::vector<cv::Point2f>> image_points_seq;
	std::vector<std::vector<cv::Point3f>> object_points_seq;
	image_points_seq.push_back(image_points);
	object_points_seq.push_back(object_points);
	cv::Mat cameraMatrix;
	cv::Mat distCoeffs;
	std::vector<cv::Mat> rvescsMat;
	std::vector<cv::Mat> tvecsMat;
	
	double err_first = cv::calibrateCamera(object_points_seq, image_points_seq, cv::Size(Image.cols,Image.rows), cameraMatrix, distCoeffs, rvescsMat, tvecsMat,CV_CALIB_FIX_K3);
	//printf("误差：%f 像素\n ", err_first);

	std::vector<cv::Point2f> image_points_output;
	cv::projectPoints(worldPoints, rvescsMat[0], tvecsMat[0], cameraMatrix, distCoeffs, image_points_output);
	
	for (int i = 0; i < image_points_output.size(); i++)
	{
		cv::circle(Image, image_points_output[i], 2, cv::Scalar(0, 255, 0), 2);
	}
	
	//cv::circle(Image, image_points_output[i], 2, cv::Scalar(0, 255, 0), 2);
	output = Image;
}


bool getFitIndex(std::vector<cv::Point> input, std::vector<int> &output)
{
	if (input.size() != 4 && input.size() != 6)
	{
		printf("only accept size 4 or 6 group");
		return false;
	}

	auto currentGroup = input;
	std::vector<int> index;
	std::vector<double> currentGroupCosValue;
	//printf("current groupSize is %d\n", currentGroup.size());
	for (int j = 0; j < currentGroup.size(); j++)
	{
		
		auto vec1 = currentGroup[j] - currentGroup[(j - 1+ currentGroup.size()) % currentGroup.size()];
		auto vec2 = currentGroup[(j + 1) % currentGroup.size()] - currentGroup[j];
		auto cosValue = vec1.dot(vec2) / (sqrt(vec1.dot(vec1))  * sqrt(vec2.dot(vec2)));
		currentGroupCosValue.push_back(cosValue);
		//if (j == 0)
		//{
		//	printf("vec1 is %d,%d\n", vec1.x, vec1.y);
		//	printf("vec2 is %d,%d\n", vec2.x, vec2.y);
		//	/*int temp2 = ;*/
		//	int temp = (j-1+ currentGroup.size()) % currentGroup.size();
		//	printf("ppppppp %d\n", temp);
		//	printf("sdsdsdsd %d, %d\n", currentGroup[5].x, currentGroup[5].y);
		//}
	}
	
	if (currentGroup.size() == 4)
	{
		int translateDistance;
		int sequnceIndex;
		/*for (int i = 0; i < currentGroupCosValue.size(); i++)
		{
			printf("cosValue is %f", currentGroupCosValue[i]);
		}*/
		bool isFouded = false;
		for (int i = 0; i < degreeSequenceIndex.size(); i++)
		{
			if (isTwoSequenceSimilar(currentGroupCosValue, degreeSequenceIndex[i], translateDistance))
			{
				sequnceIndex = i;
				isFouded = true;
				break;
			}
		}

		
		if (!isFouded)
		{
			return false;
		}
		else
		{
			//printf("sucess at %d, %d.\n", sequnceIndex,translateDistance);
			std::vector<int> indexOfInputPoint(4);
			for (int i = 0; i < currentGroup.size(); i++)
			{
				indexOfInputPoint[(i+translateDistance) % currentGroup.size()] = pointIndex[sequnceIndex][i];
			}
			output = indexOfInputPoint;
			return true;
		}

	}
	else
	{
		int translateDistance;
		int sequnceIndex;
		bool isFouded = false;
		/*for (int i = 0; i < currentGroupCosValue.size(); i++)
		{
			printf("cosValue is %f", currentGroupCosValue[i]);
		}
		printf("\n");*/
		for (int i = 0; i < sixPointDegreeSequenceIndex.size(); i++)
		{
			if (isTwoSequenceSimilar(currentGroupCosValue, sixPointDegreeSequenceIndex[i], translateDistance))
			{
				sequnceIndex = i;
				isFouded = true;
				break;
			}
		}


		if (!isFouded)
		{
			return false;
		}
		else
		{
			//printf("sucess at %d, %d.\n", sequnceIndex, translateDistance);
			std::vector<int> indexOfInputPoint(6);
			for (int i = 0; i < currentGroup.size(); i++)
			{
				indexOfInputPoint[(i + translateDistance) % currentGroup.size()] = sixPointIndex[sequnceIndex][i];
			}
			output = indexOfInputPoint;
			return true;
		}
	}

}


bool isTwoSequenceSimilar(std::vector<double> input1, std::vector<double> input2, int &_translateDistance)
{
	int output=0;
	if (input1.size() != input2.size())
	{
		printf("Two squence don't have the same size.\n");
		return false;
	}

	bool isSimilar = false;
	for (int i = 0; i < input1.size(); i++)
	{
		/*printf("i is %d\n", i);*/
		if (!isApproxed(input1[i], input2[0]))
		{
			continue;
		}
		else
		{
		/*	printf("arrived here\n");*/
			isSimilar = true;
			for (int j = 1; j < input2.size(); j++)
			{
				if (!isApproxed(input1[(i + j) % input1.size()], input2[j]))
				{
					isSimilar = false;
					break;
				}
			}
		}

		if (isSimilar)
		{
			
			output = i;
			_translateDistance = output;
			return true;
			//break;
		}
	}

	return isSimilar;
}

bool isApproxed(double a1, double a2)
{
	return abs(a1 - a2) < 0.15;
}

void mainxxxx()
{
	cv::VideoCapture capture;
	capture.open(0);

	cv::Mat frame;
	cv::Mat grayFrme;
	cv::Mat threeChanels[3];
	cv::Mat binaryFrame;

	cv::Mat Image = cv::imread("Image4.bmp");

	std::vector<std::vector<cv::Point>> allContours;
	std::vector<std::vector<cv::Point>> validContours;
	//int minPoint = 2000, maxPoint = 4000;
	int minDistance = 0, maxDistance = 100;
	
	cv::Mat gaussianImage;

	
	cv::split(Image, threeChanels);
	//cv::GaussianBlur(Image, gaussianImage, cv::Size(3, 3), 2.0,2.0);

	cv::threshold(threeChanels[0], binaryFrame, 60, 255, CV_THRESH_BINARY_INV);

	/************************/
	/*寻找有中心黑点的圆*/
	cv::Mat labels, stats, centroid;
	int nccomps = cv::connectedComponentsWithStats(binaryFrame, labels, stats, centroid);

	int center = 0;
	std::vector<cv::Point> allCenterPoint;
	allCenterPoint.clear();
	for (int i = 0; i < nccomps; i++)
	{
		int area = stats.at<int>(i, cv::CC_STAT_AREA);
		
		if (area < 100 && area >10)
		{ 
			int cx = stats.at<int>(i, cv::CC_STAT_LEFT) + stats.at<int>(i, cv::CC_STAT_WIDTH)/2;
			int cy = stats.at<int>(i, cv::CC_STAT_TOP) + stats.at<int>(i, cv::CC_STAT_HEIGHT)/2;
			allCenterPoint.push_back(cv::Point(cx, cy));
			center++;
		}
	}

	/*************************/
	/*将所有检测的的中心点进行分组*/
	std::vector<std::vector<cv::Point>> centerPointGroups;
	getPointGroups(allCenterPoint, centerPointGroups);

	/*删除非4或6的组*/
	std::vector<std::vector<cv::Point>>::iterator it;
	for (it= centerPointGroups.begin(); it!= centerPointGroups.end();)
	{
		printf("iterator\n");
		if (it->size() != 4 && it->size() != 6)
		{
			it = centerPointGroups.erase(it);
		}
		else
			it++;
	}

	/*顺时针存储点*/
	std::vector<std::vector<cv::Point>> sortedGroups;
	for (int i = 0; i < centerPointGroups.size(); i++)
	{
		std::vector<cv::Point> sortedGroup;
		sortPointGroup(centerPointGroups[i], sortedGroup);
		sortedGroups.push_back(sortedGroup);
		printf("sortedGroup size is %d \n",sortedGroup.size());
	}
	
	/*std::vector<cv::Point> sortedGroup;
	sortPointGroup(centerPointGroups[2], sortedGroup);*/

	/*printf("cos222 value list size is %d \n", sortedGroup.size());*/
	/*std::vector<std::vector<cv::Point>> groupTemp;
	groupTemp.push_back(sortedGroup);*/
	cv::drawContours(Image, sortedGroups, -1, (255), 5);
	/*cv::drawContours(Image,validContours,)*/
	while (true)
	{
		//capture >> frame;
		/*cv::imshow("frame", frame);*/


		cv::imshow("single chanel", threeChanels[0]);
		//cv::imshow("gaussianImage", gaussianImage);
		cv::imshow("binary frame", binaryFrame);
		cv::imshow("contours", Image);
		if (cv::waitKey(30) > 0)
			break;
	}
}


void sortPointGroup(std::vector<cv::Point> input , std::vector<cv::Point> &output)
{
	/*对每一组里的点进行逆时针排序*/
	
	cv::Point leftBottomPoint = input[0];
	int leftBottomPosition = 0;
	/*找到左下角点*/
	for (int j = 1; j < input.size(); j++)
	{
		if (input[j].y > leftBottomPoint.y)
		{
			leftBottomPoint = input[j];
			leftBottomPosition = j;
		}
		else if (input[j].y == leftBottomPoint.y)
		{
			if (input[j].x < leftBottomPoint.x)
			{
				leftBottomPoint = input[j];
				leftBottomPosition = j;
			}
		}
	}

	//printf("left bottom point is %d, %d.\n", leftBottomPoint.x, leftBottomPoint.y);
	/*根据点乘结果来排序*/
	std::vector<std::pair<int,double>> cosValueList;
	for (int j = 0; j < input.size(); j++)
	{
		if (j == leftBottomPosition)
		{
			cosValueList.push_back(std::pair<int, double>(j,10));
			continue;
		}

		cv::Point vecPoint = cv::Point((input[j] - leftBottomPoint).x, (leftBottomPoint.y - input[j].y));
		//printf("vecPoint value is %d, %d \n", vecPoint.x,vecPoint.y);
		auto cosValue = vecPoint.dot(cv::Point(1, 0)) / sqrt(vecPoint.dot(vecPoint));
		cosValueList.push_back(std::pair<int, double>(j,cosValue));
		//printf("cos value is %f \n", cosValue);
	}

	//printf("cos value list size is %d \n", cosValueList.size());

	std::vector<cv::Point> sortedGroup;
	//std::pair<int, double> previousMaxPair;
	//int maxPosition = getMaxPair(cosValueList, previousMaxPair);
	////printf("maxPair is %d, %f", maxPair.first, maxPair.second);
	//std::vector<std::pair<int, double>>::iterator it = cosValueList.begin() + maxPosition;
	//cosValueList.erase(it);
	while (!cosValueList.empty())
	{

		//printf("cos value list is");
		/*for (int i = 0; i < cosValueList.size(); i++)
		{
			printf("%f,", cosValueList[i].second);
		}*/
		//printf("\n");
		std::pair<int, double> maxPair;
		int maxPosition = getMaxPair(cosValueList, maxPair);

		//printf("maxPair is %d, %f", maxPair.first, maxPair.second);
		std::vector<std::pair<int, double>>::iterator it = cosValueList.begin() + maxPosition;
		cosValueList.erase(it);
		
		/*if (abs(maxPair.second - previousMaxPair.second) > 0.1)
		{
			sortedGroup.push_back(input[previousMaxPair.first]);
			previousMaxPair = maxPair;
			printf("push previous \n");
		}
		else
		{
			cv::Point vecPoint = cv::Point((input[maxPair.first] - sortedGroup.back()).x, (sortedGroup.back().y - input[maxPair.first].y));
			auto distance = (vecPoint.dot(vecPoint));

			cv::Point vecPreviousPoint = cv::Point((input[previousMaxPair.first] - sortedGroup.back()).x, (sortedGroup.back().y - input[previousMaxPair.first].y));
			auto distancePrevious = (vecPreviousPoint.dot(vecPreviousPoint));
			
			if (distance < distancePrevious)
			{
				sortedGroup.push_back(input[maxPair.first]);
			}
			else
			{
				sortedGroup.push_back(input[previousMaxPair.first]);
				previousMaxPair = maxPair;
			}

		}*/
		sortedGroup.push_back(input[maxPair.first]);
		//printf("push point  %d, %d \n", input[maxPair.first].x, input[maxPair.first].y);
	}

	//sortedGroup.push_back(input[previousMaxPair.first]);

	output = sortedGroup;
	//printf("sortedGroup size is %d \n", sortedGroup.size());
	//printf("output size is %d \n", output.size());
}

int getMaxPosition(std::vector<std::pair<int, double>> input)
{
	int maxPosition = input[0].first;
	double maxValue = input[0].second;
	for (int i = 1; i < input.size(); i++)
	{
		if (input[i].second > maxValue)
		{
			maxValue = input[i].second;
			maxPosition = input[i].first;
		}
	}

	return maxPosition;
}

int getMaxPair(std::vector<std::pair<int, double>> input, std::pair<int, double> &output)
{
	int maxPosition = 0;
	std::pair<int, double> maxPair = input[0];
	for (int i = 1; i < input.size(); i++)
	{
		if (input[i].second > maxPair.second)
		{
			maxPair = input[i];
			maxPosition = i;
		}
	}

	output = maxPair;
	return maxPosition;
}

void getPointGroups(std::vector<cv::Point>allCenterPoint, std::vector<std::vector<cv::Point>> &output,int minDistance,int maxDistance)
{
	std::vector<std::vector<cv::Point>> centerPointGroups;
	for (int i = 0; i < allCenterPoint.size(); i++)
	{
		auto currentPoint = allCenterPoint[i];
		bool labeled = false;

		for (int j = 0; j < centerPointGroups.size(); j++)
		{
			auto currentGroup = centerPointGroups[j];
			for (int k = 0; k < currentGroup.size(); k++)
			{
				auto vec = (currentPoint - currentGroup[k]);
				int distance = sqrtf(vec.dot(vec));
				if (distance > minDistance && distance < maxDistance)
				{
					centerPointGroups[j].push_back(currentPoint);
					labeled = true;
					break;
				}
			}

			if (labeled)
			{
				break;
			}
		}

		if (!labeled)
		{
			std::vector<cv::Point> newGroup;
			newGroup.push_back(currentPoint);
			centerPointGroups.push_back(newGroup);
		}
	}

	/*printf("center : %d \n", center);
	printf("group num: %d.\n", centerPointGroups.size());*/
	/*for (int i = 0; i < centerPointGroups.size(); i++)
	{
		printf("group %d has %d points.\n", i, centerPointGroups[i].size());
	}
*/
	output = centerPointGroups;
}