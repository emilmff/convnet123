#include <Eigen/Eigen> 
#include <random>
#include "stdint.h"
#include <chrono>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>

struct d
{
	Eigen::VectorXd features = Eigen::VectorXd::Zero(784);
	Eigen::VectorXd classes = Eigen::VectorXd::Zero(10);
};

class data
{
	std::vector<d*>* trainingData;
	std::vector<d*>* testData;
public:
	data();
	void readInputData(std::string, bool);
	void readLabelData(std::string, bool);
	void normalizeTrain();
	void normalizeTest();
	uint32_t format(const unsigned char* bytes);
	std::vector<d*>* getTrainingData();
	std::vector<d*>*getTestData();
};
