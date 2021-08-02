#include "nn.h"
#pragma warning(disable : 4996)


data::data()
{
    testData = new std::vector<d*>;
    trainingData = new std::vector<d*>;
	readInputData("train-images.idx3-ubyte", true);
	readLabelData("train-labels.idx1-ubyte", true);
    readInputData("t10k-images.idx3-ubyte", false);
    readLabelData("t10k-labels.idx1-ubyte", false);
    normalizeTrain();
    normalizeTest();
}

void data::readInputData(std::string dir, bool train)
{
    uint32_t magic = 0;
    uint32_t numImages = 0;
    uint32_t numRows = 0;
    uint32_t numCols = 0;

    unsigned char bytes[4];
    FILE* f = fopen(dir.c_str(), "rb");
    if (f)
    {
        int i = 0;
        while (i < 4)
        {
            if (fread(bytes, sizeof(bytes), 1, f))
            {
                switch (i)
                {
                case 0:
                    magic = format(bytes);
                    i++;
                    break;
                case 1:
                    numImages = format(bytes);
                    i++;
                    break;
                case 2:
                    numRows = format(bytes);
                    i++;
                    break;
                case 3:
                    numCols = format(bytes);
                    i++;
                    break;
                }
            }
        }
        std::cout << "done getting file header" << std::endl;
        for (i = 0; i < numImages; i++)
        {
            d* u = new d;
            uint8_t element[1];
            for (int j = 0; j < numRows*numCols; j++)
            {
                if (fread(element, sizeof(element), 1, f))
                {
                    u->features(j) = (element[0]);
                }
                else std::cout << "shits fucked";
            }
            if (train)trainingData->push_back(u);
            else testData->push_back(u);
        }
        std::cout << "read succesfully" << std::endl;
    }
    else
    {
        std::cout << "wrong filepath";
        exit(1);
    }
}

void data::readLabelData(std::string path, bool train)
{
    uint32_t magic = 0;
    uint32_t numImages = 0;
    unsigned char bytes[4];
    FILE* f = fopen(path.c_str(), "r");
    if (f)
    {
        int i = 0;
        while (i < 2)
        {
            if (fread(bytes, sizeof(bytes), 1, f))
            {
                switch (i)
                {
                case 0:
                    magic = format(bytes);
                    i++;
                    break;
                case 1:
                    numImages = format(bytes);
                    i++;
                    break;
                }
            }
        }

        for (unsigned j = 0; j < numImages; j++)
        {
            uint8_t element[1];
            if (fread(element, sizeof(element), 1, f))
            {
                if(train)trainingData->at(j)->classes(element[0]) = 1.0;
                else testData->at(j)->classes(element[0]) = 1.0;
            }
        }

        std::cout << "read labels" << std::endl;
    }
    else
    {
        std::cout << "wrong file path";
        exit(1);
    }
}

uint32_t data::format(const unsigned char* bytes)
{
    return (uint32_t)((bytes[0] << 24) |
                      (bytes[1] << 16) |
                      (bytes[2] << 8) |
                      (bytes[3]));
}

void data::normalizeTrain()
{
    std::vector<double> mins, maxes;
    for (int i = 0; i < trainingData->at(0)->features.rows();i++)
    {
        mins.push_back(trainingData->at(0)->features(i));
        maxes.push_back(trainingData->at(0)->features(i));
    }

    int size = trainingData->size();

    for (int i = 1; i < size; i++)
    {
        d* unit = trainingData->at(i);
        for (int j = 0; j < unit->features.rows(); j++)
        {
            double value = unit->features(j);
            if (value < mins.at(j)) mins.at(j) = value;
            if (value > maxes.at(j)) maxes.at(j) = value;
        }
    }
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < trainingData->at(0)->features.rows(); j++)
        {
            if (maxes.at(j) - mins.at(j) == 0.0)
            {
                trainingData->at(i)->features(j) = 0.0;
            }
            else
            {
                trainingData->at(i)->features(j) = (double)(trainingData->at(i)->features(j) - mins.at(j)) / (maxes.at(j) - mins.at(j));
            }
        }
    }
}

void data::normalizeTest()
{
    std::vector<double> mins, maxes;
    for (int i = 0; i < trainingData->at(0)->features.rows(); i++)
    {
        mins.push_back(testData->at(0)->features(i));
        maxes.push_back(testData->at(0)->features(i));
    }

    int size = testData->size();

    for (int i = 1; i < size; i++)
    {
        d* unit = testData->at(i);
        for (int j = 0; j < unit->features.rows(); j++)
        {
            double value = (double)unit->features(j);
            if (value < mins.at(j)) mins.at(j) = value;
            if (value > maxes.at(j)) maxes.at(j) = value;
        }
    }
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < testData->at(0)->features.rows(); j++)
        {
            if (maxes.at(j) - mins.at(j) == 0)
            {
                testData->at(i)->features(j) = 0.0;
            }
            else
            {
                testData->at(i)->features(j) = (double)(testData->at(i)->features(j) - mins.at(j)) / (maxes.at(j) - mins.at(j));
            }
        }
    }
}

std::vector<d*>* data::getTrainingData()
{
    return trainingData;
}
std::vector<d*>* data::getTestData()
{
    return testData;
}