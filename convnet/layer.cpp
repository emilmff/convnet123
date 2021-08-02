#include "layer.h"

softmaxLayer::softmaxLayer(int prevsize, int s)
{
	prevSize = prevsize;
	size = s;
	initializeWeights();
}

void softmaxLayer::initializeWeights()
{
	std::default_random_engine generator;
	generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
	std::normal_distribution<double> distribution(0.0, 1.0);
	biases = Eigen::VectorXd(size);
	weights = Eigen::MatrixXd(size, prevSize);

	for (int i = 0; i < size; ++i)
	{
		biases(i) = distribution(generator);
		for (int e = 0; e < prevSize; ++e)
		{
			weights(i, e) = distribution(generator);
		}
	}
}

Eigen::VectorXd softmaxLayer::activateZ(const Eigen::VectorXd& prev)
{
	return (weights * prev) + biases;
}


fullyConnectedLayer::fullyConnectedLayer(int prevsize, int s)
{
	prevSize = prevsize;
	size = s;
	initializeWeights();
}

void fullyConnectedLayer::initializeWeights()
{
	generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
	std::normal_distribution<double> distribution(0.0, 1.0);
	biases = Eigen::VectorXd(size);
	weights = Eigen::MatrixXd(size, prevSize);

	for (int i = 0; i < size; ++i)
	{
		biases(i) = distribution(generator);
		for (int e = 0; e < prevSize; ++e)
		{
			weights(i, e) =  distribution(generator);
		}
	}
}

Eigen::VectorXd fullyConnectedLayer::activateZ(const Eigen::VectorXd& prev)
{
	return (weights * prev) + biases;
}

poolingLayer::poolingLayer(int prevSize)
{
	size = prevSize;
}

void poolingLayer::initializeWeights()
{
	return;
}

Eigen::VectorXd poolingLayer::activateZ(const Eigen::VectorXd& prev)
{
	int currSize = (size*size) / (poolSize * poolSize);
	Eigen::VectorXd z(currSize);
	for (int i = 0; i < size; i += poolSize)
	{
		for (int e = 0; e < size; e += poolSize)
		{
			z(i / 2 + (e / 2) * (size / poolSize)) = std::max({prev(i + e * size), prev(i + 1 + e * size),prev(i + (e + 1) * size), prev(i + 1 + (e + 1) * size) });
		}
	}
	return z;
}

convLayer::convLayer(int prevsize)
{
	size = prevsize - squareSize + 1;
	initializeWeights();
}

void convLayer::initializeWeights()
{
	std::default_random_engine generator;
	generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
	std::normal_distribution<double> distribution(0.0, 1.0);
	weights = Eigen::MatrixXd(squareSize, squareSize);
	biases = Eigen::VectorXd(1);
	biases(0) = distribution(generator);
	for (int i = 0; i < squareSize; ++i)
	{
		for (int e = 0; e < squareSize; ++e)
		{
			weights(i, e) = distribution(generator);
		}
	}
}

Eigen::VectorXd convLayer::activateZ(const Eigen::VectorXd& prev)
{
	Eigen::VectorXd z(size * size);
	for (int i = 0; i < size; ++i)
	{
		for (int e = 0; e < size; ++e)
		{
			double sum{};
			for (int j = 0; j < squareSize; ++j)
			{
				for (int o = 0; o < squareSize; ++o)
				{
					sum += weights(j, o) * prev(i + j + (size+squareSize-1) * (e + o));
				}
			}
			z(i + e * size) = sum + biases(0);
		}
	}
	return z;
}
