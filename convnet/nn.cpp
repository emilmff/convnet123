/*
#include "nn.h"

double relu::f(double a)
{
	return a > 0 ? a : 0.0;
}

double relu::df(double a)
{
	return a > 0 ? 1.0 : 0.0;;
}

Eigen::VectorXd relu::a(const Eigen::VectorXd& z)
{
	Eigen::VectorXd a(z.rows(), z.cols());
	for (int i = 0; i < z.rows(); ++i) {
		for(int e = 0; e < z.cols(); ++e)
		{
			a(i, e) = f(z(i,e));
		}
	}
	return a;
}

Eigen::VectorXd relu::d(const Eigen::VectorXd& z)
{
	Eigen::VectorXd a(z.rows());
	for (int i = 0; i < z.rows(); ++i)
	{
		a(i) = df(z(i));
	}
	return a;
}

fullyConnectedLayer::fullyConnectedLayer(int prevsize, int s)
{
	prevSize = prevsize;
	size = s;
	initializeWeights();
}

void fullyConnectedLayer::initializeWeights()
{
	std::default_random_engine generator;
	generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
	std::normal_distribution<double> distribution(0.0, 1.0);
	biases = Eigen::VectorXd(size);
	weights = Eigen::MatrixXd(size, prevSize);

	for(int i = 0; i < size; ++i)
	{
		biases(i) = distribution(generator);
		for(int e = 0; e < prevSize; ++e)
		{
			weights(i, e) = distribution(generator) / sqrt(prevSize);
		}
	}
}

Eigen::VectorXd fullyConnectedLayer::activateZ(Eigen::VectorXd& prev)
{
	Eigen::VectorXd z = (weights * prev) + biases;
	return z;
}

poolingLayer::poolingLayer(int sampleSize)
{
	size = sampleSize;
}

void poolingLayer::initializeWeights()
{
	return;
}

Eigen::VectorXd poolingLayer::activateZ(Eigen::VectorXd& prev)
{
	Eigen::VectorXd z(prev.rows()/ 4);
	int prevSize = prev.rows() / 24;
	for(int i = 0; i < prevSize; i += size)
	{
		for(int e = 0;e < prevSize;e += size)
		{
			int l1{};
			int l2{};
			double value{};
			for(int j = 0; j < size; ++j)
			{
				for(int k = 0; k < size; ++k)
				{
					if (prev(i + j + (prevSize * (e + k))) > value) {
						value = prev(i + j + (prevSize * (e + k)));
						l1 = j;
						l2 = k;
					}
				}
			}
			//int doeswork{};
			for (int j = 0; j < size; ++j)
			{
				for (int k = 0; k < size; ++k)
				{
					if (j != l1 || k != l2) {
						//doeswork++;
						prev(i + j + (prevSize * (e + k))) = -1.0;
					}
				}
			}
			z((i/2) + (e/2) * (z.rows()/12)) = value;
		}
	}
	//std::cout << prev << std::endl << std::endl << std::endl;
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
	for(int i = 0; i < squareSize; ++i)
	{
		for(int e = 0; e < squareSize; ++e)
		{
			weights(i, e) = distribution(generator);
		}
	}
}

Eigen::VectorXd convLayer::activateZ(Eigen::VectorXd& prev)
{
	Eigen::VectorXd z(size*size);
	for(int i = 0; i < size; ++i)
	{
		for (int e = 0; e < size; ++e)
		{
			double sum{};
			for (int j = 0; j < squareSize; ++j)
			{
				for (int o = 0; o < squareSize; ++o)
				{
					sum += weights(j, o) * prev(i + j + 28*(e + o));
				}
			}
			z(i + e*size) = sum + biases(0);
		}
	}
	return z;
}

NN::NN(std::vector<int> fullLayers, int convNum,int input, int output)
{
	inputSize = input;
	outputSize = output;
	numConv = convNum;
	for(int i = 0; i < numConv; ++i) {
		layers.push_back(new convLayer(inputSize));
		++layersSize;
	}
	for(int i = 0; i < numConv; ++i) {
		layers.push_back(new poolingLayer(2));
		++layersSize;
	}
	for (int i = 0; i < fullLayers.size(); ++i) {
		if (i == 0) layers.push_back(new fullyConnectedLayer(((inputSize - 4) / 2) * ((inputSize - 4) / 2) * numConv, fullLayers.at(i)));
		else layers.push_back(new fullyConnectedLayer(layers.at(layersSize-1)->size, fullLayers.at(i)));
		++layersSize;
	}
	layers.push_back(new fullyConnectedLayer(layers.at(layersSize-1)->size, outputSize));
	++layersSize;
}

Eigen::VectorXd NN::vectorUnion(const Eigen::VectorXd& first, const Eigen::VectorXd& second)
{
	Eigen::VectorXd third(first.rows()+second.rows());
	for(int e = 0; e < first.rows(); ++e)
	{
		third(e) = first(e);
	}
	for(int e = 0; e < second.rows();++e)
	{
		third(first.rows() + e) = second(e);
	}
	return third;
}

void NN::feedForward(std::vector<Eigen::VectorXd>& zs, std::vector<Eigen::VectorXd>& as)
{
	Eigen::VectorXd z;
	int i{};
	for(int e = 0; e < numConv; ++e)
	{
		z = layers.at(i)->activateZ(as.at(0));
		zs.push_back(z);
		as.push_back(f.a(z));
		++i;
	}
	Eigen::VectorXd a;
	for(int e = 1;e < numConv+1;++e)
	{
		z = layers.at(i)->activateZ(as.at(e));
		if (e == 1) a = z;
		else a = vectorUnion(a, z);
		++i;
	}
	std::cout << a.rows() << std::endl;
	zs.push_back(a);
	as.push_back(a);
	for (int e = 0; e < layersSize - numConv * 2; ++e)
	{
		z = layers.at(i)->activateZ(a);
		a = f.a(z);
		zs.push_back(z);
		as.push_back(a);
		++i;
	}
}

void NN::backProp(const Eigen::VectorXd& input, const Eigen::VectorXd& output, std::vector<Eigen::MatrixXd>& nablaW, std::vector<Eigen::VectorXd>& nablaB)
{
	std::vector<Eigen::VectorXd> activations = { input };
	std::vector<Eigen::VectorXd> zs;
	feedForward(zs, activations);
	Eigen::VectorXd delta = (activations.back() - output);
	//std::cout <<  zs.at(3).sum() << std::endl << std::endl << std::endl << "bruh" << std::endl;
	//Eigen::VectorXd delta = (activations.back() - output).cwiseProduct(f.d(zs.back()));
	nablaB.back() += delta;
	nablaW.back() += delta * (activations.at(activations.size() - 2)).transpose();
	for (int i = 2; i < 1 + layersSize - 2 * numConv ; i++)
	{
		delta = (layers.at(layersSize - i + 1)->weights.transpose() * delta).cwiseProduct(f.d(zs.at(zs.size() - i)));
		nablaB.at(layersSize - i) += delta;
		nablaW.at(layersSize - i) += delta * activations.at(activations.size() - 1- i).transpose();
	}
	convBackProp(input, delta, activations, zs, nablaW, nablaB);
}

void NN::convBackProp(const Eigen::VectorXd& input, const Eigen::VectorXd& delta, const std::vector<Eigen::VectorXd>& activations, const std::vector<Eigen::VectorXd>& zs, std::vector<Eigen::MatrixXd>& nablaW, std::vector<Eigen::VectorXd>& nablaB)
{
	int size = layers.at(0)->size;
	for(int i = 0; i < numConv; ++i)
	{
		Eigen::VectorXd convDelta = Eigen::VectorXd::Zero(size * size);
		for (int e = 0; e < size; ++e)
		{
			for (int j = 0; j < size; ++j)
			{
				if (activations.at(i+1)(e + j * size) != -1.0)
				{
					for (int k = 0; k < delta.rows(); ++k)
					{
						convDelta(e + j * size) += delta(k) * layers.at(numConv * 2)->weights(k, (e / 2) + (size / 2) * (j / 2) + i * layers.at(numConv * 2)->weights.cols() / numConv) * f.df(zs.at(i)(e + size * j));
					}                     
				}                    
			}
		}
		nablaB.at(i)(0) += convDelta.sum();
		for (int l = 0; l < layers.at(i)->squareSize; ++l)
		{
			for (int k = 0; k < layers.at(i)->squareSize; ++k)
			{
				for(int e = 0; e < size; ++e)
				{
					for(int j = 0; j < size; ++j)
					{
						nablaW.at(i)(l, k) += convDelta(e + j * size) * input(e + l + (j + k) * inputSize);
					}
				}
			}
		}
		//std::cout << nablaW.at(0)<< std::endl << std::endl << std::endl << "bruh" << std::endl;
	}
}

void NN::updateMiniBatch(std::vector<d*>* miniBatch, double eta, double lambda, double dataSize)
{
	std::vector<Eigen::VectorXd> nablaB;
	std::vector<Eigen::MatrixXd> nablaW;
	for (int i = 0; i < layersSize;++i)
	{
		nablaB.push_back(Eigen::VectorXd::Zero(layers.at(i)->biases.rows()));
		nablaW.push_back(Eigen::MatrixXd::Zero(layers.at(i)->weights.rows(), layers.at(i)->weights.cols()));
	}
	for (auto* d : *miniBatch)
	{
		backProp(d->features, d->classes,nablaW,nablaB);
	}
	for (int i = 0; i < layersSize; ++i)
	{                                                                                                                                                              
		layers.at(i)->weights = (1.0 - eta * (lambda / dataSize)) * layers.at(i)->weights - (eta / miniBatch->size()) * nablaW.at(i);       
		//layers.at(i)->weights -= (eta/miniBatch->size()) * nablaW.at(i);
		layers.at(i)->biases -= (eta / miniBatch->size()) * nablaB.at(i);
	}
}

double NN::evaulate(std::vector<d*>* testData)
{
	double score = 0.0;
	for (auto* d : *testData)
	{
		std::vector<Eigen::VectorXd> activations;
		std::vector<Eigen::VectorXd> zs;
		activations = { d->classes };
		zs = {};
		int pos = 0;
		feedForward(zs, activations);
		Eigen::VectorXd o = activations.back();
		//std::cout << o << std::endl;
		o.maxCoeff(&pos);
		double correct = 0.0;
		d->classes.maxCoeff(&correct);
		if (pos == correct) score++;
	}
	return score;
}


void NN::SGD(std::vector<d*>* trainData, std::vector<d*>* testData, double eta,double lambda, int epochs, int batchSize)
{
	int l = trainData->size();
	int t = testData->size();

	for (int j = 0; j < epochs; j++)
	{
		auto rng = std::default_random_engine{ std::random_device{}() };
		std::shuffle(trainData->begin(), trainData->end(), rng);
		std::vector<std::vector<d*>*>* batches = new std::vector<std::vector<d*>*>;
		int index = 0;
		for (int k = 0; k < t / batchSize; k++)
		{
			std::vector<d*>* temp = new std::vector<d*>;
			for (int i = 0; i < batchSize; i++)
			{
				temp->push_back(trainData->at(index++));
			}
			batches->push_back(temp);
		}
		for (auto* b : *batches)
		{
			//std::cout << layers.at(0)->weights.sum() << std::endl << std::endl;
			updateMiniBatch(b, eta,lambda, t);
		}
		double score = evaulate(testData);
		std::cout << "performance on epoch: " << j << "-" << score / (double)t << std::endl;
		delete batches;
		eta *= 0.98;
	}
}


*/