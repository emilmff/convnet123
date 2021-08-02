#include "cnn.h"

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
	Eigen::VectorXd a(z.rows());
	for (int i = 0; i < z.rows(); ++i)
	{
		a(i) = f(z(i));
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

cnn::cnn(std::vector<int> layerDims, int out, int convCount, int inputS, int convS, int poolingS)
{
	convNum = convCount;
	inputSize = inputS;
	convSize = convS;
	outputSize = out;
	poolingSize = poolingS;
	fullyConnectedNum = layerDims.size();
	for (int i = 0; i < layers.size(); ++i)
	{
		switch (i)
		{
		case 0:
			for (int j = 0; j < convNum; ++j)
			{
				layers.at(i).push_back(new convLayer(inputSize));
			}
			break;
		case 1:
			for (int j = 0; j < convNum; j++)
			{
				layers.at(i).push_back(new poolingLayer(convSize));
			}
			break;
		case 2:
			for (int j = 0; j < fullyConnectedNum; j++)
			{
				if (j == 0) layers.at(i).push_back(new fullyConnectedLayer(poolingSize * poolingSize * convNum, layerDims.at(j)));
				else layers.at(i).push_back(new fullyConnectedLayer(layerDims.at(j - 1), layerDims.at(j)));
			}
			break;
		case 3:
			if (fullyConnectedNum == 0) layers.at(i).push_back(new softmaxLayer(poolingSize * poolingSize * convNum, outputSize));
			else layers.at(i).push_back(new softmaxLayer(layerDims.back(), outputSize));
			break;
		}
	}
	fillNablas();
}

Eigen::VectorXd cnn::vectorUnion(const Eigen::VectorXd& first, const Eigen::VectorXd& second)
{
	Eigen::VectorXd third(first.rows() + second.rows());
	for (int e = 0; e < first.rows(); ++e)
	{
		third(e) = first(e);
	}
	for (int e = 0; e < second.rows(); ++e)
	{
		third(first.rows() + e) = second(e);
	}
	return third;
}
void cnn::fillNablas()
{
	for (int i = 0; i < convNum; ++i)
	{
		nablaB.at(0).push_back(Eigen::VectorXd::Zero(layers.at(0).at(i)->biases.rows()));
		nablaW.at(0).push_back(Eigen::MatrixXd::Zero(layers.at(0).at(i)->weights.rows(), layers.at(0).at(i)->weights.cols()));
	}
	for (int i = 0; i < fullyConnectedNum; ++i)
	{
		nablaB.at(1).push_back(Eigen::VectorXd::Zero(layers.at(2).at(i)->biases.rows()));
		nablaW.at(1).push_back(Eigen::MatrixXd::Zero(layers.at(2).at(i)->weights.rows(), layers.at(2).at(i)->weights.cols()));
	}
	nablaB.at(2).push_back(Eigen::VectorXd::Zero(layers.at(3).at(0)->biases.rows()));
	nablaW.at(2).push_back(Eigen::MatrixXd::Zero(layers.at(3).at(0)->weights.rows(), layers.at(3).at(0)->weights.cols()));
}

std::vector<std::vector<Eigen::VectorXd>> cnn::feedForward(const Eigen::VectorXd& input) 
{
	std::vector<std::vector<Eigen::VectorXd>> zs = { std::vector<Eigen::VectorXd>{}, std::vector<Eigen::VectorXd>{}, std::vector<Eigen::VectorXd>{}, std::vector<Eigen::VectorXd>{} };
	Eigen::VectorXd z;
	for (int i = 0; i < layers.size(); ++i)
	{
		switch (i)
		{
		case 0:
			for (int j = 0; j < convNum; ++j)
			{
				zs.at(i).push_back(layers.at(i).at(j)->activateZ(input));
			}
			break;
		case 1:
			z = layers.at(i).at(0)->activateZ(f.a(zs.at(i-1).at(0)));
			for (int j = 1; j < convNum; j++)
			{
				z = vectorUnion(z, layers.at(i).at(j)->activateZ(f.a(zs.at(i-1).at(j))));
			}
			zs.at(i).push_back(z);
			break;
		case 2:
			for (int j = 0; j < fullyConnectedNum; j++)
			{
				if (j == 0)
				{
					zs.at(i).push_back(layers.at(i).at(0)->activateZ(f.a(zs.at(i-1).at(0))));
				}
				else
				{
					zs.at(i).push_back(layers.at(i).at(j)->activateZ(f.a(zs.at(i).at(j - 1))));
				}
			}
			break;
		case 3:
			if (fullyConnectedNum == 0) zs.at(i).push_back(layers.at(i).at(0)->activateZ(f.a(zs.at(i - 2).at(0))));
			else zs.at(i).push_back(layers.at(i).at(0)->activateZ(f.a(zs.at(i - 1).back())));
			break;
		}
	}
	//std::cout << zs.at(2).at(1) << std::endl << std::endl << std::endl;
	return zs;
}

Eigen::VectorXd cnn::softmax(Eigen::VectorXd z)
{
	double sum{};
	for (int i = 0; i < z.rows(); ++i)
	{
		sum += exp(z(i));
	}
	Eigen::VectorXd as(z.rows());
	for (int i = 0; i < z.rows(); ++i)
	{
		as(i) = exp(z(i)) / sum;
	}
	//std::cout << sum << "\n" << "\n";
	return as;
}

void cnn::backProp(const Eigen::VectorXd& input, const Eigen::VectorXd& output)
{
	std::vector<std::vector<Eigen::VectorXd>> zs = feedForward(input);
	Eigen::VectorXd out = softmax(zs.at(3).at(0));
	//std::cout << out.sum() << std::endl << std::endl << std::endl;
	Eigen::VectorXd delta = out - output;
	nablaB.at(2).at(0) += delta;
	if (fullyConnectedNum != 0) {
		nablaW.at(2).at(0) += delta * f.a(zs.at(2).back()).transpose();
		fullyConnectedBProp(input, delta, zs);
	}
	else{
		nablaW.at(2).at(0) += delta * f.a(zs.at(1).at(0)).transpose();
		getConvNabla(getConvDelta(delta, zs), input);
	}
}     

void cnn::fullyConnectedBProp(const Eigen::VectorXd& input, const Eigen::VectorXd& delta, const std::vector<std::vector<Eigen::VectorXd>>& zs)
{
	Eigen::VectorXd d = (layers.at(3).at(0)->weights.transpose() * delta).cwiseProduct(f.d(zs.at(2).at(0)));
	nablaB.at(1).at(0) += d;
	nablaW.at(1).at(0) += d * f.a(zs.at(1).at(0)).transpose();
	//std::cout << layers.at(2).at(0)->weights.sum()<<"\n" << "\n";
	getConvNabla(getConvDelta(d, zs), input);
}

std::vector<Eigen::VectorXd> cnn::getConvDelta(const Eigen::VectorXd& delta, const std::vector<std::vector<Eigen::VectorXd>>& zs)
{
	std::vector<Eigen::VectorXd> deltas;
	for (int i = 0; i < convNum; ++i)
	{
		Eigen::VectorXd convDelta = Eigen::VectorXd::Zero(convSize*convSize);
		for (int e = 0; e < convSize; e += pool)
		{
			for (int j = 0; j < convSize; j += pool)
			{
				double max = std::max({ zs.at(0).at(i)(e + j * convSize), zs.at(0).at(i)(e + 1 + j * convSize), zs.at(0).at(i)(e + (j + 1) * convSize), zs.at(0).at(i)(e + 1 + (j + 1) * convSize) });
				for (int l = 0; l < pool; ++l)
				{
					for (int k = 0; k < pool; ++k)
					{
						if (zs.at(0).at(i)(e+l + (j+k) * convSize) == max)
						{
							for (int x = 0; x < delta.rows(); ++x)
							{
								if(fullyConnectedNum) convDelta(e + l + (j + k) * convSize) += delta(x) * layers.at(2).at(0)->weights(k, (e / 2) + (poolingSize) * (j / 2) + i * layers.at(2).at(0)->weights.cols() / convNum) * f.df(zs.at(0).at(i)(e + convSize * j));
								else convDelta(e + l + (j + k) * convSize) += delta(x) * layers.at(3).at(0)->weights(k, (e / 2) + (poolingSize) * (j / 2) + i * layers.at(3).at(0)->weights.cols() / convNum) * f.df(zs.at(0).at(i)(e + convSize * j));;
							}
							l += 2;
							k += 2;
						}
					}
				}
			}
		}
		deltas.push_back(convDelta);
		//std::cout << doeswork << std::endl << std::endl;
	}
	return deltas;
}

void cnn::getConvNabla(const std::vector<Eigen::VectorXd>& deltas, const Eigen::VectorXd& input)
{
	for (int i = 0; i < convNum; ++i)
	{
		nablaB.at(0).at(i)(0) += deltas.at(i).sum();
		for (int l = 0; l < conv; ++l)
		{
			for (int k = 0; k < conv; ++k)
			{
				for (int e = 0; e < convSize; ++e)
				{
					for (int j = 0; j < convSize; ++j)
					{
						nablaW.at(0).at(i)(l, k) += deltas.at(i)(e + j * convSize) * input(e + l + (j + k) * inputSize);
					}
				}
			}
		}
	}
	//std::cout <<deltas.at(0) << std::endl << std::endl << std::endl << std::endl << std::endl;
}

cnnImpl::cnnImpl(cnn* n, double e, double l, int epochs, int batch)
{
	network = n;
	eta = e;
	lambda = l;
	numEpochs = epochs;
	batchSize = batch;
}

void cnnImpl::updateMiniBatch(std::vector<d*>* miniBatch, double trainDataSize)
{
	for (auto* d : *miniBatch)
	{
		network->backProp(d->features, d->classes);
	}

	for (int i = 0; i < network->convNum; ++i)
	{
		network->layers.at(0).at(i)->weights = (1.0 - eta * (lambda / trainDataSize)) * network->layers.at(0).at(i)->weights - (eta / miniBatch->size()) * network->nablaW.at(0).at(i);
		network->layers.at(0).at(i)->biases -= (eta / miniBatch->size()) * network->nablaB.at(0).at(i);
	}
	for (int i = 0; i < network->fullyConnectedNum; ++i)
	{
		network->layers.at(2).at(i)->weights = (1.0 - eta * (lambda / trainDataSize)) * network->layers.at(2).at(i)->weights - (eta / miniBatch->size()) * network->nablaW.at(1).at(i);
		network->layers.at(2).at(i)->biases -= (eta / miniBatch->size()) * network->nablaB.at(1).at(i);
	}

	network->layers.at(3).at(0)->weights = (1.0 - eta * (lambda / trainDataSize)) * network->layers.at(3).at(0)->weights - (eta / miniBatch->size()) * network->nablaW.at(2).at(0);
	network->layers.at(3).at(0)->biases -= (eta / miniBatch->size()) * network->nablaB.at(2).at(0);

	for (int i = 0; i < network->nablaB.size(); ++i)
	{
		network->nablaW.at(i).clear();
		network->nablaB.at(i).clear();
	}
	network->fillNablas();
}

double cnnImpl::evaulate(std::vector<d*>* testData)
{
	double score = 0.0;
	for (auto* d : *testData)
	{
		int pos = 0;
		Eigen::VectorXd o = network->softmax(network->feedForward(d->features).back().back());
		//std::cout << o.sum() << std::endl << std::endl;
		//std::cout << d->classes << std::endl << std::endl << std::endl;
		o.maxCoeff(&pos);
		double correct = 0.0;
		d->classes.maxCoeff(&correct);
		if (pos == correct) score++;
	}
	return score;
}

void cnnImpl::SGD(std::vector<d*>* trainData, std::vector<d*>* testData)
{
	int l = trainData->size();
	int t = testData->size();
	for (int j = 0; j < numEpochs; j++)
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
			//std::cout << network->layers.at(3).at(0)->weights.sum() << std::endl << std::endl << std::endl;
			updateMiniBatch(b, (double)t);
		}
		double score = evaulate(testData);
		std::cout << "performance on epoch: " << j << "-" << score / (double)t << std::endl;
		delete batches;
		eta *= 0.98;
	}
}

int main()
{
	data* t = new data;
	cnn* n = new cnn({100},10, 3, 28, 24,12);
	cnnImpl* network = new cnnImpl(n, 0.03, 0.1, 60, 1);
	network->SGD(t->getTrainingData(), t->getTestData());
	return 0;
};