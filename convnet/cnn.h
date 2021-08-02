#include "layer.h"

struct activationFunction
{
	virtual double f(double) = 0;
	virtual double df(double) = 0;
	virtual Eigen::VectorXd a(const Eigen::VectorXd&) = 0;
	virtual Eigen::VectorXd d(const Eigen::VectorXd&) = 0;
};

struct relu : public activationFunction
{
	double f(double) override;
	double df(double) override;
	Eigen::VectorXd a(const Eigen::VectorXd&) override;
	Eigen::VectorXd d(const Eigen::VectorXd&) override;
};

struct sigmoid : public activationFunction
{
	double f(double) override;
	double df(double) override;
	Eigen::VectorXd a(const Eigen::VectorXd&) override;
	Eigen::VectorXd d(const Eigen::VectorXd&) override;
};

class cnn
{
public:
	relu f;
	int inputSize, convSize, poolingSize;
	int convNum, fullyConnectedNum, outputSize;
	static constexpr int pool{ 2 };
	static constexpr int conv{ 5 };
	std::vector<std::vector<Eigen::VectorXd>> nablaB = { std::vector<Eigen::VectorXd>{}, std::vector<Eigen::VectorXd>{},std::vector<Eigen::VectorXd>{} };
	std::vector<std::vector<Eigen::MatrixXd>> nablaW = { std::vector<Eigen::MatrixXd>{}, std::vector<Eigen::MatrixXd>{},std::vector<Eigen::MatrixXd>{} };
	std::vector<std::vector<layer*>> layers = { std::vector<layer*>{},std::vector<layer*>{},std::vector<layer*>{},std::vector<layer*>{} };
	cnn(std::vector<int>,int, int, int,int,int);
	~cnn();
	Eigen::VectorXd vectorUnion(const Eigen::VectorXd&, const Eigen::VectorXd&);
	void fillNablas();
	std::vector<std::vector<Eigen::VectorXd>> feedForward(const Eigen::VectorXd&);
	Eigen::VectorXd softmax(Eigen::VectorXd);
	void backProp(const Eigen::VectorXd&, const Eigen::VectorXd&);
	void fullyConnectedBProp(const Eigen::VectorXd&, const Eigen::VectorXd&, const std::vector<std::vector<Eigen::VectorXd>>&);
	std::vector<Eigen::VectorXd> getConvDelta(const Eigen::VectorXd&, const std::vector<std::vector<Eigen::VectorXd>>&);
	void getConvNabla(const std::vector<Eigen::VectorXd>&,const Eigen::VectorXd&);
};

class cnnImpl
{
	cnn* network;
	double eta;
	double lambda;
	int numEpochs;
	int batchSize;
public:
	cnnImpl(cnn*,double,double,int,int);
	~cnnImpl();
	void updateMiniBatch(std::vector<d*>*,double);
	double evaulate(std::vector<d*>*);
	void SGD(std::vector<d*>*, std::vector<d*>*);
};