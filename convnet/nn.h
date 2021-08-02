#include "datamanager.h"
/*
struct activationFunction
{
	virtual double f(double a) = 0;
	virtual double df(double a) = 0;
	virtual Eigen::VectorXd a(const Eigen::VectorXd&) = 0;
	virtual Eigen::VectorXd d(const Eigen::VectorXd&) = 0;
};

struct relu : public activationFunction
{
	double f(double a) override;
	double df(double a) override;
	Eigen::VectorXd a(const Eigen::VectorXd&) override;
	Eigen::VectorXd d(const Eigen::VectorXd&) override;
};

struct layer
{
	int size{};
	int prevSize{};
	static constexpr int squareSize{ 5 };
	Eigen::MatrixXd weights;
	Eigen::VectorXd biases;
	virtual void initializeWeights() = 0;
	virtual Eigen::VectorXd activateZ(Eigen::VectorXd& prev) = 0;
};

struct fullyConnectedLayer : public layer
{
	fullyConnectedLayer(int,int);
	void initializeWeights() override;
	Eigen::VectorXd activateZ(Eigen::VectorXd& prev) override;
};

struct poolingLayer : public layer
{
	poolingLayer(int);
	void initializeWeights() override;
	Eigen::VectorXd activateZ(Eigen::VectorXd& prev) override;
};

struct convLayer : public layer
{
	convLayer(int);
	void initializeWeights() override;
	Eigen::VectorXd activateZ(Eigen::VectorXd& prev) override;
};

class NN
{
	int numConv{};
	int layersSize{};
	int inputSize{};
	int outputSize{};
	std::vector<layer*> layers;
	relu f;
public:
	NN(std::vector<int>,int,int,int);
	~NN();
	Eigen::VectorXd vectorUnion(const Eigen::VectorXd&, const Eigen::VectorXd&);
	void feedForward(std::vector<Eigen::VectorXd>&, std::vector<Eigen::VectorXd>&);
	void backProp(const Eigen::VectorXd&, const Eigen::VectorXd&, std::vector<Eigen::MatrixXd>&, std::vector<Eigen::VectorXd>&);
	void convBackProp(const Eigen::VectorXd&, const Eigen::VectorXd&,const std::vector<Eigen::VectorXd>&, const std::vector<Eigen::VectorXd>&, std::vector<Eigen::MatrixXd>&, std::vector<Eigen::VectorXd>&);
	void updateMiniBatch(std::vector<d*>*, double, double, double);
	double evaulate(std::vector<d*>*);
	void SGD(std::vector<d*>*, std::vector<d*>*, double,double, int, int);
};
*/