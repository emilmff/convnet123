#include "datamanager.h"

struct layer
{
	int size{};
	int prevSize{};
	static constexpr int squareSize{ 5 };
	static constexpr int poolSize{ 2 };
	std::default_random_engine generator;
	Eigen::MatrixXd weights;
	Eigen::VectorXd biases;
	virtual void initializeWeights() = 0;
	virtual Eigen::VectorXd activateZ(const Eigen::VectorXd& prev) = 0;
};

struct softmaxLayer : public layer
{
	softmaxLayer(int, int);
	void initializeWeights() override;
	Eigen::VectorXd activateZ(const Eigen::VectorXd& prev) override;
};


struct fullyConnectedLayer : public layer
{
	fullyConnectedLayer(int, int);
	void initializeWeights() override;
	Eigen::VectorXd activateZ(const Eigen::VectorXd& prev) override;
};

struct poolingLayer : public layer
{
	poolingLayer(int);
	void initializeWeights() override;
	Eigen::VectorXd activateZ(const Eigen::VectorXd& prev) override;
};

struct convLayer : public layer
{
	convLayer(int);
	void initializeWeights() override;
	Eigen::VectorXd activateZ(const Eigen::VectorXd& prev) override;
};