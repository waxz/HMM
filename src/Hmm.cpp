//
// Created by waxz on 18-12-24.
//

#include <HMM/Hmm.h>

void Hmm::updateGama(Eigen::MatrixXd &Q, int new_obs, Eigen::MatrixXd &A, Eigen::MatrixXd &B) {

    std::cout << "updateGama, get Gama:\n" << Q << "\nA:\n" << A << "\nB:\n" << B << "\nnew_obs: " << new_obs << std::endl;
}

void Hmm::updateQ(Eigen::MatrixXd &Q, int new_obs, Hmm::Tensor3f &Gama) {
    std::cout << "updateQ, get Q:\n" << Q << "\nGama:\n" << Gama << "\nnew_obs: " << new_obs << std::endl;

}

void Hmm::updateFi(Hmm::Tensor4f &Fi, Hmm::Tensor3f &Gama, Eigen::MatrixXd &Q, int new_obs, double learning_rate) {
    std::cout << "updateFi, get Q:\n" << Q << "\nGama:\n" << Gama << "\nnew_obs: " << new_obs << std::endl;
    std::cout << "updateFi, get Fi:\n" << Fi << "\nlearning_rate:\n" << learning_rate << std::endl;

}

void Hmm::updateModel(HmmParams &params) {
    std::cout << "updateModel, get Fi:\n" << params.Fi() << std::endl;

}

void Hmm::learn(HmmParams &params, std::vector<int > training_data, float learning_rate) {

    std::cerr<< "training data size " << training_data.size() << std::endl;

    // check training data, length
    if (training_data.empty()){

        std::cerr<< "training data empty!" << std::endl;
        return;

    }


    // update param
    // 1) Gama
    // 2) Fi
    // 3) Q

    // ? when to update A,B
    for (auto ob : training_data){

        updateGama(params.Q(), ob,params.A(), params.B());
        updateFi(params.Fi(), params.Gama(), params.Q(), ob, learning_rate);
        updateQ(params.Q(), ob, params.Gama());
    }
}

void Hmm::predict(HmmParams &params, std::vector<int> training_data) {
    std::cerr<< "training data size " << training_data.size() << std::endl;

    // check training data, length
    if (training_data.empty()){

        std::cerr<< "training data empty!" << std::endl;
        return;

    }

}