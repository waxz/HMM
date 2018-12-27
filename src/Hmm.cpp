//
// Created by waxz on 18-12-24.
//

#include <HMM/Hmm.h>

void Hmm::updateGama(HmmParams &params, int new_obs) {

    std::cout << "updateGama, get Gama:\n" << params.Q() << "\nA:\n" << params.A() << "\nB:\n" << params.B() << "\nnew_obs: " << new_obs << std::endl;

    auto QAB = params.Q()*(params.A()*params.B().col(new_obs));
    float p_sum = QAB(0,0);// = Q*(A*B.col(new_obs));

    std::cout << "p_sum: " << p_sum <<std::endl;

//    auto AB =
    auto B_eye = params.A();

    B_eye.setZero();

    B_eye.diagonal() = params.B().col(new_obs);

    Eigen::MatrixXf AB_eye;
    AB_eye = params.A()*B_eye;

    if (p_sum == 0.0){
        p_sum = 0.001;
    }

    AB_eye.setZero();


    AB_eye.array() /= p_sum;

    // matrix to tensor
    Eigen::TensorMap<Eigen::Tensor<float, 2>> t_AB(AB_eye.data(),AB_eye.rows(),AB_eye.cols());

    params.Gama().chip(new_obs,2) = t_AB;

#if 0

#endif
}

void Hmm::updateQ(HmmParams &params, int new_obs) {
    std::cout << "updateQ, get Q:\n" << params.Q() << "\nGama:\n" << params.Gama() << "\nnew_obs: " << new_obs << std::endl;

}

void Hmm::updateFi(HmmParams &params, int new_obs, double learning_rate) {
    std::cout << "updateFi, get Q:\n" << params.Q() << "\nGama:\n" << params.Gama() << "\nnew_obs: " << new_obs << std::endl;
    std::cout << "updateFi, get Fi:\n" << params.Fi() << "\nlearning_rate:\n" << learning_rate << std::endl;

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

        updateGama(params, ob);
        updateFi(params, ob, learning_rate);
        updateQ(params, ob);
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