//
// Created by waxz on 18-12-24.
//

#include <HMM/Hmm.h>

 void Hmm::updateGama(HmmParams &params, int new_obs) {


     auto Bi = params.B().col(new_obs);
     Eigen::Matrix<float, 1, 1> QAB = params.Q() * (params.A() * Bi);
#if 0
    Eigen::Matrix<float,STATE_DIM,STATE_DIM> B_eye = Eigen::Matrix<float,STATE_DIM,STATE_DIM>::Zero();

    B_eye.diagonal() = params.B().col(new_obs);
#endif
     auto B_eye = Bi.asDiagonal();

    Eigen::Matrix<float,STATE_DIM,STATE_DIM> AB_eye = params.A()*B_eye;

    Eigen::Matrix<float,2,2> eye_matrix = Eigen::Matrix<float, 2, 2>::Identity();

    eye_matrix *= 1.0f/QAB(0,0);

     AB_eye *= eye_matrix;

    // matrix to tensor
    Eigen::TensorMap<Eigen::Tensor<float, 2>> t_AB(AB_eye.data(),AB_eye.rows(),AB_eye.cols());

    params.Gama().chip(new_obs,2) = t_AB;

#if 0
    std::cout << "updateGama, get Q:\n" << params.Q() << "\nA:\n" << params.A() << "\nB:\n" << params.B() << "\nnew_obs: " << new_obs << std::endl;
    std::cout << "updated Gama:\n" << params.Gama() << std::endl;
#endif
}

 void Hmm::updateQ(HmmParams &params, int new_obs) {


    TensorGama2 g = params.Gama().chip(new_obs,2);

    Eigen::Map<Eigen::Matrix<float, 2, 2>> G(g.data());
     params.Q() *= G;


#if 0
    std::cout << "updateQ, get Q:\n" << params.Q() << "\nGama:\n" << params.Gama() << "\nnew_obs: " << new_obs << std::endl;

    std::cout << "updated Q: " << params.Q() << std::endl;

#endif

}

void Hmm::updateFi(HmmParams &params, int new_obs, double learning_rate) {

#if 0
    std::cout << "updateFi, get Q:\n" << params.Q() << "\nGama:\n" << params.Gama() << "\nnew_obs: " << new_obs << std::endl;
    std::cout << "\nFi:\n" << params.Fi() << "\nlearning_rate:\n" << learning_rate << std::endl;
#endif

    auto dim0 = params.Fi().dimension(0);
    auto dim1 = params.Fi().dimension(1);
    auto dim2 = params.Fi().dimension(2);
    auto dim3 = params.Fi().dimension(3);

    TensorFi &Fi = params.Fi();
    TensorFi &Fi_updated = params.Fi();

    TensorGama &Gama = params.Gama();
    MatrixQ &Q = params.Q();
    for (int i = 0; i < dim0; i++){
        for (int j = 0; j < dim1; j++){
            for (int h = 0; h < dim2; h++){
                for (int k = 0; k < dim3; k++){

                    float f = 0.0;
                    for (int l = 0; l< dim1;l++){
                        f += Gama(l, h, new_obs)*(Fi(i,j,l,k) + learning_rate*(deltaCompare2(new_obs,k) * deltaCompare4(i,j,l,h)*Q(0,l) - Fi(i,j,l,k) ));
                    }
                    Fi_updated(i,j,h,k) = f;
                }
            }
        }
    }
    params.setFi(std::move(Fi_updated));
//    params.Fi() = Fi_updated;
#if 0
    std::cout << "updated Fi:\n" << params.Fi() << std::endl;
//    exit(23);
#endif
}

void Hmm::updateModel(HmmParams &params, bool A_update, bool B_update) {
#if 0
    std::cout << "updateModel, get Fi:\n" << params.Fi() << std::endl;
#endif

    auto dim0 = STATE_DIM;
    auto dim1 = OBS_DIM;

    Eigen::array<int, 2> Ai({2,3});
    Eigen::array<int, 2> Bi({0,2});

    TensorFi &Fi = params.Fi();

    Eigen::TensorFixedSize<float, Eigen::Sizes<STATE_DIM, STATE_DIM>> A_sum = Fi.sum(Ai);
    Eigen::TensorFixedSize<float, Eigen::Sizes<STATE_DIM, OBS_DIM>> B_sum = Fi.sum(Bi);


    Eigen::Map<Eigen::Matrix<float, STATE_DIM, STATE_DIM>> A_sum_m(A_sum.data());
    Eigen::Map<Eigen::Matrix<float, STATE_DIM, OBS_DIM>> B_sum_m(B_sum.data());


    Eigen::TensorFixedSize<float, Eigen::Sizes<STATE_DIM>> Ar;
    Eigen::TensorFixedSize<float, Eigen::Sizes<>> row_sum;

    Eigen::TensorFixedSize<float, Eigen::Sizes<OBS_DIM>> Br;

    for (int i = 0; i < dim0; i++){


        if (A_update){
            Ar = A_sum.chip(i, 0);
            row_sum = Ar.sum();
            //            Eigen::Tensor<float,0> A2_row_sum = Ar.sum();

            params.A().row(i) = A_sum_m.row(i) / row_sum(0);

        }

        if (B_update){
            Br = B_sum.chip(i, 0);
            row_sum = Br.sum();

            params.B().row(i) = B_sum_m.row(i) / row_sum(0);

        }


    }

#if 0

    std::cout << "updated A:\n" << params.A();
    std::cout << "updated B:\n" << params.B() << std::endl;
//    exit(89);
#endif


}

void Hmm::learn(HmmParams &params, std::vector<int> &training_data, float learning_rate, int cache_step) {

#if 0
    std::cerr<< "training data size " << training_data.size() << std::endl;
#endif
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
#if 0
    long i = 0;
    time_util::Timer t1;
    time_util::Timer t2;
    time_util::Timer t3;
    time_util::Timer t4;
    time_util::Timer t0;


    t1.start();
    for (auto ob : training_data){

        updateGama(params, ob);
    }
    t1.stop();
    std::cout << " updateGama time: " << t1. elapsedMicroseconds() << std::endl;


    t2.start();
    for (auto ob : training_data){
        updateFi(params, ob, learning_rate);
    }
    t2.stop();
    std::cout << " updateFi time: " << t2. elapsedMicroseconds() << std::endl;


    t3.start();
    for (auto ob : training_data){

        updateQ(params, ob);
    }
    t3.stop();
    std::cout << " updateQ time: " << t3. elapsedMicroseconds() << std::endl;


    t4.start();
    for (auto ob : training_data){
        updateModel(params,true, true);
    }
    t4.stop();
    std::cout << " updateModel time: " << t4. elapsedMicroseconds() << std::endl;

    exit(0);
#endif
#if 1
    time_util::Timer t;
    t.start();
    for (auto ob : training_data){
        Eigen::internal::set_is_malloc_allowed(false);
        // It's NOT OK to allocate here
        // An assertion will be triggered if an Eigen-related heap allocation takes place
        updateGama(params, ob);

        Eigen::internal::set_is_malloc_allowed(true);
        // It's OK to allocate again

        updateFi(params, ob, learning_rate);

        updateQ(params, ob);

        updateModel(params,true, false);
    }

    t.stop();
    std::cout << " training time: " << t. elapsedMicroseconds() << std::endl;
#endif
}

void Hmm::predict(HmmParams &params, std::vector<int> &training_data) {
#if 0
    std::cerr<< "training data size " << training_data.size() << std::endl;
#endif
    // check training data, length
    if (training_data.empty()){

        std::cerr<< "training data empty!" << std::endl;
        return;

    }

    // B matrix
    std::vector<Eigen::Matrix<float, STATE_DIM, STATE_DIM>> &Bo = params.Bi();


    for (auto ob : training_data) {

        params.Q() *= params.A() * Bo[ob];
    }
    params.Q() /= params.Q().sum();

}