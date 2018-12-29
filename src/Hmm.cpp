//
// Created by waxz on 18-12-24.
//

#include <HMM/Hmm.h>

void Hmm::updateGama(HmmParams &params, int new_obs) {


    auto QAB = params.Q()*(params.A()*params.B().col(new_obs));
    float p_sum = QAB(0,0);// = Q*(A*B.col(new_obs));


//    auto AB =
    auto B_eye = params.A();

    B_eye.setZero();

    B_eye.diagonal() = params.B().col(new_obs);

    Eigen::MatrixXf AB_eye;
    AB_eye = params.A()*B_eye;

    if (p_sum == 0.0){
        p_sum = 0.001;
    }

//    AB_eye.setZero();


    AB_eye.array() /= p_sum;

    // matrix to tensor
    Eigen::TensorMap<Eigen::Tensor<float, 2>> t_AB(AB_eye.data(),AB_eye.rows(),AB_eye.cols());

    params.Gama().chip(new_obs,2) = t_AB;
#if 0
    std::cout << "updateGama, get Q:\n" << params.Q() << "\nA:\n" << params.A() << "\nB:\n" << params.B() << "\nnew_obs: " << new_obs << std::endl;

    std::cout << "p_sum: " << p_sum <<std::endl;

    std::cout << "t_AB:\n" << t_AB << std::endl;
    std::cout << "updated Gama:\n" << params.Gama() << std::endl;


#endif
}

void Hmm::updateQ(HmmParams &params, int new_obs) {

    //
    Eigen::Tensor<float, 2> g = params.Gama().chip(new_obs,2);
    Eigen::Map<Eigen::MatrixXf> G(g.data(),params.Gama().dimension(0),params.Gama().dimension(1) );

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
#if 0
    Eigen::MatrixXf delta1(1,dim3);
    delta1.setZero();
    delta1(0, new_obs) = 1.0;

    Eigen::MatrixXf delta2 = Eigen::MatrixXf::Identity(dim2,dim2);
#endif
    auto Fi = params.Fi();
    for (int i = 0; i < dim0; i++){
        for (int j = 0; j < dim1; j++){
            for (int h = 0; h < dim2; h++){
                for (int k = 0; k < dim3; k++){
#if 1

                    float f = 0.0;
                    for (int l = 0; l< dim1;l++){
//                        float t = params.Gama()(l, h, new_obs);
//                        float t2 = Fi(i,j,l,k);
f += params.Gama()(l, h, new_obs)*(Fi(i,j,l,k) + learning_rate*(deltaCompare2(new_obs,k) * deltaCompare4(i,j,l,h)*params.Q()(0,l) - Fi(i,j,l,k) ));
                    }
                    params.Fi()(i,j,h,k) = f;
#endif
//                    float t1 = params.Gama()(0, h, new_obs)*(Fi(i,j,0,k) + learning_rate*(deltaCompare2(new_obs,k) * deltaCompare4(i,j,0,h)*params.Q()(0,0) - Fi(i,j,0,k) ));
//                    float t2 = params.Gama()(1, h, new_obs)*(Fi(i,j,1,k) + learning_rate*(deltaCompare2(new_obs,k) * deltaCompare4(i,j,1,h)*params.Q()(0,1) - Fi(i,j,1,k) ));
//
//                    params.Fi()(i,j,h,k) = params.Gama()(0, h, new_obs)*(Fi(i,j,0,k) + learning_rate*(deltaCompare2(new_obs,k) * deltaCompare4(i,j,0,h)*params.Q()(0,0) - Fi(i,j,0,k) )) +
//                                           params.Gama()(1, h, new_obs)*(Fi(i,j,1,k) + learning_rate*(deltaCompare2(new_obs,k) * deltaCompare4(i,j,1,h)*params.Q()(0,1) - Fi(i,j,1,k) ));

/*
* self.Fi[i][j][h][k]  =
* (Gama_[0][h][new_obs_k] * (Fi[i][j][0][k] + self.time_factor*(delta(new_obs_k, k)*g_delta(i, j ,0, h)*Q_[0] - Fi[i][j][0][k] ) ) ) +\
* (Gama_[1][h][new_obs_k] * (Fi[i][j][1][k] + self.time_factor*(delta(new_obs_k, k)*g_delta(i, j ,1, h)*Q_[1] - Fi[i][j][1][k] ) ) )
*/
                }
            }
        }
    }
#if 0
    std::cout << "updated Fi:\n" << params.Fi() << std::endl;
#endif
}

void Hmm::updateModel(HmmParams &params, bool A_update, bool B_update) {
#if 0
    std::cout << "updateModel, get Fi:\n" << params.Fi() << std::endl;
#endif

    auto dim0 = params.A().cols();
    auto dim1 = params.B().cols();

    Eigen::array<int, 2> Ai({2,3});
    Eigen::array<int, 2> Bi({0,2});

    Eigen::Tensor<float,2> A_sum = params.Fi().sum(Ai);
    Eigen::Tensor<float,2> B_sum = params.Fi().sum(Bi);

    Eigen::Map<Eigen::MatrixXf> A_sum_m(A_sum.data(), dim0, dim0);
    Eigen::Map<Eigen::MatrixXf> B_sum_m(B_sum.data(), dim0, dim1);
#if 0
    Eigen::array<int,4> start({0,0,0,0});
    Eigen::array<int,4> extend({1,1,2,3});
    Eigen::Tensor<float,4> ms = params.Fi().slice(start, extend);

    std::cout << "ms:\n" << ms << std::endl;
    std::cout << "ms sum: " << ms.sum() << std::endl;


    std::cout << "A_sum: \n" << A_sum << std::endl;
#endif

    for (int i = 0; i < dim0; i++){


        if (A_update){
            Eigen::Tensor<float,1> Ar = A_sum.chip(i,0);
            Eigen::Tensor<float,0> A_row_sum = Ar.sum();
            params.A().row(i) = A_sum_m.row(i)/A_row_sum(0);

        }

        if (B_update){
            Eigen::Tensor<float,1> Br = B_sum.chip(i,0);
            Eigen::Tensor<float,0> B_row_sum = Br.sum();

            params.B().row(i) = B_sum_m.row(i)/B_row_sum(0);

        }


    }



}

void Hmm::learn(HmmParams &params, std::vector<int > training_data, float learning_rate, int cache_step ) {

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

    long i = 0;
    for (auto ob : training_data){

        i ++;
        updateGama(params, ob);
        updateFi(params, ob, learning_rate);
        updateQ(params, ob);
        if (i > cache_step){
            updateModel(params,true, false);
        }
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