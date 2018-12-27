//
// Created by waxz on 18-12-24.
//

#ifndef HMM_HMM_H
#define HMM_HMM_H

#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

#include <vector>
#include <iostream>
#include <memory>
#include <cassert>

namespace Hmm{
    typedef Eigen::Tensor<float,2> Tensor2f;
    typedef Eigen::Tensor<float,3> Tensor3f;
    typedef Eigen::Tensor<float,4> Tensor4f;
    // simple help function

    // data structure to store hmm related params
    // save data to
    // use tensor
    class HmmParams{
    private:
        std::shared_ptr<Eigen::MatrixXf> Q_ptr;
        std::shared_ptr<Eigen::MatrixXf> A_ptr;
        std::shared_ptr<Eigen::MatrixXf> B_ptr;
        std::shared_ptr<Tensor4f> Fi_ptr;
        std::shared_ptr<Tensor3f> Gama_ptr;

        int state_dim;
        int obs_dim;
        bool valid;

    public:
        bool isValid(){
            return valid;
        }



#if 0
        Eigen::MatrixXf& Q;
        Eigen::MatrixXf& A;
        Eigen::MatrixXf& B;
        Eigen::MatrixXf& Fi;
        Eigen::MatrixXf& Gama;
#endif

        Eigen::MatrixXf& Q(){
            return *Q_ptr;
        }
        Eigen::MatrixXf& A(){
            return *A_ptr;
        }
        Eigen::MatrixXf& B(){
            return *B_ptr;
        }
        Tensor4f& Fi(){
            return *Fi_ptr;
        }
        Tensor3f& Gama(){
            return *Gama_ptr;
        }


        HmmParams(Eigen::MatrixXf& Q_, Eigen::MatrixXf& A_,
                  Eigen::MatrixXf& B_, Tensor4f& Fi_,
                  Tensor3f& Gama_):
                Q_ptr(std::make_shared<Eigen::MatrixXf>(Q_)),
                A_ptr(std::make_shared<Eigen::MatrixXf>(A_)),
                B_ptr(std::make_shared<Eigen::MatrixXf>(B_)),
                Fi_ptr(std::make_shared<Tensor4f>(Fi_)),
                Gama_ptr(std::make_shared<Tensor3f>(Gama_)),
                valid(false){
            // check dimention

            state_dim = A_.cols();
            obs_dim = B_.cols();
            assert(A_.cols() == A_.rows());
            assert(B_.rows() == state_dim);
            assert(Q_.rows() == 1);
            assert(Q_.cols() == state_dim);


            assert(Gama_.dimension(0) == state_dim);
            assert(Gama_.dimension(1) == state_dim);
            assert(Gama_.dimension(2) == obs_dim);

            assert(Fi_.dimension(0) == state_dim);
            assert(Fi_.dimension(1) == state_dim);
            assert(Fi_.dimension(2) == state_dim);
            assert(Fi_.dimension(3) == obs_dim);

            valid = true;

        }

        // save model
        void save(std::string file_path){

        }

        // load model

        void load(std::string file_path){

        }

    };
// equation 2.1
    inline bool deltaCompare2(int i, int j){
        return (i == j) ? 1 : 0;
    }

// equation 2.9
    inline bool dealtaCompare4(int i, int j, int l, int h){
        return deltaCompare2(i, l)*deltaCompare2(j, h);
    }

/* equation 2.12
 * Gama(T) is updated with new_observation_k, transition_prob_A, emition_prob_B and Q(T-1),
 * Gama(T) = f(y(T), A, B, Q(T-1))
 * */

    void updateGama(HmmParams &params,  int new_obs);



/* equation 2.13 : Q[l](T − 1) ≡ P(x(T−1) = l| y(0→T−1))
 * Q is updated with Gama and Q
 * Q(T) == f(Gama(T), Q(T-1))
 * definition : State estimates, Q[l](T) (i.e., the probability of being in the state l at time T)
 * l : state l in {s1, s2, ...}
 * T : time step {0, 1, 2, 3, ...}
 * y : observation
 * update recursively : equation 2.13
 * Q[l](0) = P(x0 = l)
 * P(x0 = l) : initial state distribution
 * */
    void updateQ(HmmParams &params, int new_obs);

/* equation 2.11
 * update Fi with Gama, y(T), Q(T-1)
 * Fi(T) = f(Gama(T), y(T), Q(T-1), Fi(T-1))
 *
 * */
    void updateFi(HmmParams &params,  int new_obs, double learning_rate);



    /* update A, B
     * equation 2.14
     * */

    void updateModel(HmmParams& params);

/* =================
 *
 * HMM model:
 * state space, observation space, transition_matrix A, emition_matrix B, initial_distribution pi
 * only process data
 * no internal state reserved
 *
 *
 *
 *
 * */
    void learn(HmmParams& params, std::vector<int > training_data, float learning_rate);


    /* predict current state with latest observation
     * store prediction result in Q
     * */

    void predict(HmmParams& params, std::vector<int> training_data);


}


#endif //HMM_HMM_H
