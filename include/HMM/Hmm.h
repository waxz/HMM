//
// Created by waxz on 18-12-24.
//

#ifndef HMM_HMM_H
#define HMM_HMM_H
#include <HMM/time.h>

#define EIGEN_RUNTIME_NO_MALLOC // Define this symbol to enable runtime tests for allocations
#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
//#include <unsupported/Eigen/CXX11/TensorFixedSize>
#include <vector>
#include <iostream>
#include <memory>
#include <cassert>
#include <valarray>
#define STATE_DIM 2
#define OBS_DIM 3

namespace Hmm{

    typedef Eigen::Tensor<float,4> TensorX4;
    typedef Eigen::Tensor<float,3> TensorX3;

    typedef Eigen::TensorFixedSize<float,Eigen::Sizes<STATE_DIM,STATE_DIM,OBS_DIM>> TensorGama;
    typedef Eigen::TensorFixedSize<float,Eigen::Sizes<STATE_DIM,STATE_DIM>> TensorGama2;

    typedef Eigen::TensorFixedSize<float,Eigen::Sizes<STATE_DIM,STATE_DIM,STATE_DIM,OBS_DIM>> TensorFi;
    typedef Eigen::Matrix<float,1,STATE_DIM> MatrixQ;
    typedef Eigen::Matrix<float,STATE_DIM,STATE_DIM> MatrixTrans;
    typedef Eigen::Matrix<float,STATE_DIM,OBS_DIM> MatrixObs;


    // simple help function

    // data structure to store hmm related params
    // save data to
    // use tensor
    class HmmParams{
    protected:
        std::shared_ptr<MatrixQ> Q_ptr;
        std::shared_ptr<MatrixTrans> A_ptr;
        std::shared_ptr<MatrixObs> B_ptr;
        std::shared_ptr<TensorFi> Fi_ptr;
        std::shared_ptr<TensorGama> Gama_ptr;
        std::vector<Eigen::Matrix<float, STATE_DIM, STATE_DIM>> Bo;
        bool valid;

    public:

        ~HmmParams(){

        }
        int state_dim;
        int obs_dim;
//        constexpr int getStateDim(){
//            return state_dim;
//        }
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

        inline std::vector<Eigen::Matrix<float, STATE_DIM, STATE_DIM>> &
        Bi() {


            if (Bo.empty()) {
                Eigen::Matrix<float, STATE_DIM, OBS_DIM> B = *B_ptr;

                for (int i = 0; i < OBS_DIM; i++) {
                    Eigen::Matrix<float, STATE_DIM, STATE_DIM> Bi = Eigen::Matrix<float, STATE_DIM, STATE_DIM>::Zero();

                    Bi.diagonal() = B.col(i);


                    Bo.push_back(Bi);


//                    exit(12);
                }
            }

            return Bo;
        }

        MatrixQ& Q(){
            return *Q_ptr;
        }
        float& Q(int i, int j){
            return (*Q_ptr)(i,j);

        }
        template <typename T>
        void setQ(T &m){
            *Q_ptr = m;

//            std::swap(*Q_ptr,m);
        }

        template <typename T>
        void setFi(T&& m){

            *Fi_ptr = m;
        }
        MatrixTrans& A(){
            return *A_ptr;
        }
        MatrixObs& B(){
            return *B_ptr;
        }
        TensorFi& Fi(){
            return *Fi_ptr;
        }
        inline float& Fi(int i, int j, int h, int k){
            return (*Fi_ptr)(i,j,h,k);

        }
        TensorGama& Gama(){
            return *Gama_ptr;
        }


        HmmParams(Eigen::MatrixXf& Q_, Eigen::MatrixXf& A_,
                  Eigen::MatrixXf& B_, TensorX4& Fi_,
                  TensorX3& Gama_):
                Q_ptr(std::make_shared<MatrixQ>(Q_)),
                A_ptr(std::make_shared<MatrixTrans>(A_)),
                B_ptr(std::make_shared<MatrixObs>(B_)),
                Fi_ptr(std::make_shared<TensorFi>(Fi_)),
                Gama_ptr(std::make_shared<TensorGama>(Gama_)),
                Bo(),
                valid(false) {
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
    inline float deltaCompare2(int i, int j){
        return (i == j) ? 1.0f : 0.0f;
    }

// equation 2.9
    inline float deltaCompare4(int i, int j, int l, int h){
        return deltaCompare2(i, l)*deltaCompare2(j, h);
    }

/* equation 2.12
 * Gama(T) is updated with new_observation_k, transition_prob_A, emition_prob_B and Q(T-1),
 * Gama(T) = f(y(T), A, B, Q(T-1))
 * */

    inline  void updateGama(HmmParams &params,  int new_obs);



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
    inline void updateQ(HmmParams &params, int new_obs);

/* equation 2.11
 * update Fi with Gama, y(T), Q(T-1)
 * Fi(T) = f(Gama(T), y(T), Q(T-1), Fi(T-1))
 *
 * */
    inline void updateFi(HmmParams &params,  int new_obs, double learning_rate);



    /* update A, B
     * equation 2.14
     * */

    inline void updateModel(HmmParams& params, bool A_update, bool B_update);

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
    void learn(HmmParams &params, std::vector<int> &training_data, float learning_rate, int cache_step = 0);


    /* predict current state with latest observation
     * store prediction result in Q
     * */

    void predict(HmmParams &params, std::vector<int> &training_data);

    /*
     * get stable state probility if no observation data get
     * */

    /* == save and load
     *
     * */
}


#endif //HMM_HMM_H
