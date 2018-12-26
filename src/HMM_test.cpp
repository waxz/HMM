#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>
#include <HMM/Hmm.h>
#include <unsupported/Eigen/CXX11/Tensor>
#define debug_on true
#define debug_log(message) {if (!debug_on) {return 0;};std::cout << "file: " <<__FILE__ << "\nfunction: " << __FUNCTION__ << "\nline: " << __LINE__ << "\nmessage: "<< message << std::endl; return 0;}

using std::cout;
using std::endl;

int main(){

    // initialise model A B Q Fi Gama
    Eigen::MatrixXd Q(1, 2);
    Eigen::MatrixXd A(2, 2);
    Eigen::MatrixXd B(2, 3);
    Hmm::Tensor4f Fi(2,2,2,3);
    Hmm::Tensor3f Gama(2,2,3);









    Hmm::HmmParams params(Q, A, B, Fi, Gama);

    float lr = 0.01;
    std::vector<int> data = {1,2,1,1,0,0,1,1};
    Hmm::learn(params,data,lr);

    std::cout << "m1:\n" << Q.rows() << std::endl;

//    std::cout << "p m1:\n" << params.Q() << std::endl;

//    params.Q()(0,0) = 990;
    params.Fi()(0,0,0,0) = 999;

    std::cout << "p Fi:\n" << params.Fi() << std::endl;

    std::cout << "Fi:\n" << Fi << std::endl;


#if 0
    Eigen::MatrixXf M1 = Eigen::MatrixXf::Random(3,8);
    cout << "Column major input:" << endl << M1 << "\n";

    Eigen::Map<Eigen::MatrixXf,0,Eigen::OuterStride<> > M2(M1.data(), M1.rows(), 2, Eigen::OuterStride<>(M1.outerStride()*3));
    cout << "1 column over 3:" << endl << M2 << "\n";
    typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> RowMajorMatrixXf;
    RowMajorMatrixXf M3(M1);
    cout << "Row major input:" << endl << M3 << "\n";
    Eigen::Map<RowMajorMatrixXf,0,Eigen::Stride<Eigen::Dynamic,3> > M4(M3.data(), M3.rows(), (M3.cols()+2)/3,
                                                                       Eigen::Stride<Eigen::Dynamic,3>(M3.outerStride(),3));
    cout << "1 column over 3:" << endl << M4 << "\n"<<endl;


    Eigen::Tensor<double,3> m(3,10,10);            //Initialize
    m.setRandom();                                 //Set random values
    Eigen::array<long,3> offset = {0,0,0};         //Starting point
    Eigen::array<long,3> extent = {1,10,10};       //Finish point
//    std::cout << "tensor slice: " << m.slice(offset, extent).reshape(Eigen::array<long,2>{10,10}) << std::endl;  //Reshape the slice into a 10x10 matrix


    auto j = m.slice(offset, extent).reshape(Eigen::array<long,2>{10,10});

    cout << "eee" << m.dimension(0) <<endl;
//    std::cout << "j;" << j << std::endl;

    int storage[8] = {0,1,2,3,4,5,6,7};
    Eigen::TensorMap<Eigen::Tensor<int, 3>> t_4d(storage, 2, 2, 2);

    Eigen::Tensor<int, 3> t_d(2,2,2);
    t_d.setValues({{{3,5},{6,8}},{{9,3},{5,5}}});
    cout << "t_d: \n" << t_d << endl;


    Eigen::TensorRef<Eigen::Tensor<int, 3>> ref = t_4d ;

    cout << "ref: " << ref(1,0,1) << endl;

    Eigen::Tensor<int, 3> tt(2,2,2);

    Eigen::DefaultDevice de;
    tt.device(de)= t_4d + t_4d;


    cout << "tt:\n" << tt << endl;


    // Create a tensor of 2 dimensions
    Eigen::Tensor<int, 2> a(2, 3);
    a.setValues({{1, 2, 3}, {6, 5, 4}});
// Reduce it along the second dimension (1)...
    Eigen::array<int, 2> dims({0, 2});
// ...using the "maximum" operator.
// The result is a tensor with one dimension.  The size of
// that dimension is the same as the first (non-reduced) dimension of a.
    Eigen::Tensor<int, 1> b = t_4d.maximum(dims);
    cout << "a" << endl << a << endl << endl;

    // Shuffle all dimensions to the left by 1.
    Eigen::Tensor<float, 3> input(20, 30, 50);
// ... set some values in input.

    Eigen::array<Eigen::DenseIndex, 2> strides({1,2});
    cout << "stride: " << a.stride(strides) << endl;


    Eigen::array<int,2> offset1({0,1});
    Eigen::array<int,2> extend1({2,2});

    Eigen::Tensor<int, 2> slice =  a.slice(offset1, extend1) ;
    cout << "a: \n" << a << endl;

    cout << "slice: \n" << slice << endl;
    cout << "chip: " << a.chip(1,0) << endl;

    Eigen::array<int,2> jj({1,8});
    Eigen::Tensor<int, 2> r = t_4d.reshape(jj);
    cout << "t4d" << r << endl;

    cout << "d " << a(1,0) << endl;

    Eigen::Tensor<double,3> m1(1,3,2);
    m1.setValues({{{1.0,2.0},{0.0,3.0},{0.0,3.0}}});

    Eigen::Tensor<double,2> m2(2,1);
    m2.setValues({{1.0},{3.0}});
    Eigen::MatrixXd m3(2,1);

    m3(0,0) = 1.0;
    m3(1,0) = 3.0;

    Eigen::TensorMap<Eigen::Tensor<double, 2>> m4(m3.data(), 2,1);

    cout << "m4:\n" << m4 << endl;

    Eigen::array<Eigen::IndexPair<int>, 1> prod_dim = {Eigen::IndexPair<int>(2,0)};
    auto m1m2 = m1.contract(m4, prod_dim);
    cout << " product: \n" << m1m2 << endl;
#endif
}