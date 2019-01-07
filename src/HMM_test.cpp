#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>
#include <fstream>
#include <HMM/Hmm.h>
#include <unsupported/Eigen/CXX11/Tensor>
#define debug_on true
#define debug_log(message) {if (!debug_on) {return 0;};std::cout << "file: " <<__FILE__ << "\nfunction: " << __FUNCTION__ << "\nline: " << __LINE__ << "\nmessage: "<< message << std::endl; return 0;}

using std::cout;
using std::endl;



std::vector<int> fileReader(std::string file_path){
    std::ifstream ifs(file_path);

    std::vector<int> nums;

    std::string data;
    ifs >> data;

    int s = 0;
    int e = 0;

    auto sz = data.size();
    for(decltype(sz) i = 0; i < sz; i++){

        if (data[i] == ','){

            e = i;
            float num = atoi(data.substr(s, e-s).c_str());
            nums.push_back(num);
            s = i + 1;
        }
    }

    return nums;
}


struct Matrix{

    int rows;
    int cols;
    float data[];

};


/*
 * space and time
 *
 * fast and slow
 *
 * computation and load file
 *
 * prediction fast iterate on each cells for a new observation sequence
 *
 * offline , full history, pre-train fast iteration on a full observation
 *
 * online train
 *
 *
 * laser processor : get map , robot pose, laser ,
 * output a vector of 
 *
 * */


/*
 * map of vctor
 * <cell_id, new_obs>
 *
 * */



int main(){


    std::string fp = "/home/waxz/data.txt";

    auto sim_data = fileReader(fp);



    // initialise model A B Q Fi Gama
    Eigen::MatrixXf Q(1, STATE_DIM) ;
    Q << 0.5, 0.5;
    Eigen::MatrixXf A(STATE_DIM, STATE_DIM);
    A << 0.5,0.5,0.5,0.5;

    Eigen::MatrixXf B(STATE_DIM, OBS_DIM);
    B << 0.2,0.5,0.3,0.4,0.2,0.4;

    Hmm::TensorX4 Fi(STATE_DIM,STATE_DIM,STATE_DIM,OBS_DIM);
    Hmm::TensorX3 Gama(STATE_DIM,STATE_DIM,OBS_DIM);

    Fi.setZero();
    Gama.setZero();

    Eigen::Matrix<float,2,2> A_(2, 2);
    A_ << 0.5,0.5,0.5,0.5;

//    Eigen::Matrix<float,2,3> B_;
//    B_ << 0.2,0.5,0.3,0.4,0.2,0.4;
    Eigen::Map<Eigen::Matrix<float,2,3>> B_(B.data());

    Hmm::HmmParams params(Q, A, B, Fi, Gama);

    time_util::Timer tm;
    tm.start();
    float  ss = 0.0;
    for (int i=0;i<5000*24*2;i++){
//        Eigen::Matrix<float,2,2> ta = params.A();
//        Eigen::Matrix<float,2,3> tb = params.B();
//
//        Eigen::Matrix<float,2,3> ab = params.A()*params.B();
//        Eigen::Matrix<float,1,2> qa = params.Q()*params.A();
        Hmm::MatrixQ qq = params.Q() * params.A();
//        qq(0,0) = 89.44;
        params.setQ(qq);
//        pq = 45.6;
//        params.Q(0,0) = 78.9;
//        params.Q_ptr.get()->operator()(0,0) = 67.8;
//        float jg = ab(0);
//        ss += jg;

//        cout << "jg " << jg;

    }
    tm.stop();
    auto pq = params.Q(0,0) ;

    cout << "tm time " << tm.elapsedMicroseconds() <<"sum: " << ss <<"pq: " << pq << endl;
//    return 0;
//    Eigen::Tensor<float, 3> mt(3,3,2);
//    mt.setValues({{{1,2},{2,9},{3,0}},{{4,3},{5,2},{6,3}},{{7,4},{8,5},{9,6}}});
//
//    Eigen::array<int, 2> d1({2,0});
//    Eigen::Tensor<float, 1> mts = mt.sum(d1);
//    cout << "mts:\n" << mts << endl;
//    return 0;



    Hmm::TensorFi Fit(Fi);


//    return 0;
    std::vector<Hmm::HmmParams> params_vec;

    time_util::Timer timer2;
    timer2.start();
#if 0
    int gz = 20*20*100*100;
    for(int i=0; i<2;i++){
        Hmm::HmmParams params1(Q, A, B, Fi, Gama);

        params_vec.push_back(params);

    }

    params_vec[0].A()(0,0) = 9090;
    cout << "update param: " << params.A()(0,0) << endl;
    timer2.stop();
    cout << "create time: " << timer2.elapsedSeconds() << endl;
    cout << "memory size: " << gz * 50 * 4 /1e6 <<"MB" <<endl;
    {
        int scope_a = 909;
    }
    return 0;
#endif
    float lr = 0.001;
    std::vector<int> data = {0,1,2};


//    timer2.Sleep(5);

#if 0
    updateGama(params, 0);
//    updateGama(params, 1);
//    updateGama(params, 2);

    updateFi(params, 0, lr);
    cout << "Fi:\n" << params.Fi()(0,0,0,0) << ", " << params.Fi()(0,1,1,0) << ", "
         << params.Fi()(1,0,0,0) << ", " << params.Fi()(1,1,1,0) << endl;
    updateQ(params, 0);
#endif

#if 1
    bool first = true;
    time_util::Timer timer;
    timer.start();
    int bz = 50;
    for (int i = 0; i < bz;i++){
//        cout << "=== " << i << endl;
        int cs = (first) ? 100:0;
        time_util::Timer timer1;
        timer1.start();
        Hmm::learn(params,sim_data,lr, cs);

        timer1.stop();
        cout << "one loop run time: " << timer1.elapsedMicroseconds() << endl;

        cout << "A:\n" << params.A() << endl;
        cout << "B:\n" << params.B() << endl;
    }
    timer.stop();
    cout << "all run time: " << timer.elapsedSeconds() << endl;
#endif

//    Hmm::updateModel(params);




#if 0

    for (int i = 0 ; i < 2 ; i++){
        Eigen::Tensor<float, 3> tensor = params.Fi().chip(i,0);
        Eigen::Tensor<float, 2> tensor2 = params.Gama().chip(i,0);
        cout << "Gama " << i << "====\n" << tensor2 << endl;
        for (int j=0;j <2 ;j++){
            cout << "Fi " << i << ", " << j <<"====\n" << tensor.chip(j,0) << endl;

        }
    }
#endif

    return 0 ;

#if 0


    Eigen::array<int, 3> o({0,0,0});

    cout << "gama:\n" << Gama <<endl;
    cout << "chip:\n" << Gama.chip(1,0) << endl;
    Eigen::Tensor<float,2 > j = Gama.chip(1,0).setZero();
    cout << j.dimensions().size() <<"," << j.dimension(0) << ", " << j.dimension(1) << endl;
    cout << "gama:\n" << Gama <<endl;

    A.setRandom();
    auto d = A.diagonal();
    d(0) = 2;
    A.diagonal() = d;
    cout << "A:\n" << A << endl;
    cout << "B:\n" << A*B << endl;
    Eigen::MatrixXf C;
    C = A*B;
    C.setRandom();
    C *= 2.3;
    // map matrix to tensor
    Eigen::Tensor<float,2> m;
    Eigen::MatrixXf n(2,2);
    n.setRandom();
    float da[4] = {1.2,2.2,3.3,4.4};
    Eigen::TensorMap<Eigen::Tensor<float, 2>> mm(n.data(), 2,2);
    cout << "gama:\n" << Gama <<endl;

    Gama.chip(1,2) = mm;
    cout << "gama:\n" << Gama <<endl;

    cout << "mm:\n" << mm <<endl;

    Eigen::array<int,2> offset({0,1});

    auto fs = Gama.sum(offset);

    cout << "fs:\n" << fs << endl;



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