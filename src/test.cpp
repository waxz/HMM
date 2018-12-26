#include "HMM/eigen_boost_serialization.hpp"

#include <vector>
#include <array>
#include <type_traits>
#include <fstream>

// include headers that implement a archive in simple xml format
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
// include headers that implement a archive in simple binary format
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
// include headers that implement a archive in simple text format
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

//#include <boost/test/minimal.hpp>
#include <Eigen/Core>


class gps_position
{
private:
    friend class boost::serialization::access;
    // When the class Archive corresponds to an output archive, the
    // & operator is defined similar to <<.  Likewise, when the class Archive
    // is a type of input archive the & operator is defined similar to >>.
#if 0
    template<class Archive>

    void serialize(Archive & ar, const unsigned int version)
    {
        ar & degrees;
        ar & minutes;
        ar & seconds;
    }
#endif

public:
    int degrees;
    int minutes;
    float seconds;
    gps_position(){};
    gps_position(int d, int m, float s) :
            degrees(d), minutes(m), seconds(s)
    {}
    void show()const {
        std::cout << "deg " << degrees<<std::endl;
    }
};

namespace boost {
    namespace serialization {

        template<class Archive>
        void serialize(Archive & ar, gps_position & g, const unsigned int version)
        {
            ar & g.degrees;
            ar & g.minutes;
            ar & g.seconds;
        }

    } // namespace serialization
} // namespace boost


int main() {
    // create and open a character archive for output
    std::ofstream ofs("filename", std::ios_base::binary);

    /* how to store and restore parameters
     * how to store and restore training data
     *
     * for hmm, concern about model parameters and training data
     *
     * one hmm object , deal with all model parameter and trinning data
     * feed data as reference
     * [Q, Gama, A, B, Fi, Data,Lr]
     *
     *
     *
     * */


    Eigen::MatrixXf m(4096,4000);

    m.setRandom();

    // create class instance
    const gps_position g(35, 59, 24.567f);
    std::cout ;
    g.show();

    // save data to archive
    {
        boost::archive::binary_oarchive oa(ofs);
        // write class instance to archive
        oa << m;
        // archive and stream closed when destructors are called
    }

    // ... some time later restore the class instance to its orginal state
    gps_position newg;
    Eigen::MatrixXf nm;
    {
        // create and open an archive for input
        std::ifstream ifs("filename", std::ios_base::binary);
        boost::archive::binary_iarchive ia(ifs);
        // read class state from archive

        ia >> nm;
        // archive an
        // d stream closed when destructors are called
    }

//    newg.show();
//    std::cout << "m\n" <<m << std::endl;

//    std::cout << "nm\n" <<nm << std::endl;
    return 0;
}


#if 0
int test_main(int, char**)
{
    std::string save_path = "./test.xml"; // *.txt/.bin/.mpac

    // save
    Eigen::MatrixXf A = Eigen::MatrixXf::Random(4, 3);
//    Matrix::Util::save(A, path);

    // load
    Eigen::MatrixXf B;

    std::ofstream ofs(save_path, std::ios_base::binary);
    boost::archive::binary_oarchive oar(ofs);




//    Matrix::Util::load(B, path);
//
//    // check
//    for ( std::size_t i = 0; i < A.rows(); ++i )
//    {
//        for ( std::size_t j = 0; j < A.cols(); ++j )
//        {
////            BOOST_CHECK(std::is_near(A(i, j), B(i, j)));
//        }
//    }


    return EXIT_SUCCESS;
}
#endif