#include "gtest/gtest.h"

#include <cmath>
#include <tuple>
#include <type_traits>

// assumes a constants/<dim><type>.h is already included

class ArrayNdTestFixture : public ::testing::TestWithParam<int> {
protected:
  double getRate() { return std::ldexp(1.0, GetParam() + 3); }
};



typedef std::tuple<int,int,int> testConfig;

#define TEST_RATE zfp_mode_fixed_rate
#define TEST_PREC zfp_mode_fixed_precision
#define TEST_ACCU zfp_mode_fixed_accuracy
#define TEST_RVRS zfp_mode_reversible

#define TEST_INDEX_IMP 0
#define TEST_INDEX_VRB 1
#define TEST_INDEX_HY4 2
#define TEST_INDEX_HY8 3

#define TEST_INDEX_TYPE_IMP zfp::index::implicit
#define TEST_INDEX_TYPE_VRB zfp::index::verbatim
#define TEST_INDEX_TYPE_HY4 zfp::index::hybrid4
#define TEST_INDEX_TYPE_HY8 zfp::index::hybrid8

class CArrayNdTestFixture : public ::testing::TestWithParam<testConfig> {
protected:
  static double         getRate(int param)      { return std::ldexp(1.0, param + 3); }
  static unsigned int   getPrecision(int param) { return 1u << (param + 3); }
  static double         getTolerance(int param) { return std::ldexp(1.0, -(1u << param)); }

  // get(0): config mode selection
  // get(1): config mode value selection
  // get(2): block index type selection
  zfp_config getConfig()
  {
    zfp_config config;

    switch(std::get<0>(GetParam())) {
      case zfp_mode_fixed_rate:
      {
        //TODO: check with/without align?
        config = zfp_config_rate(getRate(std::get<1>(GetParam())), true);
        break;
      }
      case zfp_mode_fixed_precision:
      {
        config = zfp_config_precision(getPrecision(std::get<1>(GetParam())));
        break;
      }
      case zfp_mode_fixed_accuracy:
      {
        config = zfp_config_accuracy(getTolerance(std::get<1>(GetParam())));
        break;
      }
      case zfp_mode_reversible:
      {
        config = zfp_config_reversible();
        break;
      }
#if 0
      case zfp_mode_expert:
      {
        //TODO: do we need this one?
        //config = zfp_config_expert(uint minbits, uint maxbits, uint maxprec, int minexp);
        //break;
      }
#endif
      default:
      {
        config = zfp_config_none();
        break;
      }
    }
    return config;
  }

public:
  struct PrintToStringParamName
  {
    static std::string IndexToStr(int idx)
    {
      switch (idx)
      {
        case TEST_INDEX_IMP:
        {
            return "Implicit";
        }
        case TEST_INDEX_VRB:
        {
            return "Verbatim";
        }
        case TEST_INDEX_HY4:
        {
            return "Hybrid4";
        }
        case TEST_INDEX_HY8:
        {
            return "Hybrid8";
        }
        default:
        {
            return "BadIdxType";
        }
      }
    }

    template <class ParamType>
    std::string operator()(const testing::TestParamInfo<ParamType>& info) const
    {
       std::stringstream out;
       switch(std::get<0>(info.param))
       {
          case zfp_mode_fixed_rate:
          {
             out << "Fixed_Rate_val" << std::get<1>(info.param) << "_idx" << IndexToStr(std::get<2>(info.param));
             break;
          }
          case zfp_mode_fixed_precision:
          {
             out << "Fixed_Precision_val" << std::get<1>(info.param) << "_idx" << IndexToStr(std::get<2>(info.param));
             break;
          }
          case zfp_mode_fixed_accuracy:
          {
             out << "Fixed_Accuracy_val" << std::get<1>(info.param) << "_idx" << IndexToStr(std::get<2>(info.param));
             break;
          }
          case zfp_mode_reversible:
          {
             out << "Reversible_idx" << IndexToStr(std::get<2>(info.param));
             break;
          }
          case zfp_mode_expert:
          {
             out << "Expert_val" << std::get<1>(info.param) << "_idx" << IndexToStr(std::get<2>(info.param));
             break;
          }
       }
       return out.str();
    }
  };

};
