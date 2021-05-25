#include "gtest/gtest.h"

#include <cmath>
#include <tuple>

// assumes a constants/<dim><type>.h is already included

class ArrayNdTestFixture : public ::testing::TestWithParam<int> {
protected:
  double getRate() { return 1u << (GetParam() + 3); }
};

typedef std::tuple<int,int,int> testConfig;

#define TEST_RATE zfp_mode_fixed_rate
#define TEST_PREC zfp_mode_fixed_precision
#define TEST_ACCU  zfp_mode_fixed_accuracy
#define TEST_RVRS zfp_mode_reversible

class CArrayNdTestFixture : public ::testing::TestWithParam<testConfig> {
protected:
  // get(0): config mode selection
  // get(1): config mode value selection
  // get(2): for later testing across multiple different indexes
  zfp_config getConfig()
  {
    zfp_config config;

    switch(std::get<0>(GetParam())) {
      case zfp_mode_fixed_rate:
      {
        //TODO: check with/without align?
        double rate = 1u << (std::get<1>(GetParam()) + 3);
        config = zfp_config_rate(rate, true);
        break;
      }
      case zfp_mode_fixed_precision:
      {
        unsigned int precision = 1u << (std::get<1>(GetParam()) + 3);
        config = zfp_config_precision(precision);
        break;
      }
      case zfp_mode_fixed_accuracy:
      {
        double tolerance = std::pow(2, -(1 << (std::get<1>(GetParam()) + 3)));
        config = zfp_config_accuracy(tolerance);
        break;
      }
      case zfp_mode_reversible:
      {
        config = zfp_config_reversible();
        break;
      }
      case zfp_mode_expert:
      {
        //TODO: do we need this one?
        //config = zfp_config_expert(uint minbits, uint maxbits, uint maxprec, int minexp);
        //break;
      }
      default:
      {
        config = zfp_config_none();
        break;
      }
    }
    return config;
  }
};
