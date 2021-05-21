#include "gtest/gtest.h"

#include <tuple>

// assumes a constants/<dim><type>.h is already included

class ArrayNdTestFixture : public ::testing::TestWithParam<int> {
protected:
  double getRate() { return 1u << (GetParam() + 3); }
};

typedef std::tuple<int,int,int> testConfig;

class CArrayNdTestFixture : public ::testing::TestWithParam<testConfig> {
protected:
  // get(0): config mode selection
  // get(1): config mode value selection
  // get(2): for later testing across multiple different indexes
  zfp_config getConfig()
  {
    zfp_config config;

    switch(std::get<0>(GetParam())) {
      case 0:
      {
        //TODO: check with/without align?
        double rate = 1u << (std::get<1>(GetParam()) + 3);
        config = zfp_config_rate(rate, true);
        break;
      }
      case 1:
      {
        double precision = 1; //TODO
        zfp_config_precision(precision);
        break;
      }
      case 2:
      {
        double tolerance = 1; //TODO
        zfp_config_accuracy(tolerance);
        break;
      }
      case 3:
      {
        zfp_config_reversible();
        break;
      }
      case 4:
      {
        //TODO: do we need this one?
        //zfp_config_expert(uint minbits, uint maxbits, uint maxprec, int minexp);
        break;
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
