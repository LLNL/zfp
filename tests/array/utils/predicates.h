#include "zfp/array1.hpp"

#include "gtest/gtest.h"

testing::AssertionResult ExpectEqPrintHexPred(const char* expected_expr, const char* actual_expr, uint64 expected, uint64 actual)
{
  if (actual == expected)
    return testing::AssertionSuccess();

  std::stringstream ss, msg;
  std::string expected_str, actual_str;

  ss.str("");
  ss << std::showbase << std::hex << expected;
  expected_str = ss.str();

  ss.str("");
  ss << std::showbase << std::hex << actual;
  actual_str = ss.str();

  msg << "\t  Expected: " << expected_expr;
  if (expected_str != expected_expr) {
    msg << "\n\t  Which is: " << std::showbase << std::hex << expected;
  }
  msg << "\nTo be equal to: " << actual_expr;
  if (actual_str != actual_expr) {
    msg << "\n\t  Which is: " << std::showbase << std::hex << actual;
  }

  return testing::AssertionFailure() << msg.str();
}

testing::AssertionResult ExpectNeqPrintHexPred(const char* expected_expr, const char* actual_expr, uint64 expected, uint64 actual)
{
  if (actual != expected)
    return testing::AssertionSuccess();

  std::stringstream ss, msg;
  std::string expected_str, actual_str;

  ss.str("");
  ss << std::showbase << std::hex << expected;
  expected_str = ss.str();

  ss.str("");
  ss << std::showbase << std::hex << actual;
  actual_str = ss.str();

  msg << "\t  Expected: " << expected_expr;
  if (expected_str != expected_expr) {
    msg << "\n\t  Which is: " << std::showbase << std::hex << expected;
  }
  msg << "\nNot to be equal to: " << actual_expr;
  if (actual_str != actual_expr) {
    msg << "\n\t  Which is: " << std::showbase << std::hex << actual;
  }

  return testing::AssertionFailure() << msg.str();
}
