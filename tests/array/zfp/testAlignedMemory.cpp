#include "zfp/array3.hpp"
using namespace zfp;

#include "gtest/gtest.h"
#include "../utils/gtestTestEnv.h"
#include "../utils/gtestSingleFixture.h"
#include "../utils/predicates.h"

#include <stdint.h>

TestEnv* const testEnv = new TestEnv;

class AlignedMemoryTest : public TestFixture {};

#define TEST_FIXTURE AlignedMemoryTest

INSTANTIATE_TEST_SUITE_P(TestManyMemoryAlignments, TEST_FIXTURE, ::testing::Range(4, 11));

TEST_P(TEST_FIXTURE, when_allocateAlignedMem_expect_addressAligned)
{
  size_t alignmentBytes = (size_t)(1u << GetParam());
  void* ptr = allocate_aligned(30, alignmentBytes);

  uintptr_t address = (uintptr_t)ptr;
  EXPECT_EQ(address % alignmentBytes, 0);

  deallocate_aligned(ptr);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  static_cast<void>(::testing::AddGlobalTestEnvironment(testEnv));
  return RUN_ALL_TESTS();
}
