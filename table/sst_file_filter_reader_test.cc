/*
 * sst_file_filter_reader_test.cc
 *
 *  Created on: Dec 10, 2018
 *      Author: wonki
 */



#include <inttypes.h>
#include <ctime>

#include "rocksdb/sst_file_filter_reader.h"
#include "rocksdb/sst_file_writer.h"
#include "util/testharness.h"
#include "util/testutil.h"
#include "utilities/merge_operators.h"

namespace rocksdb {

std::string EncodeAsString(uint64_t v) {
  char buf[16];
  snprintf(buf, sizeof(buf), "%08" PRIu64, v);
  return std::string(buf);
}

std::string EncodeAsUint64(uint64_t v) {
  std::string dst;
  PutFixed64(&dst, v);
  return dst;
}

int filterPredicateCPU(const int target, ruda::ConditionContext ctx) {
  switch (ctx._op) {
    case 0:
      return target == ctx._pivot ? 1 : 0;
    case 1:
      return target < ctx._pivot ? 1 : 0;
    case 2:
      return target > ctx._pivot ? 1 : 0;
    case 3:
      return target <= ctx._pivot ? 1 : 0;
    case 4:
      return target >= ctx._pivot ? 1 : 0;
    default:
      return 0;
  }
}

class SstFileFilterReaderTest : public testing::Test {
 public:
  SstFileFilterReaderTest() {
    options_.merge_operator = MergeOperators::CreateUInt64AddOperator();
    sst_name_ = test::PerThreadDBPath("sst_file");
  }

  void CreateFileAndCheck(const std::vector<std::string>& keys) {
    SstFileWriter writer(soptions_, options_);
    ASSERT_OK(writer.Open(sst_name_));
    for (size_t i = 0; i + 2 < keys.size(); i += 3) {
      ASSERT_OK(writer.Put(keys[i], keys[i]));
      ASSERT_OK(writer.Merge(keys[i+1], EncodeAsUint64(i+1)));
      ASSERT_OK(writer.Delete(keys[i+2]));
    }
    ASSERT_OK(writer.Finish());

    ReadOptions ropts;
    SstFileFilterReader reader(options_);
    ASSERT_OK(reader.Open(sst_name_));
    ASSERT_OK(reader.VerifyChecksum());
    std::unique_ptr<Iterator> iter(reader.NewIterator(ropts));
    iter->SeekToFirst();
    for (size_t i = 0; i + 2 < keys.size(); i += 3) {
      ASSERT_TRUE(iter->Valid());
      ASSERT_EQ(iter->key().compare(keys[i]), 0);
      ASSERT_EQ(iter->value().compare(keys[i]), 0);
      iter->Next();
      ASSERT_TRUE(iter->Valid());
      ASSERT_EQ(iter->key().compare(keys[i+1]), 0);
      ASSERT_EQ(iter->value().compare(EncodeAsUint64(i+1)), 0);
      iter->Next();
    }
    ASSERT_FALSE(iter->Valid());
  }

  void FilterWithCPU(std::vector<std::string> &keys, ruda::ConditionContext ctx, std::vector<int> &results) {
	SstFileWriter writer(soptions_, options_);
	srand((unsigned int)time(NULL));
	ASSERT_OK(writer.Open(sst_name_));
	for (size_t i = 0; i < keys.size(); i ++) {
		ASSERT_OK(writer.Put(keys[i], EncodeAsUint64(rand() % 100)));
	}
	ASSERT_OK(writer.Finish());

	ReadOptions ropts;
    SstFileFilterReader reader(options_);
    reader.Open(sst_name_);
    reader.VerifyChecksum();

    std::unique_ptr<Iterator> iter(reader.NewIterator(ropts));
    iter->SeekToFirst();
    for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    	iter->Valid();
    	results.emplace_back(filterPredicateCPU(atoi(iter->value().data()), ctx));
      }

  }

//  void FilterWithGPU(const std::vector<std::string>& keys) {
//  }

 protected:
  Options options_;
  EnvOptions soptions_;
  std::string sst_name_;
};

const uint64_t kNumKeys = 100;

TEST_F(SstFileFilterReaderTest, Basic) {
  std::vector<std::string> keys;
  for (uint64_t i = 0; i < kNumKeys; i++) {
    keys.emplace_back(EncodeAsString(i));
  }
  CreateFileAndCheck(keys);
}

TEST_F(SstFileFilterReaderTest, Uint64Comparator) {
  options_.comparator = test::Uint64Comparator();
  std::vector<std::string> keys;
  for (uint64_t i = 0; i < kNumKeys; i++) {
    keys.emplace_back(EncodeAsUint64(i));
  }
  CreateFileAndCheck(keys);
}

TEST_F(SstFileFilterReaderTest, FilterTestWithCPU) {
  options_.comparator = test::Uint64Comparator();
  std::vector<std::string> keys;
  for (uint64_t i = 0; i < kNumKeys; i++) {
	keys.emplace_back(EncodeAsUint64(i));
  }
  ruda::ConditionContext ctx = { ruda::EQ, 5,};
  std::vector<int> results;
  FilterWithCPU(keys, ctx, results);

  std::cout << "[FILTER_TEST] Results" << std::endl;
  for (unsigned int i = 0; i < results.size(); ++i) {
    std::cout << results[i] << " ";
  }
  std::cout << std::endl;
}

//TEST_F(SstFileFilterReaderTest, FilterTestWithGPU) {
//  options_.comparator = test::Uint64Comparator();
//  std::vector<std::string> keys;
//  for (uint64_t i = 0; i < kNumKeys; i++) {
//    keys.emplace_back(EncodeAsUint64(i));
//  }
//  FilterWithGPU(keys);
//}

}  // namespace rocksdb

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

