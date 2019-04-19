/*
 * sst_file_filter_reader_test.cc
 *
 *  Created on: Dec 10, 2018
 *      Author: wonki
 */



#include <inttypes.h>
#include <chrono>
#include <ctime>
#include <sstream>
#include <fstream>

#include "accelerator/cuda/filter.h"
#include "accelerator/avx/filter.h"
#include "rocksdb/sst_file_filter_reader.h"
#include "rocksdb/sst_file_writer.h"
#include "table/block.h"
#include "util/testharness.h"
#include "util/testutil.h"
#include "utilities/merge_operators.h"

#define VALUE_RANGE 100000
#define GPU_LIMIT_NUM_OF_KEY 60000000

namespace rocksdb {

enum class FilterType {
  CPU_ITERATOR, AVX_ITERATOR, THRUST_ITERATOR, CPU_BLOCK, GPU_BLOCK,
};

inline std::string toStringFilterType(FilterType type) {
  switch (type) {
    case FilterType::CPU_ITERATOR:
      return "CPU_ITERATOR";
    case FilterType::AVX_ITERATOR:
      return "AVX_ITERATOR";
    case FilterType::THRUST_ITERATOR:
      return "THRUST_ITERATOR";
    case FilterType::CPU_BLOCK:
      return "CPU_BLOCK";
    case FilterType::GPU_BLOCK:
      return "GPU_BLOCK";
    default:
      return "";
  }
}

template <typename T>
std::string serialize(const T * data) {
	std::string d(sizeof(T), L'\0');
	memcpy(&d[0], data, d.size());
	return d;
}

std::string serializeVector(std::vector<int> v) {
	std::ostringstream oss;
	std::copy(v.begin(), v.end() - 1, std::ostream_iterator<int>(oss, " "));
	oss << v.back();
	return oss.str();
}


template <typename T>
std::unique_ptr<T> deserialize(const std::string & data) {
	if(data.size() != sizeof(T))
		return nullptr;
	auto d = std::unique_ptr<T>();
	memcpy(d.get(), data.data(), data.size());
	return d;
}

std::vector<int> deserializeVector(std::string s) {
	std::istringstream is(s);
	std::vector<int> v((std::istream_iterator<int>(is)), std::istream_iterator<int>());
	return v;
}

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

class SstFileFilterAnalyzer {
 public:
  SstFileFilterAnalyzer(std::string file_name)
    : csv_file_name(file_name)
  {
    csv_file.open(csv_file_name);
    // Columns
    csv_file << "NumEntries" << ","
        << "FilterPercent" << ","
        << "GetDataExecutionTime" << ","
        << "FilterExecutionTime" << ","
        << "TotalExecutionTime" << ","
        << std::endl;
  }

  ~SstFileFilterAnalyzer() {
    csv_file.close();
  }

  void writeLine(size_t num_keys, long pivot, FilterType filter_type,
                 std::chrono::duration<double, std::milli> get_time,
                 std::chrono::duration<double, std::milli> filter_time,
                 std::chrono::duration<double, std::milli> total_time) {
    csv_file << num_keys << ","
        << (((double) pivot / VALUE_RANGE) * 100) << ","
        << toStringFilterType(filter_type) << ","
        << get_time.count() << ","
        << filter_time.count() << ","
        << total_time.count() << ","
        << std::endl;
  }

  std::string csv_file_name;
  std::ofstream csv_file;
};

class SstFileFilterReaderTest : public testing::Test {
 public:
  SstFileFilterReaderTest() : analyzer("sst_file_filter_reader_test_results.csv") {
    std::cout << "Constructor" << std::endl;
    options_.merge_operator = MergeOperators::CreateUInt64AddOperator();
  }

  void FileWrite(uint64_t kNumKeys, std::string &file_name) {
	  SstFileWriter writer(soptions_, options_);
	  srand((unsigned int)time(NULL));
	  ASSERT_OK(writer.Open(file_name));
	  for (size_t i = 0; i < kNumKeys; i ++) {
	  	ASSERT_OK(writer.Put(
          EncodeAsUint64(i), EncodeAsUint64(rand() % VALUE_RANGE)));
	  }
	  ASSERT_OK(writer.Finish());
  }

  void GetDataBlocks(std::vector<char>& data,
                     std::vector<uint64_t>& seek_indices,
                     size_t &results_count,
                     std::string &file_name) {
    ReadOptions ropts;
    SstFileFilterReader reader(options_);
    reader.Open(file_name);
    reader.VerifyChecksum();
    reader.GetDataBlocks(ropts, data, seek_indices);
    results_count = reader.GetTableProperties()->num_entries;
  }

  void FilterDataBlocksOnCpu(std::vector<char>& data,
                             std::vector<uint64_t>& seek_indices,
                             accelerator::FilterContext& ctx,
                             std::vector<Slice> keys,
                             std::vector<Slice> values) {
    // Decode
    // TODO(totoro): Implements this logics to DataBulkCpuIter.
    std::cout << "[FilterDataBlocksOnCpu]" << std::endl;
    const char *start = &data[0];
    for (size_t i = 0; i < seek_indices.size(); ++i) {
      const char *limit = (i == seek_indices.size() - 1)
          ? start + data.size()
          : start + seek_indices[i + 1];
      const char *subblock = start + seek_indices[i];
      while (subblock < limit) {
        uint32_t shared, non_shared, value_length;
        Slice key;
        subblock = DecodeEntry()(subblock, limit, &shared, &non_shared,
                                &value_length);
        if (shared == 0) {
          key = Slice(subblock, non_shared);
        } else {
          // TODO(totoro): We need to consider 'shared' data within subblock.
          key = Slice(subblock, shared + non_shared);
        }

        Slice value = Slice(subblock + non_shared, value_length);

        // Next DataKey...
        uint64_t next_offset = static_cast<uint64_t>(
          (value.data() + value.size()) - start
        );

        uint64_t decoded_value = DecodeFixed64(value.data());
        int decoded_value_int = (int) decoded_value;
        bool filter_result = false;
        switch (ctx._op) {
          case accelerator::EQ:
            filter_result = decoded_value_int == ctx._pivot;
            break;
          case accelerator::LESS:
            filter_result = decoded_value_int < ctx._pivot;
            break;
          case accelerator::GREATER:
            filter_result = decoded_value_int > ctx._pivot;
            break;
          case accelerator::LESS_EQ:
            filter_result = decoded_value_int <= ctx._pivot;
            break;
          case accelerator::GREATER_EQ:
            filter_result = decoded_value_int >= ctx._pivot;
            break;
          default:
            break;
        }
        if (filter_result) {
          keys.emplace_back(std::move(key));
          values.emplace_back(std::move(value));
        }

        subblock = start + next_offset;
      }
    }
  }

  std::string fileNameGenerate(const char *prefix, size_t count) {
    std::stringstream generator;
    generator << prefix << count;
    return generator.str();
  }

  void SetUpTest(size_t num_keys) {
    sst_names_.clear();
    std::cout << "Setup start, num_keys: " << num_keys << std::endl;

    options_.comparator = test::Uint64Comparator();
    size_t file_count = 0;
    if (num_keys <= GPU_LIMIT_NUM_OF_KEY) {
      sst_names_.push_back(fileNameGenerate("sst_testfile_", ++file_count));
      std::string &sst_name = *sst_names_.rbegin();
      FileWrite(num_keys, sst_name);
      ruda::gpuWarmingUp();  // Note(totoro): Removes initial delay on GPU.
      std::cout << "Setup finished" << std::endl;
      return;
    }

    // If num_keys overflowed limitation of gpu allocatable key count,
    // split to multiple files...
    while (num_keys > GPU_LIMIT_NUM_OF_KEY) {
      sst_names_.push_back(fileNameGenerate("sst_testfile_", ++file_count));
      std::string &sst_name = *sst_names_.rbegin();
      FileWrite(GPU_LIMIT_NUM_OF_KEY, sst_name);
      num_keys -= GPU_LIMIT_NUM_OF_KEY;
    }

    // Create last file with remain num keys...
    if (num_keys > 0) {
      sst_names_.push_back(fileNameGenerate("sst_testfile_", ++file_count));
      std::string &sst_name = *sst_names_.rbegin();
      FileWrite(num_keys, sst_name);
    }

    ruda::gpuWarmingUp();  // Note(totoro): Removes initial delay on GPU.
    std::cout << "Setup finished" << std::endl;
  }

  void TestWarmingUp() {
    std::chrono::high_resolution_clock::time_point begin, end;
    options_.comparator = test::Uint64Comparator();

    std::chrono::duration<double, std::milli> duration_time(0), get_time(0);

    for (auto &file_name : sst_names_) {
      std::vector<int> keys, values;
      begin = std::chrono::high_resolution_clock::now();
      std::cout << "File name: " << file_name << std::endl;
      ReadOptions ropts;
      SstFileFilterReader reader(options_);
      reader.Open(file_name);
      reader.VerifyChecksum();
      std::unique_ptr<Iterator> iter(reader.NewIterator(ropts));
      for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
        keys.emplace_back(atoi(iter->key().data()));
        values.emplace_back(atoi(iter->value().data()));
      }
      end = std::chrono::high_resolution_clock::now();
      duration_time = end - begin;
      get_time += duration_time;
    }

    std::cout << "[WarmingUp] Execution Time: " << get_time.count()
        << std::endl;
  }

  void IteratorFilterTest(size_t num_keys, FilterType filter_type,
                          accelerator::FilterContext &ctx) {
    if (
      filter_type != FilterType::CPU_ITERATOR &&
      filter_type != FilterType::AVX_ITERATOR &&
      filter_type != FilterType::THRUST_ITERATOR
    ) {
      return;
    }

    std::chrono::high_resolution_clock::time_point begin, end;
    std::vector<int> filtered_keys, filtered_values;
    options_.comparator = test::Uint64Comparator();
    std::chrono::duration<double, std::milli> duration_time(0), get_time(0), filter_time(0), total_time(0);

    for (auto &file_name : sst_names_) {
      std::vector<long> keys, values, results;
      begin = std::chrono::high_resolution_clock::now();
      std::cout << "File name: " << file_name << std::endl;
      ReadOptions ropts;
      SstFileFilterReader reader(options_);
      reader.Open(file_name);
      reader.VerifyChecksum();
      std::unique_ptr<Iterator> iter(reader.NewIterator(ropts));
      for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
        keys.emplace_back(atoi(iter->key().data()));
        values.emplace_back(atoi(iter->value().data()));
      }
      end = std::chrono::high_resolution_clock::now();
      duration_time = end - begin;
      get_time += duration_time;

      begin = std::chrono::high_resolution_clock::now();
      switch (filter_type) {
        case FilterType::CPU_ITERATOR: {
          for (long value : values) {
            results.push_back(ctx(value));
          }
          break;
        }
        case FilterType::AVX_ITERATOR: {
          avx::simpleIntFilter(values, ctx, results);
          break;
        }
        case FilterType::THRUST_ITERATOR: {
          ruda::sstThrustFilter(values, ctx, results);
          break;
        }
        default:
          return;
      }
      for (size_t i = 0; i < results.size(); ++i) {
        if (results[i]) {
          filtered_keys.push_back(keys[i]);
          filtered_values.push_back(values[i]);
        }
      }
      end = std::chrono::high_resolution_clock::now();
      duration_time = end - begin;
      filter_time += duration_time;
    }

    total_time = get_time + filter_time;
    std::cout << "[" << toStringFilterType(filter_type) << "]"
        << "[GetValuesFromSST] Execution Time: " << get_time.count()
        << std::endl;
    std::cout << "[" << toStringFilterType(filter_type) << "]"
        << "[FilterValuesFromSST] Execution Time: "
        << filter_time.count() << std::endl;
    std::cout << "[" << toStringFilterType(filter_type) << "]"
        << " Total Execution Time: " << total_time.count()
        << std::endl;
    analyzer.writeLine(
        num_keys, ctx._pivot, filter_type, get_time, filter_time, total_time);
  }

  void BlockFilterTest(size_t num_keys, FilterType filter_type,
                       accelerator::FilterContext &ctx) {
    if (
      filter_type != FilterType::CPU_BLOCK &&
      filter_type != FilterType::GPU_BLOCK
    ) {
      return;
    }

    std::chrono::high_resolution_clock::time_point begin, end;
    options_.comparator = test::Uint64Comparator();
    std::vector<Slice> keys, values;
    std::chrono::duration<double, std::milli> duration_time(0), get_time(0), filter_time(0), total_time(0);

    for (auto &file_name : sst_names_) {
      std::vector<char> data;
      std::vector<uint64_t> seek_indices;
      size_t results_count;

      begin = std::chrono::high_resolution_clock::now();
      GetDataBlocks(data, seek_indices, results_count, file_name);
      end = std::chrono::high_resolution_clock::now();

      duration_time = end - begin;
      get_time += duration_time;
      begin = std::chrono::high_resolution_clock::now();
      switch (filter_type) {
        case FilterType::CPU_BLOCK: {
          FilterDataBlocksOnCpu(data, seek_indices, ctx, keys, values);
          break;
        }
        case FilterType::GPU_BLOCK: {
          ruda::sstStreamIntBlockFilter(
              data, seek_indices, ctx, results_count, keys, values);
          break;
        }
        default:
          return;
      }
      end = std::chrono::high_resolution_clock::now();
      duration_time = end - begin;
      filter_time += duration_time;
    }

    total_time = get_time + filter_time;

    std::cout << "[" << toStringFilterType(filter_type) << "]"
        << "[GetDataBlocks] Execution Time: " << get_time.count()
        << std::endl;
    std::cout << "[" << toStringFilterType(filter_type) << "]"
        << "[FilterAndDecodeDataBlocks] Execution Time: "
        << filter_time.count() << std::endl;
    std::cout << "[" << toStringFilterType(filter_type) << "]"
        << " Total Execution Time: " << total_time.count()
        << std::endl;
    analyzer.writeLine(
        num_keys, ctx._pivot, filter_type, get_time, filter_time, total_time);
  }

 protected:
  Options options_;
  EnvOptions soptions_;

  std::vector<std::string> sst_names_;
  SstFileFilterAnalyzer analyzer;
};

TEST_F(SstFileFilterReaderTest, Filter_1_000_000) {
  size_t num_keys = 1000000;
  SetUpTest(num_keys);

  std::cout << "//////////////////////////////////////////////////////" << std::endl
      << "[TEST][Filter_1_000_000]" << std::endl
      << "//////////////////////////////////////////////////////" << std::endl;
  {
    // 0% Filtered
    std::cout << "_____________________________________________________" << std::endl
        << "[TEST][Filter_1_000_000] 0\% Filtered" << std::endl;
    long pivot = VALUE_RANGE * 0;
    accelerator::FilterContext ctx = { accelerator::LESS, pivot,};
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::CPU_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::AVX_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::THRUST_ITERATOR, ctx);
    BlockFilterTest(num_keys, FilterType::GPU_BLOCK, ctx);
  }
  {
    // 20% Filtered
    std::cout << "_____________________________________________________" << std::endl
        << "[TEST][Filter_1_000_000] 20\% Filtered" << std::endl;
    long pivot = VALUE_RANGE * 0.2;
    accelerator::FilterContext ctx = { accelerator::LESS, pivot,};
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::CPU_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::AVX_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::THRUST_ITERATOR, ctx);
    BlockFilterTest(num_keys, FilterType::GPU_BLOCK, ctx);
  }
  {
    // 40% Filtered
    std::cout << "_____________________________________________________" << std::endl
        << "[TEST][Filter_1_000_000] 40\% Filtered" << std::endl;
    long pivot = VALUE_RANGE * 0.4;
    accelerator::FilterContext ctx = { accelerator::LESS, pivot,};
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::CPU_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::AVX_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::THRUST_ITERATOR, ctx);
    BlockFilterTest(num_keys, FilterType::GPU_BLOCK, ctx);
  }
  {
    // 60% Filtered
    std::cout << "_____________________________________________________" << std::endl
        << "[TEST][Filter_1_000_000] 60\% Filtered" << std::endl;
    long pivot = VALUE_RANGE * 0.6;
    accelerator::FilterContext ctx = { accelerator::LESS, pivot,};
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::CPU_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::AVX_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::THRUST_ITERATOR, ctx);
    BlockFilterTest(num_keys, FilterType::GPU_BLOCK, ctx);
  }
  {
    // 80% Filtered
    std::cout << "_____________________________________________________" << std::endl
        << "[TEST][Filter_1_000_000] 80\% Filtered" << std::endl;
    long pivot = VALUE_RANGE * 0.8;
    accelerator::FilterContext ctx = { accelerator::LESS, pivot,};
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::CPU_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::AVX_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::THRUST_ITERATOR, ctx);
    BlockFilterTest(num_keys, FilterType::GPU_BLOCK, ctx);
  }
  {
    // 100% Filtered
    std::cout << "_____________________________________________________" << std::endl
        << "[TEST][Filter_1_000_000] 100\% Filtered" << std::endl;
    long pivot = VALUE_RANGE * 1;
    accelerator::FilterContext ctx = { accelerator::LESS, pivot,};
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::CPU_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::AVX_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::THRUST_ITERATOR, ctx);
    BlockFilterTest(num_keys, FilterType::GPU_BLOCK, ctx);
  }
}

TEST_F(SstFileFilterReaderTest, Filter_10_000_000) {
  size_t num_keys = 10000000;
  SetUpTest(num_keys);

  std::cout << "//////////////////////////////////////////////////////" << std::endl
      << "[TEST][Filter_10_000_000]" << std::endl
      << "//////////////////////////////////////////////////////" << std::endl;
  {
    // 0% Filtered
    std::cout << "_____________________________________________________" << std::endl
        << "[TEST][Filter_10_000_000] 0\% Filtered" << std::endl;
    long pivot = VALUE_RANGE * 0;
    accelerator::FilterContext ctx = { accelerator::LESS, pivot,};
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::CPU_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::AVX_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::THRUST_ITERATOR, ctx);
    BlockFilterTest(num_keys, FilterType::GPU_BLOCK, ctx);
  }
  {
    // 20% Filtered
    std::cout << "_____________________________________________________" << std::endl
        << "[TEST][Filter_10_000_000] 20\% Filtered" << std::endl;
    long pivot = VALUE_RANGE * 0.2;
    accelerator::FilterContext ctx = { accelerator::LESS, pivot,};
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::CPU_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::AVX_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::THRUST_ITERATOR, ctx);
    BlockFilterTest(num_keys, FilterType::GPU_BLOCK, ctx);
  }
  {
    // 40% Filtered
    std::cout << "_____________________________________________________" << std::endl
        << "[TEST][Filter_10_000_000] 40\% Filtered" << std::endl;
    long pivot = VALUE_RANGE * 0.4;
    accelerator::FilterContext ctx = { accelerator::LESS, pivot,};
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::CPU_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::AVX_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::THRUST_ITERATOR, ctx);
    BlockFilterTest(num_keys, FilterType::GPU_BLOCK, ctx);
  }
  {
    // 60% Filtered
    std::cout << "_____________________________________________________" << std::endl
        << "[TEST][Filter_10_000_000] 60\% Filtered" << std::endl;
    long pivot = VALUE_RANGE * 0.6;
    accelerator::FilterContext ctx = { accelerator::LESS, pivot,};
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::CPU_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::AVX_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::THRUST_ITERATOR, ctx);
    BlockFilterTest(num_keys, FilterType::GPU_BLOCK, ctx);
  }
  {
    // 80% Filtered
    std::cout << "_____________________________________________________" << std::endl
        << "[TEST][Filter_10_000_000] 80\% Filtered" << std::endl;
    long pivot = VALUE_RANGE * 0.8;
    accelerator::FilterContext ctx = { accelerator::LESS, pivot,};
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::CPU_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::AVX_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::THRUST_ITERATOR, ctx);
    BlockFilterTest(num_keys, FilterType::GPU_BLOCK, ctx);
  }
  {
    // 100% Filtered
    std::cout << "_____________________________________________________" << std::endl
        << "[TEST][Filter_10_000_000] 100\% Filtered" << std::endl;
    long pivot = VALUE_RANGE * 1;
    accelerator::FilterContext ctx = { accelerator::LESS, pivot,};
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::CPU_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::AVX_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::THRUST_ITERATOR, ctx);
    BlockFilterTest(num_keys, FilterType::GPU_BLOCK, ctx);
  }
}

TEST_F(SstFileFilterReaderTest, Filter_100_000_000) {
  size_t num_keys = 100000000;
  SetUpTest(num_keys);

  std::cout << "//////////////////////////////////////////////////////" << std::endl
      << "[TEST][Filter_100_000_000]" << std::endl
      << "//////////////////////////////////////////////////////" << std::endl;
  {
    // 0% Filtered
    std::cout << "_____________________________________________________" << std::endl
        << "[TEST][Filter_100_000_000] 0\% Filtered" << std::endl;
    long pivot = VALUE_RANGE * 0;
    accelerator::FilterContext ctx = { accelerator::LESS, pivot,};
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::CPU_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::AVX_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::THRUST_ITERATOR, ctx);
    BlockFilterTest(num_keys, FilterType::GPU_BLOCK, ctx);
  }
  {
    // 20% Filtered
    std::cout << "_____________________________________________________" << std::endl
        << "[TEST][Filter_100_000_000] 20\% Filtered" << std::endl;
    long pivot = VALUE_RANGE * 0.2;
    accelerator::FilterContext ctx = { accelerator::LESS, pivot,};
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::CPU_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::AVX_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::THRUST_ITERATOR, ctx);
    BlockFilterTest(num_keys, FilterType::GPU_BLOCK, ctx);
  }
  {
    // 40% Filtered
    std::cout << "_____________________________________________________" << std::endl
        << "[TEST][Filter_100_000_000] 40\% Filtered" << std::endl;
    long pivot = VALUE_RANGE * 0.4;
    accelerator::FilterContext ctx = { accelerator::LESS, pivot,};
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::CPU_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::AVX_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::THRUST_ITERATOR, ctx);
    BlockFilterTest(num_keys, FilterType::GPU_BLOCK, ctx);
  }
  {
    // 60% Filtered
    std::cout << "_____________________________________________________" << std::endl
        << "[TEST][Filter_100_000_000] 60\% Filtered" << std::endl;
    long pivot = VALUE_RANGE * 0.6;
    accelerator::FilterContext ctx = { accelerator::LESS, pivot,};
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::CPU_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::AVX_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::THRUST_ITERATOR, ctx);
    BlockFilterTest(num_keys, FilterType::GPU_BLOCK, ctx);
  }
  {
    // 80% Filtered
    std::cout << "_____________________________________________________" << std::endl
        << "[TEST][Filter_100_000_000] 80\% Filtered" << std::endl;
    long pivot = VALUE_RANGE * 0.8;
    accelerator::FilterContext ctx = { accelerator::LESS, pivot,};
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::CPU_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::AVX_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::THRUST_ITERATOR, ctx);
    BlockFilterTest(num_keys, FilterType::GPU_BLOCK, ctx);
  }
  {
    // 100% Filtered
    std::cout << "_____________________________________________________" << std::endl
        << "[TEST][Filter_100_000_000] 100\% Filtered" << std::endl;
    long pivot = VALUE_RANGE * 1;
    accelerator::FilterContext ctx = { accelerator::LESS, pivot,};
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::CPU_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::AVX_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::THRUST_ITERATOR, ctx);
    BlockFilterTest(num_keys, FilterType::GPU_BLOCK, ctx);
  }
}

TEST_F(SstFileFilterReaderTest, Filter_1_000_000_000) {
  size_t num_keys = 1000000000;
  SetUpTest(num_keys);

  std::cout << "//////////////////////////////////////////////////////" << std::endl
      << "[TEST][Filter_1_000_000_000]" << std::endl
      << "//////////////////////////////////////////////////////" << std::endl;
  {
    // 0% Filtered
    std::cout << "_____________________________________________________" << std::endl
        << "[TEST][Filter_1_000_000_000] 0\% Filtered" << std::endl;
    long pivot = VALUE_RANGE * 0;
    accelerator::FilterContext ctx = { accelerator::LESS, pivot,};
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::CPU_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::AVX_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::THRUST_ITERATOR, ctx);
    BlockFilterTest(num_keys, FilterType::GPU_BLOCK, ctx);
  }
  {
    // 20% Filtered
    std::cout << "_____________________________________________________" << std::endl
        << "[TEST][Filter_1_000_000_000] 20\% Filtered" << std::endl;
    long pivot = VALUE_RANGE * 0.2;
    accelerator::FilterContext ctx = { accelerator::LESS, pivot,};
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::CPU_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::AVX_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::THRUST_ITERATOR, ctx);
    BlockFilterTest(num_keys, FilterType::GPU_BLOCK, ctx);
  }
  {
    // 40% Filtered
    std::cout << "_____________________________________________________" << std::endl
        << "[TEST][Filter_1_000_000_000] 40\% Filtered" << std::endl;
    long pivot = VALUE_RANGE * 0.4;
    accelerator::FilterContext ctx = { accelerator::LESS, pivot,};
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::CPU_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::AVX_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::THRUST_ITERATOR, ctx);
    BlockFilterTest(num_keys, FilterType::GPU_BLOCK, ctx);
  }
  {
    // 60% Filtered
    std::cout << "_____________________________________________________" << std::endl
        << "[TEST][Filter_1_000_000_000] 60\% Filtered" << std::endl;
    long pivot = VALUE_RANGE * 0.6;
    accelerator::FilterContext ctx = { accelerator::LESS, pivot,};
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::CPU_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::AVX_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::THRUST_ITERATOR, ctx);
    BlockFilterTest(num_keys, FilterType::GPU_BLOCK, ctx);
  }
  {
    // 80% Filtered
    std::cout << "_____________________________________________________" << std::endl
        << "[TEST][Filter_1_000_000_000] 80\% Filtered" << std::endl;
    long pivot = VALUE_RANGE * 0.8;
    accelerator::FilterContext ctx = { accelerator::LESS, pivot,};
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::CPU_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::AVX_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::THRUST_ITERATOR, ctx);
    BlockFilterTest(num_keys, FilterType::GPU_BLOCK, ctx);
  }
  {
    // 100% Filtered
    std::cout << "_____________________________________________________" << std::endl
        << "[TEST][Filter_1_000_000_000] 100\% Filtered" << std::endl;
    long pivot = VALUE_RANGE * 1;
    accelerator::FilterContext ctx = { accelerator::LESS, pivot,};
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::CPU_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::AVX_ITERATOR, ctx);
    TestWarmingUp();
    IteratorFilterTest(num_keys, FilterType::THRUST_ITERATOR, ctx);
    BlockFilterTest(num_keys, FilterType::GPU_BLOCK, ctx);
  }
}

// TEST_F(SstFileFilterReaderTest, FilterOnCpuBlock) {
//   std::chrono::high_resolution_clock::time_point begin, end;

//   options_.comparator = test::Uint64Comparator();
//   std::vector<char> data;
//   std::vector<uint64_t> seek_indices;
//   size_t results_count;

//   begin = std::chrono::high_resolution_clock::now();
//   GetDataBlocks(data, seek_indices, results_count);
//   end = std::chrono::high_resolution_clock::now();

//   std::chrono::duration<float, std::milli> elapsed = end - begin;
//   std::cout << "[CPU][GetDataBlocks] Execution Time: " << elapsed.count()
//       << std::endl;
//   accelerator::FilterContext ctx = { accelerator::LESS, 50,};
//   std::vector<Slice> keys, values;
//   begin = std::chrono::high_resolution_clock::now();
//   FilterDataBlocksOnCpu(data, seek_indices, ctx, keys, values);
//   end = std::chrono::high_resolution_clock::now();
//   elapsed = end - begin;
//   std::cout << "[CPU][FilterAndDecodeDataBlocksOnCpu] Execution Time: "
//       << elapsed.count() << std::endl;
// }

// TEST_F(SstFileFilterReaderTest, FilterOnThrust) {
//   std::chrono::high_resolution_clock::time_point begin, end;

//   std::vector<int> filtered_keys, filtered_values;
//   accelerator::FilterContext ctx = { accelerator::LESS, 50,};

//   options_.comparator = test::Uint64Comparator();

//   std::chrono::duration<float, std::milli> get_time, filter_time, total_time;

//   for (auto &file_name : sst_names_) {
//     std::vector<int> keys, values, results;
//     begin = std::chrono::high_resolution_clock::now();
//     std::cout << "File name: " << file_name << std::endl;
//     ReadOptions ropts;
//     SstFileFilterReader reader(options_);
//     reader.Open(file_name);
//     reader.VerifyChecksum();
//     std::unique_ptr<Iterator> iter(reader.NewIterator(ropts));
//     for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
//       keys.emplace_back(atoi(iter->key().data()));
//       values.emplace_back(atoi(iter->value().data()));
//     }
//     end = std::chrono::high_resolution_clock::now();
//     get_time += (end - begin);

//     begin = std::chrono::high_resolution_clock::now();
//     ruda::sstThrustFilter(values, ctx, results);
//     for (size_t i = 0; i < results.size(); ++i) {
//       if (results[i]) {
//         filtered_keys.push_back(keys[i]);
//         filtered_values.push_back(values[i]);
//       }
//     }
//     end = std::chrono::high_resolution_clock::now();
//     filter_time += (end - begin);
//   }

//   std::cout << "[THRUST][GetValuesFromSST] Execution Time: " << get_time.count()
//       << std::endl;
//   std::cout << "[THRUST][FilterValuesFromSST] Execution Time: "
//       << filter_time.count() << std::endl;
//   total_time = get_time + filter_time;
//   std::cout << "[THRUST] Total Execution Time: " << total_time.count()
//       << std::endl;
// }

// TEST_F(SstFileFilterReaderTest, FilterOnGpu) {
//   std::chrono::high_resolution_clock::time_point begin, end;

//   options_.comparator = test::Uint64Comparator();
//   std::vector<Slice> keys, values;
//   accelerator::FilterContext ctx = { accelerator::LESS, 50,};

//   std::chrono::duration<float, std::milli> get_time, filter_time, total_time;

//   for (auto &file_name : sst_names_) {
//     std::vector<char> data;
//     std::vector<uint64_t> seek_indices;
//     size_t results_count;

//     begin = std::chrono::high_resolution_clock::now();
//     GetDataBlocks(data, seek_indices, results_count, file_name);
//     end = std::chrono::high_resolution_clock::now();

//     get_time += end - begin;
//     begin = std::chrono::high_resolution_clock::now();
//     FilterDataBlocksOnGpu(data, seek_indices, ctx, results_count, keys, values);
//     end = std::chrono::high_resolution_clock::now();
//     filter_time += end - begin;
//   }

//   std::cout << "[GPU][GetDataBlocks] Execution Time: " << get_time.count()
//         << std::endl;
//   std::cout << "[GPU][FilterAndDecodeDataBlocksOnGpu] Execution Time: "
//         << filter_time.count() << std::endl;
//   total_time = get_time + filter_time;
//   std::cout << "[GPU] Total Execution Time: " << total_time.count()
//       << std::endl;

//   // std::cout << "Filter Results" << std::endl;
//   // for (size_t i = 0; i < keys.size(); ++i) {
//   //   std::cout << "keys[" << DecodeFixed64(keys[i].data())
//   //       << "] values[" << DecodeFixed64(values[i].data()) << "]"
//   //       << std::endl;
//   // }
// }

// TEST_F(SstFileFilterReaderTest, FilterOnAvx) {
//   std::chrono::high_resolution_clock::time_point begin, end;

//   std::vector<long> filtered_keys, filtered_values;
//   accelerator::FilterContext ctx = { accelerator::LESS, 50,};

//   options_.comparator = test::Uint64Comparator();

//   std::chrono::duration<float, std::milli> get_time, filter_time, total_time;

//   for (auto &file_name : sst_names_) {
//     std::vector<long> keys, values, results;
//     begin = std::chrono::high_resolution_clock::now();
//     std::cout << "File name: " << file_name << std::endl;
//     ReadOptions ropts;
//     SstFileFilterReader reader(options_);
//     reader.Open(file_name);
//     reader.VerifyChecksum();
//     std::unique_ptr<Iterator> iter(reader.NewIterator(ropts));
//     for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
//       keys.emplace_back(atoi(iter->key().data()));
//       values.emplace_back(atoi(iter->value().data()));
//     }
//     end = std::chrono::high_resolution_clock::now();
//     get_time += (end - begin);

//     begin = std::chrono::high_resolution_clock::now();
//     avx::simpleIntFilter(values, ctx, results);
//     for (size_t i = 0; i < results.size(); ++i) {
//       if (results[i]) {
//         filtered_keys.push_back(keys[i]);
//         filtered_values.push_back(values[i]);
//       }
//     }
//     end = std::chrono::high_resolution_clock::now();
//     filter_time += (end - begin);
//   }

//   std::cout << "[AVX][GetValuesFromSST] Execution Time: " << get_time.count()
//       << std::endl;
//   std::cout << "[AVX][FilterValuesFromSST] Execution Time: "
//       << filter_time.count() << std::endl;
//   total_time = get_time + filter_time;
//   std::cout << "[AVX] Total Execution Time: " << total_time.count()
//       << std::endl;

//   // std::cout << "Filter Results" << std::endl;
//   // for (size_t i = 0; i < values.size(); ++i) {
//   //   std::cout << "values[" << values[i]
//   //       << "] results[" << results[i] << "]"
//   //       << std::endl;
//   // }
// }

// TEST_F(SstFileFilterReaderTest, RecordFilterOnGpu) {
//   std::cout << "[GPU][RecordFilterOnGpu] START" << std::endl;
//   int point = 0;
//   std::chrono::high_resolution_clock::time_point begin, end;

//   std::cout << "[GPU][RecordFilterOnGpu] POINT " << point++ << std::endl;
//   accelerator::FilterContext ctx = { accelerator::LESS_EQ, 50,};
//   std::vector<uint> schema_type, schema_length;
//   SlicewithSchema schema("123", 3, ctx, 0, schema_type, schema_length);
//   std::cout << "[GPU][RecordFilterOnGpu] POINT " << point++ << std::endl;

//   options_.comparator = test::Uint64Comparator();
//   std::vector<char> data;
//   std::vector<uint64_t> seek_indices;
//   size_t results_count;
//   begin = std::chrono::high_resolution_clock::now();
//   GetDataBlocks(data, seek_indices, results_count);
//   end = std::chrono::high_resolution_clock::now();
//   std::chrono::duration<float, std::milli> elapsed = end - begin;
//   std::cout << "[GPU][GetDataBlocks] Execution Time: " << elapsed.count()
//       << std::endl;
//   std::vector<PinnableSlice> values;
//   begin = std::chrono::high_resolution_clock::now();
//   ruda::recordBlockFilter(data, seek_indices, schema, results_count, values);
//   end = std::chrono::high_resolution_clock::now();
//   elapsed = end - begin;
//   std::cout << "[GPU][ruda::recordBlockFilter] Execution Time: "
//       << elapsed.count() << std::endl;

//   // std::cout << "Filter Results" << std::endl;
//   // for (size_t i = 0; i < values.size(); ++i) {
//   //   std::cout << "values[" << values[i]
//   //       << "] results[" << results[i] << "]"
//   //       << std::endl;
//   // }
// }

}  // namespace rocksdb

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

