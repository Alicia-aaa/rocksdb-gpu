/*
 * sst_file_filter_reader.cc
 *
 *  Created on: Dec 10, 2018
 *      Author: wonki
 */


#include "rocksdb/sst_file_filter_reader.h"

#include "db/db_iter.h"
#include "options/cf_options.h"
#include "table/get_context.h"
#include "table/table_reader.h"
#include "table/table_builder.h"
#include "util/file_reader_writer.h"

namespace rocksdb {
struct SstFileFilterReader::Rep {
  Options options;
  EnvOptions soptions;
  ImmutableCFOptions ioptions;
  MutableCFOptions moptions;

  std::unique_ptr<TableReader> table_reader;

  Rep(const Options& opts)
      : options(opts),
        soptions(options),
        ioptions(options),
        moptions(ColumnFamilyOptions(options)) {}
};

SstFileFilterReader::SstFileFilterReader(const Options& options)
    : rep_(new Rep(options)) {}

SstFileFilterReader::~SstFileFilterReader() {}

Status SstFileFilterReader::Open(const std::string& file_path) {
  auto r = rep_.get();
  Status s;
  uint64_t file_size = 0;
  std::unique_ptr<RandomAccessFile> file;
  std::unique_ptr<RandomAccessFileReader> file_reader;
  s = r->options.env->GetFileSize(file_path, &file_size);
  if (s.ok()) {
    s = r->options.env->NewRandomAccessFile(file_path, &file, r->soptions);
  }
  if (s.ok()) {
    file_reader.reset(new RandomAccessFileReader(std::move(file), file_path));
  }
  if (s.ok()) {
    s = r->options.table_factory->NewTableReader(
        TableReaderOptions(r->ioptions, r->moptions.prefix_extractor.get(),
                           r->soptions, r->ioptions.internal_comparator),
        std::move(file_reader), file_size, &r->table_reader);
  }
  return s;
}

Status SstFileFilterReader::BulkReturn(const std::string& file_path, char * scratch) {
	/* NOT YET IMPLEMENTED */
  auto r = rep_.get();
  Status s;
  uint64_t file_size = 0;
  std::unique_ptr<RandomAccessFile> file;
  std::unique_ptr<RandomAccessFileReader> file_reader;
  s = r->options.env->GetFileSize(file_path, &file_size);
  if (s.ok()) {
    s = r->options.env->NewRandomAccessFile(file_path, &file, r->soptions);
  }
  if (s.ok()) {
    file_reader.reset(new RandomAccessFileReader(std::move(file), file_path));
  }

  if (s.ok()) {
    s = r->options.table_factory->NewTableReader(
        TableReaderOptions(r->ioptions, r->moptions.prefix_extractor.get(),
                           r->soptions, r->ioptions.internal_comparator),
        std::move(file_reader), file_size, &r->table_reader);
  }
  auto total_data_length = r->table_reader->GetTableProperties().get()->data_size;
  Slice result;
  file->Read(0, total_data_length, &result, scratch);

  return s;
}

Status SstFileFilterReader::GetDataBlocks(const ReadOptions& options,
                                          std::vector<Slice>& blocks) {
  auto r = rep_.get();
  return r->table_reader->GetDataBlocks(options, blocks);
}

Iterator* SstFileFilterReader::NewIterator(const ReadOptions& options) {
  auto r = rep_.get();
  auto sequence = options.snapshot != nullptr ?
                  options.snapshot->GetSequenceNumber() :
                  kMaxSequenceNumber;
  auto internal_iter = r->table_reader->NewIterator(
      options, r->moptions.prefix_extractor.get());
  return NewDBIterator(r->options.env, options, r->ioptions, r->moptions,
                       r->ioptions.user_comparator, internal_iter, sequence,
                       r->moptions.max_sequential_skip_in_iterations,
                       nullptr /* read_callback */);
}

Status SstFileFilterReader::filterWithCPU() {
	Status s;

	return s;
}

Status SstFileFilterReader::filterWithGPU() {
	Status s;

	return s;
}

std::shared_ptr<const TableProperties> SstFileFilterReader::GetTableProperties() const {
  return rep_->table_reader->GetTableProperties();
}

Status SstFileFilterReader::VerifyChecksum() {
  return rep_->table_reader->VerifyChecksum();
}

}  // namespace rocksdb
