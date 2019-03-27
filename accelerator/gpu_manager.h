/*
 * gpu_manager.h
 *
 *  Created on: Mar 22, 2019
 *      Author: wonki
 */
#include <vector>
#include <thread>
#include <mutex>
#include "db/version_set.h"

namespace rocksdb {
    class GPUManager {
        public:
            GPUManager(uint table_num_, bool _start, std::vector<SlicewithSchema> *schemakey_) :
                q_thread(), value_provider(), value_consumer() {
                table_num = table_num_;
                start = _start;
                read_options = nullptr;
                schemakey = schemakey_;
                current = nullptr;
                asyncValues = new std::vector<rocksdb::PinnableSlice>[table_num];
            }
            ~GPUManager() { q_stop(); p_stop(); c_stop(); }

            GPUManager(GPUManager const &) = delete;
            GPUManager& operator=(GPUManager const &) = delete;

            void q_start() { q_thread = std::thread(&GPUManager::queue_job, this); }
            void q_stop() { q_thread.join(); }
            void p_start() { value_provider = std::thread(&GPUManager::provider, this); }
            void p_stop() { value_provider.join(); }
            void c_start() { value_consumer = std::thread(&GPUManager::consumer, this); }
            void c_stop() { value_consumer.join(); }

            rocksdb::ReadOptions * read_options;
            rocksdb::Version *current;
            std::vector<rocksdb::SlicewithSchema> * schemakey;
            std::vector<rocksdb::FilePicker> fp_list;
            std::vector<rocksdb::GetContext> context_list;
            std::vector<rocksdb::PinnableSlice> * asyncValues;
            bool start;
            uint table_num;

        private:
            std::mutex mtx_lock;
            std::thread q_thread;
            std::thread value_provider;
            std::thread value_consumer;
            void queue_job();
            int submit(uint i);
            bool IsFilterSkipped(int level, bool is_file_last_in_level);
            void provider();
            void consumer();
    };
}
