/*
 * gpu_manager.h
 *
 *  Created on: Mar 22, 2019
 *      Author: wonki
 */
#pragma once

#include <vector>
#include <queue>
#include <condition_variable>
#include <thread>
#include <mutex>
#include <cuda_runtime.h>
#include "db/version_set.h"
#include "table/get_context.h"

namespace rocksdb {
     
     class GPUManager {
       private:
             std::thread q_thread;
             std::thread provider;
 
        public:
            GPUManager(uint table_num_, bool _start, std::vector<rocksdb::SlicewithSchema> *schemakey_, uint stream_num) :
                q_thread(), provider() {
                table_num = table_num_;
                start = _start;
                read_options = nullptr;
                schemakey = schemakey_;
                current = nullptr;
                fp_list = new FilePicker*[table_num];
                context_list = new GetContext*[table_num];
                asyncValues = new std::vector<PinnableSlice>[table_num];
                d_schema = nullptr;
                stream_num_ = stream_num;
                cuda_stream = new cudaStream_t[stream_num];
             }
            ~GPUManager() {
                q_stop(); p_stop();
                for(uint i = 0 ; i < table_num; i++) {
                    delete fp_list[i];
                    delete context_list[i];
                }
                delete fp_list;
                delete context_list;
                delete asyncValues;
            }

            GPUManager(GPUManager const &) = delete;
            GPUManager& operator=(GPUManager const &) = delete;

            void q_start() { q_thread = std::thread(&GPUManager::queue_job, this); }
            void q_stop() { q_thread.join(); }
            
            void p_start() { provider = std::thread(&GPUManager::provide, this); }
            void p_stop() { provider.join(); }
            
            void queue_job();
            void provide();
            void submit(uint i);
            bool IsFilterSkipped(int level, bool is_file_last_in_level);
            bool isLastTable(uint i);
            void releaseManager(ruda::RudaAsyncManager * async_manager);
            void initialize();

            
            ReadOptions * read_options;
            Version *current;

            std::vector<SlicewithSchema> * schemakey;
            FilePicker ** fp_list;
            GetContext ** context_list;
            std::vector<PinnableSlice> * asyncValues;
            bool start;
            uint table_num;

            std::mutex m_mutex;
            std::mutex pro_con;
            std::condition_variable put_cond;
            std::condition_variable get_cond;
            
            std::mutex ruda_mutex;            
            uint64_t stream_num_;            
            ruda::RudaSchema * d_schema;
            cudaStream_t * cuda_stream;
            std::queue<ruda::RudaAsyncManager *> ruda_ptrs;
            
    };
}
