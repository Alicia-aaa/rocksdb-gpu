/*
 * gpu_manager.c
 *
 *  Created on: Mar 22, 2019
 *      Author: wonki
 */

#include <vector>
#include "accelerator/gpu_manager.h"
#include "db/internal_stats.h"
#include "accelerator/cuda/async_manager.h"
#include "accelerator/cuda/filter.h"

namespace rocksdb {

bool GPUManager::IsFilterSkipped(int level, bool is_file_last_in_level) {
    return current->cfd()->ioptions()->optimize_filters_for_hits &&
           (level > 0 || is_file_last_in_level) &&
           level == current->storage_info()->num_non_empty_levels() - 1;
}

bool GPUManager::isLastTable(uint i) {
    return i == table_num;
}

void GPUManager:: submit(uint i) {
    if (isLastTable(i))
        return ;

    FilePicker * fp = fp_list[i];
    FdWithKeyRange* f = fp->GetNextFileWithTable();

    while (f != nullptr) {
       while(!ruda::capacityCheck());
       ruda::RudaAsyncManager * async_manager = new ruda::RudaAsyncManager(d_schema, &m_mutex, cuda_stream);
       ruda_mutex.lock();
       ruda_ptrs.push(async_manager);
       ruda_mutex.unlock();
       current->GetTableCache()->AsyncFilter(*read_options, *(fp->GetInternalComparator()),
               i, context_list[i],
               current->GetMutableCFOptions().prefix_extractor.get(), f,
               current->cfd()->internal_stats()->GetFileReadHist(fp->GetHitFileLevel()),
               IsFilterSkipped(static_cast<int>(fp->GetHitFileLevel()), fp->IsHitFileLastInLevel()),
               fp->GetCurrentLevel(), async_manager);

       submit(i+1);
       f = fp->GetNextFileWithTable();
    }
}

void GPUManager::queue_job() {
    initialize();
    submit(0);
//    q_stop();
    start = false;   
}

void GPUManager::provide() {
    while(1) {
        ruda_mutex.lock();
        if(ruda_ptrs.empty()) continue;
        ruda::RudaAsyncManager * async_mgr = ruda_ptrs.front();
        ruda_ptrs.pop();
        ruda_mutex.unlock();
        while(async_mgr->kComplete != stream_num_);
        
        std::unique_lock<std::mutex> lock(pro_con);
        if(asyncValues[async_mgr->join_idx_].empty()) {
           async_mgr->translatePairsToSlices(*(async_mgr->h_datablocks), asyncValues[async_mgr->join_idx_]);
           get_cond.notify_one();
        } else {
           put_cond.wait(lock);
           async_mgr->translatePairsToSlices(*(async_mgr->h_datablocks), asyncValues[async_mgr->join_idx_]);
        }
        lock.unlock();
        
        async_mgr->unregisterPinnedMemory(*(async_mgr->h_datablocks), *(async_mgr->h_seek_indices));
        async_mgr->clear();       
        releaseManager(async_mgr);
    }
}

void GPUManager::releaseManager(ruda::RudaAsyncManager * async_manager) {
   ruda::releaseAsyncManager(async_manager);
}

void GPUManager::initialize() {
   ruda::initializeGlobal(*schemakey, cuda_stream, stream_num_, d_schema);
}
}
