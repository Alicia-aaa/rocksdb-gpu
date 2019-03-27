/*
 * gpu_manager.c
 *
 *  Created on: Mar 22, 2019
 *      Author: wonki
 */

#include <rocksdb/accelerator/gpu_manager.h>
#include <vector>

namespace rocksdb {

bool GPUManager::IsFilterSkipped(int level, bool is_file_last_in_level) {
    return current->cfd_->ioptions()->optimize_filters_for_hits &&
           (level > 0 || is_file_last_in_level) &&
           level == current->storage_info_.num_non_empty_levels() - 1;
}

bool isLastTable(uint i) {
    return i == table_num;
}

void GPUManager::submit(uint i) {
    if (isLastTable(i))
        return ;

    FilePicker fp = fp_list[i];
    FdWithKeyRange* f = fp.GetNextFileWithTable();

    while (f != nullptr) {
       current->GetTableCache()->AsyncFilter(*read_options, fp.GetInternalComparator(),
               fp.GetInternalKey(), (*schemakey)[i] ,&context_list[i],
               current->GetMutableCFOptions().prefix_extractor.get(), f,
               current->cfd_->internal_stats()->GetFileReadHist(fp.GetHitFileLevel()),
               IsFilterSkipped(static_cast<int>(fp.GetHitFileLevel()), fp.IsHitFileLastInLevel()),
               fp.GetCurrentLevel());

       submit(i+1);
       fp.GetNextFileWithTable();
    }
}

void GPUManager::queue_job() {
    submit(0);
    q_stop();
}

void GPUManager::provider() {

}

void GPUManager::consumer() {

}

}
