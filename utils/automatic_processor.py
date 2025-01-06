import os
import subprocess
import time
import uuid

import argparse
import pandas as pd
from tqdm import tqdm

STATUS_NO_RUN = 0
STATUS_RUNNING = 1
STATUS_FAIL = 2
STATUS_SUCCESS = 3


def view_gpu_usage():
    """ View the gpu usage.

    :return: A list of gpu ids.
    """
    try:
        # query memory usage for all GPUs
        memory_free_info = subprocess.check_output(
            'nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader',
            shell=True, encoding='utf-8')
        memory_free_values = [(gpu_index, int(free_mem)) for gpu_index, free_mem in
                              enumerate(memory_free_info.strip().split('\n'))]
        # sort from lowest to highest memory free
        return sorted(memory_free_values, key=lambda x: -x[1])
    except Exception as e:
        print('Failed to find free GPUs:\r\n' + str(e))
        exit(0)


class GPU_bank:
    """ A class to manage the gpu usage. """
    def __init__(self, maximum_num, minimum_free_memory=2000):
        self.gpu_usage = {}
        self.maximum_num = maximum_num
        self.minimum_free_memory = minimum_free_memory

    def get_freest_gpu(self):
        sorted_gpu_usage = view_gpu_usage()
        freest_gpu_num, freest_gpu_free_memory = None, 0
        if len(self.gpu_usage) < self.maximum_num:
            self.gpu_usage[sorted_gpu_usage[0][0]] = (
                    self.gpu_usage[sorted_gpu_usage[0][0]] + 1) if sorted_gpu_usage[0][0] in self.gpu_usage else 1
            freest_gpu_num, freest_gpu_free_memory = sorted_gpu_usage[0]
        elif len(self.gpu_usage) == self.maximum_num:
            for gpu_id, gpu_free_memory in sorted_gpu_usage:
                if gpu_id in self.gpu_usage:
                    self.gpu_usage[gpu_id] += 1
                    freest_gpu_num, freest_gpu_free_memory = gpu_id, gpu_free_memory
                    break
        else:
            raise NotImplementedError('gpu num is larger than maximum num')

        if freest_gpu_free_memory < self.minimum_free_memory:
            return None
        return freest_gpu_num

    def free_gpu(self, gpu_id):
        self.gpu_usage[gpu_id] -= 1
        if self.gpu_usage[gpu_id] == 0:
            del self.gpu_usage[gpu_id]


class Automatic_processor:
    """ A class to process the commands automatically.

    """
    def __init__(self, commands_working_root='/data/lihaojin/code/mo_segmentation',
                 automatic_records_root='automatic_records',
                 commands_list_filename='commands_to_be_process',
                 status_list_filename='commands_status.csv',
                 maximum_gpu_num=3, minimum_free_memory_allowed=2000,
                 single_sleep=15, round_sleep=2400):
        self.commands_working_root = commands_working_root
        self.automatic_records_root = automatic_records_root
        with open(os.path.join(automatic_records_root, commands_list_filename), 'r') as file:
            lines = file.readlines()
        self.commands_list = []
        temp_command = ''
        for line in lines:
            line = line.strip()
            if line:
                if line.startswith('#'):
                    continue
                temp_command += (' ' if temp_command else '') + line
            else:
                self.commands_list.append(temp_command)
                temp_command = ''
        if temp_command:
            self.commands_list.append(temp_command)
        self.status_list_path = os.path.join(automatic_records_root, status_list_filename)
        self.gpu_bank = GPU_bank(maximum_gpu_num, minimum_free_memory_allowed)
        self.single_sleep_time = single_sleep
        self.round_sleep_time = round_sleep

    def run(self):
        dataframe = pd.DataFrame({'command': self.commands_list,
                                  'status': [STATUS_NO_RUN] * len(self.commands_list),
                                  'output': [''] * len(self.commands_list)})
        dataframe.to_csv(self.status_list_path, index=False, sep=',')
        processes_list = [None] * len(self.commands_list)
        cache_path_list = [None] * len(self.commands_list)
        gpu_usage_list = [None] * len(self.commands_list)
        success_num = 0
        tqdm_bar = tqdm(total=len(self.commands_list))
        while 1:
            for command_id in range(len(self.commands_list)):
                if processes_list[command_id] is not None and (processes_list[command_id].poll() is None or (
                        processes_list[command_id].poll() == 0 and dataframe['status'][command_id] == STATUS_SUCCESS)):
                    continue
                elif processes_list[command_id] is not None and processes_list[command_id].poll() == 0 and \
                        (dataframe['status'][command_id] == STATUS_RUNNING or dataframe['status'][
                            command_id] == STATUS_FAIL):
                    self.gpu_bank.free_gpu(gpu_usage_list[command_id])
                    dataframe.loc[command_id, 'status'] = STATUS_SUCCESS
                    with open(cache_path_list[command_id], 'r') as file:
                        dataframe.loc[command_id, 'output'] = file.read()
                    os.remove(cache_path_list[command_id])
                    dataframe.to_csv(self.status_list_path, index=False, sep=',')
                    success_num += 1
                    tqdm_bar.update(1)
                elif processes_list[command_id] is None or (
                        processes_list[command_id].poll() is not None and processes_list[command_id].poll() != 0):
                    if processes_list[command_id] is not None and (
                            processes_list[command_id].poll() is not None and processes_list[command_id].poll() != 0):
                        dataframe.loc[command_id, 'status'] = STATUS_FAIL
                    else:
                        dataframe.loc[command_id, 'status'] = STATUS_RUNNING
                    dataframe.to_csv(self.status_list_path, index=False, sep=',')
                    freest_gpu_id = self.gpu_bank.get_freest_gpu()
                    if freest_gpu_id is None:
                        continue
                    gpu_usage_list[command_id] = freest_gpu_id
                    cache_path_list[command_id] = os.path.join(self.automatic_records_root, 'output_cache',
                                                               str(uuid.uuid1()))
                    command_to_be_process = self.commands_list[command_id] + ' --gpu_ids ' + str(
                        freest_gpu_id) + ' --result_output_file_path ' + cache_path_list[command_id]
                    processes_list[command_id] = subprocess.Popen(command_to_be_process, shell=True, text=True,
                                                                  cwd=self.commands_working_root,
                                                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    time.sleep(self.single_sleep_time)
                else:
                    print('wrong status, break')
                    exit(-1)
            if success_num == len(self.commands_list):
                break
            time.sleep(self.round_sleep_time)


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--commands_working_root', type=str, default='/data/lihaojin/code/mo_segmentation')
    argument_parser.add_argument('--automatic_records_root', type=str, default='automatic_records')
    argument_parser.add_argument('--commands_list_filename', type=str, default='commands_to_be_process')
    argument_parser.add_argument('--status_list_filename', type=str, default='commands_status.csv')
    argument_parser.add_argument('--maximum_gpu_num', type=int, default=3)
    argument_parser.add_argument('--minimum_free_memory_allowed', type=int, default=2000)
    argument_parser.add_argument('--single_sleep', type=int, default=10)
    argument_parser.add_argument('--round_sleep', type=int, default=60)
    args = argument_parser.parse_args()

    processor = Automatic_processor(args.commands_working_root, args.automatic_records_root,
                                    args.commands_list_filename, args.status_list_filename, args.maximum_gpu_num,
                                    args.minimum_free_memory_allowed, args.single_sleep, args.round_sleep)
    processor.run()

    print('Successfully finished')
