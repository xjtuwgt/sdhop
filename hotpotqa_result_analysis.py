from envs import OUTPUT_FOLDER
import os
import json

def list_all_folders(dir_name, model_type: str):
    folder_names = [os.path.join(dir_name, o) for o in os.listdir(dir_name)
     if os.path.isdir(os.path.join(dir_name, o))]
    folder_names = [i for i in folder_names if model_type in i]
    return folder_names

def list_all_txt_files(path):
    files = os.listdir(path)
    files_txt = [i for i in files if i.endswith('.txt')]
    eval_file_names = [i for i in files_txt if i.startswith('eval.epoch')]
    eval_file_names = [i for i in eval_file_names if 'gpu' not in i]
    return eval_file_names

def best_metric_collection(key_word=None, model_type='electra'):
    best_metric_dict = None
    best_joint_f1 = -1
    best_setting = None
    folder_names = list_all_folders(dir_name=OUTPUT_FOLDER, model_type=model_type)
    metric_list = []
    for folder_idx, folder_name in enumerate(folder_names):
        eval_file_names = list_all_txt_files(path=folder_name)
        trim_folder_name = folder_name[(len(OUTPUT_FOLDER)+1):]
        if key_word is not None:
            if key_word not in trim_folder_name:
                continue
        for file_idx, file_name in enumerate(eval_file_names):
            print('{} | {} | {} | {}'.format(folder_idx, file_idx, trim_folder_name, file_name))
            with open(os.path.join(folder_name, file_name)) as fp:
                lines = fp.readlines()
                for line in lines:
                    metric_dict = json.loads(line)
                    metric_list.append((os.path.join(folder_name, file_name), metric_dict['joint_f1'], metric_dict))
                    if metric_dict['joint_f1'] > best_joint_f1:
                        best_joint_f1 = metric_dict['joint_f1']
                        best_setting = os.path.join(folder_name, file_name)
                        best_metric_dict = metric_dict
    print('*' * 75)
    print('Best joint F1 = {}\nSetting = {}'.format(best_joint_f1, best_setting))
    for key, value in best_metric_dict.items():
        print('{}: {}'.format(key, value))
    print('*' * 75)

    sorted_metrics = sorted(metric_list, key=lambda x: x[1])
    for idx, metric in enumerate(sorted_metrics):
        print('{}: {}: {}'.format(idx, metric[0], metric[1]))
        for key, value in metric[2].items():
            print('{}: {}'.format(key, value))
        print('*' * 75)

if __name__ == '__main__':
    best_metric_collection()