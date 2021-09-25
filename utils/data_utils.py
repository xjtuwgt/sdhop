from utils.ioutils import load_gz_file


def length_analysis(data_file_name: str):
    examples = load_gz_file(file_name=data_file_name)
    num_sent_list = []
    for example in examples:
        num_sent_list.append(example.sent_names)
        print(example.sent_names)