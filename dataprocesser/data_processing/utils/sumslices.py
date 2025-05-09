import argparse
from data.load_data import sum_slices

# python sumslices.py --path 'D:\Projects\data\Task1\pelvis' --num 150
# python sumslices.py --path 'D:\Projects\data\Task1\brain' --num 150
# python sumslices.py --path 'D:\Projects\data\Task2\pelvis' --num 150
# python sumslices.py --path 'D:\Projects\data\Task2\brain' --num 150

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="sum slices.")
    #parser.add_argument('--config', default='./data/save_configs/sample.yaml')
    parser.add_argument('--path', default=r'D:\Projects\data\Task1\pelvis')
    parser.add_argument('--num',  default=1, help='number of patients to be summed')
    args = parser.parse_args()

    data_path=args.path
    num=int(args.num)
    sum_slices(data_path,num)