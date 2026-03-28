import gc
import os
from .nisqa_lib.NISQA_model import NisqaModel
import argparse


def nisqa_predict(mode, deg=None, deg_list=None, data_dir=None, output_dir='', csv_file=None, model: str = 'nisqa.tar',
                  csv_deg=None,
                  num_workers=0, bs=10):
    parser = argparse.ArgumentParser('')
    parser.add_argument('--mode', type=str, default='predict_file',
                        help='either predict_file, predict_dir, predict_list, or predict_csv', )
    # parser.add_argument('--pretrained_model', type=str,
    #                     help='file name of pretrained model (must be in current working folder)',
    #                     default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', 'nisqa.tar'))
    parser.add_argument('--deg', type=str, help='path to speech file')
    parser.add_argument('--deg_list', type=str, help='list of speech files')
    parser.add_argument('--data_dir', type=str, help='folder with speech files')
    parser.add_argument('--output_dir', type=str, help='folder to ouput results.csv')
    parser.add_argument('--csv_file', type=str, help='file name of csv (must be in current working folder)')
    parser.add_argument('--csv_deg', type=str, help='column in csv with files name/path')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for pytorchs dataloader')
    parser.add_argument('--bs', type=int, default=1, help='batch size for predicting')
    parser.add_argument('--ms_channel', type=int, help='audio channel in case of stereo file')

    args = parser.parse_args()
    args = vars(args)

    args['mode'] = mode
    args['pretrained_model'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', model)
    args['deg'] = deg
    args['deg_list'] = deg_list
    args['data_dir'] = data_dir
    args['output_dir'] = output_dir
    args['csv_file'] = csv_file
    args['csv_deg'] = csv_deg
    args['bs'] = bs
    args['num_workers'] = num_workers

    if args['mode'] == 'predict_file':
        if args['deg'] is None:
            raise ValueError('--deg argument with path to input file needed')
    elif args['mode'] == 'predict_dir':
        if args['data_dir'] is None:
            raise ValueError('--data_dir argument with folder with input files needed')
    elif args['mode'] == 'predict_csv':
        if args['csv_file'] is None:
            raise ValueError('--csv_file argument with csv file name needed')
        if args['csv_deg'] is None:
            raise ValueError('--csv_deg argument with csv column name of the filenames needed')
        if args['data_dir'] is None:
            args['data_dir'] = ''
    elif args['mode'] == 'predict_list':
        if args['deg_list'] is None:
            raise ValueError('--deg_list argument with list of input files needed')
        if not isinstance(args['deg_list'], list):
            raise ValueError('--deg_list argument must be a list')
        if len(args['deg_list']) == 0:
            raise ValueError('--deg_list argument must not be an empty list')
    else:
        raise NotImplementedError('--mode given not available')
    args['tr_bs_val'] = args['bs']
    args['tr_num_workers'] = args['num_workers']

    nisqa_model = NisqaModel(args)
    nisqa_ret = nisqa_model.predict()
    del nisqa_model
    gc.collect()
    return nisqa_ret


if __name__ == "__main__":
    nisqa_predict(mode='predict_file',
                  deg='./data/1dB_man_multi2_square_AUD.wav',
                  output_dir='./results')
    # nisqa_predict(mode='predict_dir',
    #               pretrained_model='./weights/chinese_nisqa_240126_193920044858/chinese_nisqa_240126_193920044858__ep_057.tar',
    #               data_dir='./data', output_dir='./results')

# 用法 python run_predict.py --mode predict_file --pretrained_model ./weights/baseline2.tar --deg  ./data/2.wav
# --output_dir ./results
# python run_predict.py --mode predict_dir --pretrained_model
# ./weights/chinese_nisqa_240126_193920044858/chinese_nisqa_240126_193920044858__ep_057.tar --data_dir ./data/
# --num_workers 0 --bs 10 --output_dir ./results
