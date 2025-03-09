import os
import re
import time

os.environ['MKL_THREADING_LAYER'] = 'GNU'

import argparse


def generate_eval_commands(data_root, model_name, model_type, task, max_v, min_v, trun_flag, trun_high, trun_low, quant_type, samples, bit_depth, quant_points_name, arch, lambda_value, epochs, epoch, save_period, learning_rate, batch_size, patch_size):
    if isinstance(trun_low, list):
        trun_low = '[' + ','.join(map(str, trun_low)) + ']'
        trun_high = '[' + ','.join(map(str, trun_high)) + ']'

    eval_command = (
        f"python -m compressai.utils.eval_model checkpoint "
        f"{data_root}/{model_type}/{task}/feature_test "
        f"-a {arch} --cuda -v "
        # lzj
        # modify the output path accordingly, use the epoch information
        f"-d {data_root}/{model_type}/{task}/{model_name}/decoded/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}/"
        f"lambda{lambda_value}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')}_epoch{epoch} "
        # load specified checkpoint accordingly, use the epoch information
        f"--per-image -p {data_root}/{model_type}/{task}/{model_name}/training_models/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}/"
        f"{arch}_lambda{lambda_value}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')}_checkpoint_epoch{epoch}.pth.tar "  
        f"--model_type={model_type} --task={task} --trun_flag={trun_flag} --trun_low={trun_low} --trun_high={trun_high} --quant_type={quant_type} --qsamples={samples} --bit_depth={bit_depth} --quant_points_name={quant_points_name} "
        # lzj
        f">>{data_root}/{model_type}/{task}/{model_name}/encoding_log/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}/"
        f"compresslzj_{model_name}_lambda{lambda_value}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')}.txt 2>&1"
    )

    return eval_command

def hyperprior_evaluate_pipeline(data_root, model_type, task, max_v, min_v, trun_flag, trun_low, trun_high, quant_type, samples, bit_depth, quant_points_name, lambda_value, epochs, save_period, learning_rate, batch_size, patch_size):
    arch = 'bmshj2018-hyperprior'
    model_name = arch.split('-')[-1]; print(model_name)
    # lzj
    for epoch in range(save_period-1, epochs, save_period):
        checkpoint_name = (f"{data_root}/{model_type}/{task}/{model_name}/training_models/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}/"
        f"{arch}_lambda{lambda_value}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')}_checkpoint_epoch{epoch}.pth.tar")
        # only perform evaluation when checkpoint exists
        if os.path.exists(checkpoint_name):
            eval_cmd = generate_eval_commands(data_root, model_name, model_type, task, max_v, min_v, trun_flag, trun_high, trun_low, quant_type, samples, bit_depth, quant_points_name, arch, lambda_value, epochs, epoch, save_period, learning_rate, batch_size, patch_size)

            time_start = time.time()
            print("\nEval Command:")
            print(eval_cmd)
            os.system(eval_cmd)
            print('encoding time: ', time.time() - time_start)

def argument_parsing():
    parser = argparse.ArgumentParser(description="Hyperprior Evaluation Pipeline")
    parser.add_argument('--lambda_value', type=float, default=0.01, help='lambda')
    parser.add_argument('--epochs', type=int, default=500, help='epochs')
    parser.add_argument('--save_period', type=int, default=20, help='save_period')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning_rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--patch_size', type=int, default=256, help='patch_size')
    parser.add_argument('--bit_depth', type=int, default=8, help='bit_depth')
    
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    # model_type = 'llama3'; task = 'csr'
    # max_v = 47.75; min_v = -78; trun_high = 5; trun_low = -5
    # lambda_value_all = [0.01405, 0.0142, 0.015, 0.07, 10]
    # epochs = 200; learning_rate = "1e-4"; batch_size = 40; patch_size = "64 4096" # height first, width later

    # model_type = 'dinov2'; task = 'cls'
    # max_v = 104.1752; min_v = -552.451; trun_high = 5; trun_low = -5
    # lambda_value_all = [0.001, 0.0017, 0.003, 0.0035, 0.01]
    # epochs = 800; learning_rate = "1e-4"; batch_size = 128; patch_size = "256 256"   # height first, width later

    model_type = 'dinov2'; task = 'seg'
    max_v = 103.2168; min_v = -530.9767; trun_high = 5; trun_low = -5
    # lambda_value_all = [0.0005, 0.001, 0.003, 0.007, 0.015]
    # epochs = 500; save_period = 20; learning_rate = "1e-4"; batch_size = 64; patch_size = "256 256"   # height first, width later

    # model_type = 'dinov2'; task = 'dpt'
    # max_v = [3.2777, 5.0291, 25.0456, 102.0307]; min_v = [-2.4246, -26.8908, -323.2952, -504.4310]; trun_high = [1,2,10,10]; trun_low = [-1,-2,-10,-10]
    # lambda_value_all = [0.001, 0.005, 0.02, 0.05, 0.12]
    # epochs = 200; learning_rate = "1e-4"; batch_size = 128; patch_size = "256 256"   # height first, width later
    
    # model_type = 'sd3'; task = 'tti'
    # max_v = 4.668; min_v = -6.176; trun_high = 4.668; trun_low = -6.176
    # lambda_value_all = [0.005, 0.01, 0.02, 0.05, 0.2]
    # epochs = 60; learning_rate = "1e-4"; batch_size = 32; patch_size = "512 512"   # height first, width later

    trun_flag = False

    if trun_flag == False: trun_high = max_v; trun_low = min_v
    quant_type = 'kmeans'; samples = 10 #; bit_depth = 8

    args = argument_parsing()
    lambda_value = args.lambda_value
    epochs = args.epochs
    save_period = args.save_period
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    patch_size = f"{args.patch_size} {args.patch_size}"
    bit_depth = args.bit_depth

    quant_points_name = f"/gdata1/gaocs/Data_FCM_NQ/{model_type}/{task}/quantization_mapping/quantization_mapping_{task}_{trun_flag}_trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}.json"
    data_root = "/gdata1/gaocs/Data_FCM_NQ"

    hyperprior_evaluate_pipeline(data_root, model_type, task, max_v, min_v, trun_flag, trun_low, trun_high, quant_type, samples, bit_depth, quant_points_name, lambda_value, epochs, save_period, learning_rate, batch_size, patch_size)