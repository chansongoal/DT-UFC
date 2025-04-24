import os
import re
import matplotlib.pyplot as plt
import time
import argparse

os.environ['MKL_THREADING_LAYER'] = 'GNU'

def generate_commands(dataset_root, data_root, source_file, model_name, model_type, task, trun_flag, trun_high, trun_low, quant_type, samples, bit_depth, quant_points_name, arch, lambda_value, epochs, save_period, learning_rate, batch_size, patch_size, pretrained_model):
    if isinstance(trun_low, list):
        trun_low = '[' + ','.join(map(str, trun_low)) + ']'
        trun_high = '[' + ','.join(map(str, trun_high)) + ']'

    training_models_path = f"{data_root}/{model_type}/{task}/{model_name}/training_models/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}"
    os.makedirs(training_models_path, exist_ok=True)
    training_log_path = f"{data_root}/{model_type}/{task}/{model_name}/training_log/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}"
    os.makedirs(training_log_path, exist_ok=True)
    eval_log_path = f"{data_root}/{model_type}/{task}/{model_name}/encoding_log/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}"
    os.makedirs(eval_log_path, exist_ok=True)

    # generate train command
    train_command = (
        f"python examples/train.py --model {arch} "
        f"-d /gdata1/gaocs/FCM_LM_Train_Data/{model_type}/{task}/{quant_type}{samples}_bitdepth{bit_depth}/crop_hgt{patch_size.split('-')[0]}_wdt{patch_size.split('-')[1]} "
        f"--checkpoint {pretrained_model} "
        f"--lambda {lambda_value} --epochs {epochs} --save_period={save_period} -lr {learning_rate} "
        f"--batch-size {batch_size} --patch-size {patch_size.split('-')[0]} {patch_size.split('-')[1]} --cuda --save "
        f"--model_type={model_type} --task={task} --trun_flag={trun_flag} --trun_low={trun_low} --trun_high={trun_high} --quant_type={quant_type} --qsamples={samples} --bit_depth={bit_depth} --quant_points_name={quant_points_name} "
        f"-mp {training_models_path}/{arch}_lambda{lambda_value}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size}_checkpoint.pth.tar "
        f">{training_log_path}/train_{model_name}_lambda{lambda_value}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size}.txt 2>&1"
    )

    eval_command = (
        f"python -m compressai.utils.eval_model checkpoint "
        f"{dataset_root}/{model_type}/{task}/feature "
        f"-a {arch} --cuda -v "
        f"--source_file {dataset_root}/{model_type}/{task}/source/{source_file} "
        f"-d {data_root}/{model_type}/{task}/{model_name}/decoded/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}/lambda{lambda_value}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')} "
        # f"--per-image -p {data_root}/{model_type}/{task}/{model_name}/training_models/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}/"
        # f"{arch}_lambda{lambda_value}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size}_checkpoint_best.pth.tar "
        f"--per-image -p /gdata/gaocs/pretrained_models/FCM_LM/{task}/trunl-6.176_trunh4.668_uniform0_bitdepth1/"
        f"{model_name}_lambda{lambda_value}_epochs{epochs}_lr1e-4_bs{batch_size}_patch{patch_size}_checkpoint.pth.tar "
        f"--model_type={model_type} --task={task} --trun_flag={trun_flag} --trun_low={trun_low} --trun_high={trun_high} --quant_type={quant_type} --qsamples={samples} --bit_depth={bit_depth} --quant_points_name={quant_points_name} "
        f">{eval_log_path}/compress_{model_name}_lambda{lambda_value}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size}.txt 2>&1"
    )

    return train_command, eval_command


def plot_train_loss(train_log_path, train_config):
    # Read all content
    log_name = os.path.join(train_log_path, train_config+'.txt')
    pdf_name = os.path.join(train_log_path, train_config+'.pdf')
    with open(log_name, 'r') as file:
        file_content = file.read()

    # Extract content that starts with "Test epoch"
    test_epoch_lines = re.findall(r"Test epoch.*", file_content)

    # Init
    losses = []
    mse_losses = []
    bpp_losses = []
    aux_losses = []

    # Extract losses
    for line in test_epoch_lines:
        match = re.search(r"Loss:\s*([\d\.]+)\s*\|\s*MSE\s*loss:\s*([\d\.]+)\s*\|\s*Bpp\s*loss:\s*([\d\.]+)\s*\|\s*Aux\s*loss:\s*([\d\.]+)", line)
        if match:
            losses.append(float(match.group(1)))
            mse_losses.append(float(match.group(2)))
            bpp_losses.append(float(match.group(3)))
            aux_losses.append(float(match.group(4)))

    start = 0
    losses = losses[start:]; mse_losses = mse_losses[start:]; bpp_losses = bpp_losses[start:]; aux_losses = aux_losses[start:]
    epochs = range(len(losses))

    plt.figure(figsize=(12,8))

    plt.subplot(221)
    plt.plot(epochs, losses, marker='o')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(222)
    plt.plot(epochs, mse_losses, marker='o')
    plt.title('MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')

    plt.subplot(223)
    plt.plot(epochs, bpp_losses, marker='o')
    plt.title('Bpp Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Bpp Loss')

    plt.subplot(224)
    plt.plot(epochs, aux_losses, marker='o')
    plt.title('Aux Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Aux Loss')

    plt.tight_layout()
    plt.savefig(pdf_name, dpi=600, format='pdf', bbox_inches='tight')

def compressai_train_evaluate_pipeline(arch, dataset_root, data_root, source_file, model_type, task, trun_flag, trun_low, trun_high, quant_type, samples, bit_depth, quant_points_name, lambda_value, epochs, save_period, learning_rate, batch_size, patch_size, pretrained_model):
    if arch == 'bmshj2018-hyperprior':
        model_name = 'hyperprior'; print(model_name)
    elif arch == 'elic2022-official':
        model_name = 'elic'; print(model_name)

    train_cmd, eval_cmd = generate_commands(dataset_root, data_root, source_file, model_name, model_type, task, trun_flag, trun_high, trun_low, quant_type, samples, bit_depth, quant_points_name, arch, lambda_value, epochs, save_period, learning_rate, batch_size, patch_size, pretrained_model)

    time_start = time.time()
    print("Train Command:")
    print(train_cmd)
    os.system(train_cmd)
    print('training time: ', time.time() - time_start)

    train_log_path = f"{data_root}/{model_type}/{task}/{model_name}/training_log/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}/"
    train_config = f"train_{model_name}_lambda{lambda_value}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')}"
    plot_train_loss(train_log_path, train_config)

    time_start = time.time()
    print("\nEval Command:")
    print(eval_cmd)
    os.system(eval_cmd)
    print('encoding time: ', time.time() - time_start)

def compressai_evaluate_pipeline(arch, dataset_root, data_root, source_file, model_type, task, trun_flag, trun_low, trun_high, quant_type, samples, bit_depth, quant_points_name, lambda_value, epochs, save_period, learning_rate, batch_size, patch_size, pretrained_model):
    if arch == 'bmshj2018-hyperprior':
        model_name = 'hyperprior'; print(model_name)
    elif arch == 'elic2022-official':
        model_name = 'elic'; print(model_name)

    if model_name == 'elic':
        if task == 'seg':
            # lambda_value_all = [0.0001, 0.00015, 0.0002, 0.0003, 0.0004, 0.00058, 0.0006, 0.0007, 0.0008, 0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.01]
            # epochs_all = [550, 550, 600, 600, 400, 600, 200, 200, 600, 600, 600, 600, 600, 600, 600, 600]
            # batch_size_all = [64, 64, 64, 64, 64, 100, 64, 64, 100, 100, 100, 100, 100, 100, 100, 100]
            lambda_value_all = [0.0001, 0.00015, 0.0002, 0.0003, 0.0004, 0.0006, 0.0007]
            epochs_all = [550, 550, 600, 600, 400, 200, 200]
            batch_size_all = [64, 64, 64, 64, 64, 64, 64]
    elif model_name == 'hyperprior':
        if task == 'seg':
            # lambda_value_all = [0.0007,	0.0008,	0.001,	0.0015,	0.002,	0.0025,	0.003,	0.004,	0.005,	0.006,	0.007,	0.01,	0.015]
            # epochs_all = [200,	200,	200,	200,	600,	600,	900,	600,	600,	1000,	1000,	1200,	1000]
            # batch_size_all = [128,	128,	128,	128,	128,	128,	128,	128,	128,	128,	128,	128,	128]
            # lambda_value_all = lambda_value_all[:11]; epochs_all = epochs_all[:11]; batch_size_all = batch_size_all[:11]
            lambda_value_all = [0.0005, 0.001, 0.003, 0.007, 0.015]
            epochs_all = [800, 800, 800, 800, 800]
            batch_size_all = [128,	128,	128,	128,	128]
        elif task == 'csr':
            lambda_value_all = [0.01405, 0.0142, 0.015, 0.07, 10]
            epochs_all = [200, 200, 200, 200, 200]
            batch_size_all = [40,	40,	40,	40,	40]
            patch_size = "64-4096"
        elif task == 'tti':
            lambda_value_all = [0.005, 0.01, 0.02, 0.05, 0.2]
            epochs_all = [60, 60, 60, 60, 60]
            batch_size_all = [32,	32,	32,	32,	32]
            patch_size = "512-512"

    for idx, lambda_value in enumerate(lambda_value_all):
        epochs = epochs_all[idx]
        batch_size = batch_size_all[idx]
        train_cmd, eval_cmd = generate_commands(dataset_root, data_root, source_file, model_name, model_type, task, trun_flag, trun_high, trun_low, quant_type, samples, bit_depth, quant_points_name, arch, lambda_value, epochs, save_period, learning_rate, batch_size, patch_size, pretrained_model)

        time_start = time.time()
        print("\nEval Command:")
        print(eval_cmd)
        os.system(eval_cmd)
        print('encoding time: ', time.time() - time_start)

def argument_parsing():
    parser = argparse.ArgumentParser(description="Train Evaluation Pipeline")
    parser.add_argument('--task', type=str, default='seg', help='task')
    parser.add_argument('--arch', type=str, default='bmshj2018-hyperprior', help='arch')
    parser.add_argument('--lambda_value', type=float, default=0.01, help='lambda')
    parser.add_argument('--epochs', type=int, default=500, help='epochs')
    parser.add_argument('--save_period', type=int, default=20, help='save_period')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning_rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--patch_size', type=str, default='256-256', help='patch_size')
    parser.add_argument('--bit_depth', type=int, default=8, help='bit_depth')
    parser.add_argument('--quant_type', type=str, default='kmeans', help='quant_type')
    parser.add_argument('--samples', type=int, default=10, help='samples')
    parser.add_argument('--pretrained_model', type=str, default='None', help='pretrained_model')
    
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = argument_parsing()
    task = args.task
    arch = args.arch
    lambda_value = args.lambda_value
    epochs = args.epochs
    save_period = args.save_period
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    patch_size = args.patch_size
    bit_depth = args.bit_depth
    quant_type = args.quant_type
    samples = args.samples
    pretrained_model = args.pretrained_model
    if pretrained_model: print(f'Pretrained model: {pretrained_model}')

    if task == 'seg': 
        model_type = 'dinov2'; source_file = 'seg_val_100.txt'; max_v = 105.95; min_v = -506.97
    elif task == 'csr': 
        model_type = 'llama3'; source_file = 'arc_challenge_test_longest500_shape.txt'; max_v = 47.75; min_v = -71.50
    elif task == 'tti': 
        model_type = 'sd3'; source_file = 'captions_val2017_select500.txt'; max_v = 4.46; min_v = -5.79
    
    trun_flag = 'False'; 
    if trun_flag == 'False': 
        trun_high = max_v; trun_low = min_v

    quant_points_name = f"/gdata1/gaocs/Data_FCM_NQ/{model_type}/{task}/quantization_mapping/quantization_mapping_{task}_trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}.json"
    data_root = "/gdata1/gaocs/Data_FCM_NQ"
    dataset_root = '/gdata1/gaocs/FCM_LM_Test_Dataset'

    # compressai_train_evaluate_pipeline(arch, dataset_root, data_root, source_file, model_type, task, trun_flag, trun_low, trun_high, quant_type, samples, bit_depth, quant_points_name, lambda_value, epochs, save_period, learning_rate, batch_size, patch_size, pretrained_model)

    compressai_evaluate_pipeline(arch, dataset_root, data_root, source_file, model_type, task, trun_flag, trun_low, trun_high, quant_type, samples, bit_depth, quant_points_name, lambda_value, epochs, save_period, learning_rate, batch_size, patch_size, pretrained_model)