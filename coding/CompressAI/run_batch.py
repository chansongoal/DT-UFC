import os
import re
import matplotlib.pyplot as plt
import time
import argparse

os.environ['MKL_THREADING_LAYER'] = 'GNU'

def generate_train_commands(train_data_root, data_root, \
                      trun_flag, trun_high, trun_low, transform_type, samples, bit_depth, \
                      train_model_type, train_task, \
                      arch, arch_name, lambda_value, epochs, save_period, learning_rate, batch_size, patch_size, pretrained_model):
    if isinstance(trun_low, list):
        trun_low = '[' + ','.join(map(str, trun_low)) + ']'
        trun_high = '[' + ','.join(map(str, trun_high)) + ']'

    training_models_path = f"{data_root}/training_models/{arch_name}/trained_{train_task}/{transform_type}{samples}_bitdepth{bit_depth}"; os.makedirs(training_models_path, exist_ok=True)
    training_log_path = f"{data_root}/training_log/{arch_name}/trained_{train_task}/{transform_type}{samples}_bitdepth{bit_depth}"; os.makedirs(training_log_path, exist_ok=True)

    if train_task == 'hybrid':
        train_data_paths = f"{train_data_root}/dinov2/seg/{transform_type}{samples}_bitdepth{bit_depth}/crop_hgt256_wdt256,{train_data_root}/llama3/csr/{transform_type}{samples}_bitdepth{bit_depth}/crop_hgt256_wdt256,{train_data_root}/sd3/tti/{transform_type}{samples}_bitdepth{bit_depth}/crop_hgt256_wdt256 "
    else:
        train_data_paths = f"{train_data_root}/{train_model_type}/{train_task}/{transform_type}{samples}_bitdepth{bit_depth}/crop_hgt{patch_size.split('-')[0]}_wdt{patch_size.split('-')[1]} "
    
    # generate train command
    train_command = (
        f"python examples/train.py --model {arch} "
        f"-d {train_data_paths} "
        f"--checkpoint {pretrained_model} "
        f"--lambda {lambda_value} --epochs {epochs} --save_period={save_period} -lr {learning_rate} "
        f"--batch-size {batch_size} --patch-size {patch_size.split('-')[0]} {patch_size.split('-')[1]} --cuda --save "
        f"--model_type={train_model_type} --train_task={train_task} --trun_flag={trun_flag} --trun_low={trun_low} --trun_high={trun_high} --transform_type={transform_type} --qsamples={samples} --bit_depth={bit_depth} "
        f"-mp {training_models_path}/{arch}_lambda{lambda_value}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size}_checkpoint.pth.tar "
        f">{training_log_path}/train_{arch_name}_lambda{lambda_value}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size}.txt 2>&1"
    )

    return train_command

def generate_eval_commands(test_data_root, data_root, \
                            trun_flag, trun_high, trun_low, transform_type, samples, bit_depth, transform_mapping_name, \
                            train_task, test_model_type, test_task, source_file, \
                            arch, arch_name, lambda_value, epochs, learning_rate, batch_size, patch_size):
    if isinstance(trun_low, list):
        trun_low = '[' + ','.join(map(str, trun_low)) + ']'
        trun_high = '[' + ','.join(map(str, trun_high)) + ']'

    training_models_path = f"{data_root}/training_models/{arch_name}/trained_{train_task}/{transform_type}{samples}_bitdepth{bit_depth}"; os.makedirs(training_models_path, exist_ok=True)
    encoding_log_path = f"{data_root}/encoding_log/{arch_name}/trained_{train_task}/{transform_type}{samples}_bitdepth{bit_depth}/{test_model_type}_{test_task}"; os.makedirs(encoding_log_path, exist_ok=True)
    decoded_path = f"{data_root}/decoded/{arch_name}/trained_{train_task}/{transform_type}{samples}_bitdepth{bit_depth}/{test_model_type}_{test_task}"

    # generate eval command
    eval_command = (
        f"python -m compressai.utils.eval_model checkpoint "
        f"{test_data_root}/{test_model_type}/{test_task}/feature "
        f"-a {arch} --cuda -v "
        f"--source_file {test_data_root}/{test_model_type}/{test_task}/source/{source_file} "
        f"-d {decoded_path}/lambda{lambda_value}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')} "
        f"--per-image -p {training_models_path}/{arch}_lambda{lambda_value}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size}_checkpoint_best.pth.tar "
        f"--model_type={test_model_type} --task={test_task} --trun_flag={trun_flag} --trun_low={trun_low} --trun_high={trun_high} --transform_type={transform_type} --qsamples={samples} --bit_depth={bit_depth} --transform_mapping_name={transform_mapping_name} "
        f">{encoding_log_path}/{arch_name}_trained_{train_task}_compress_{test_task}_lambda{lambda_value}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size}.txt 2>&1"
    )

    return eval_command

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

def compressai_train(train_data_root, data_root, \
                     trun_flag, transform_type, samples, bit_depth, \
                     train_model_type, train_task, \
                     arch, arch_name, lambda_value, epochs, save_period, learning_rate, batch_size, patch_size, pretrained_model):

    if train_task == 'seg': trun_high = 105.95; trun_low = -506.97
    elif train_task == 'csr': trun_high = 47.75; trun_low = -71.50
    elif train_task == 'tti': trun_high = 4.46; trun_low = -5.79
    elif train_task == 'hybrid': trun_high = 105.95; trun_low = -506.97

    train_cmd = generate_train_commands(train_data_root, data_root, \
                                        trun_flag, trun_high, trun_low, transform_type, samples, bit_depth, \
                                        train_model_type, train_task, \
                                        arch, arch_name, lambda_value, epochs, save_period, learning_rate, batch_size, patch_size, pretrained_model)

    time_start = time.time()
    print("Train Command:")
    print(train_cmd)
    os.system(train_cmd)
    print('training time: ', time.time() - time_start)

    training_log_path = f"{data_root}/training_log/{arch_name}/trained_{train_task}/{transform_type}{samples}_bitdepth{bit_depth}"
    train_config = f"train_{arch_name}_lambda{lambda_value}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')}"
    plot_train_loss(training_log_path, train_config)

def compressai_test(test_data_root, data_root, \
                    trun_flag, transform_type, samples, bit_depth, \
                    train_task, \
                    arch, arch_name, lambda_value, epochs, learning_rate, batch_size, patch_size):

    test_task_all = ['csr', 'seg', 'tti']
    # test_task_all = ['tti']

    for idx, test_task in enumerate(test_task_all):
        if test_task == 'seg': test_model_type = 'dinov2'; source_file = 'seg_val_100.txt'; trun_high = 105.95; trun_low = -506.97
        elif test_task == 'csr': test_model_type = 'llama3'; source_file = 'arc_challenge_test_longest500_shape.txt'; trun_high = 47.75; trun_low = -71.50
        elif test_task == 'tti': test_model_type = 'sd3'; source_file = 'captions_val2017_select500.txt'; trun_high = 4.46; trun_low = -5.79

        transform_mapping_name = f'{data_root}/transform_mapping/{test_model_type}_{test_task}/transform_mapping_{test_task}_{transform_type}{samples}_bitdepth{bit_depth}.json'

        eval_cmd = generate_eval_commands(test_data_root, data_root, \
                                          trun_flag, trun_high, trun_low, transform_type, samples, bit_depth, transform_mapping_name, \
                                          train_task, test_model_type, test_task, source_file, \
                                          arch, arch_name, lambda_value, epochs, learning_rate, batch_size, patch_size)
        time_start = time.time()
        print("\nEval Command:")
        print(eval_cmd)
        os.system(eval_cmd)
        print('encoding time: ', time.time() - time_start)
    

def compressai_pipeline(pipeline_config, train_data_root, test_data_root, data_root, \
                        transform_type, samples, bit_depth, \
                        train_model_type, train_task, \
                        arch, lambda_value, epochs, save_period, learning_rate, batch_size, patch_size, pretrained_model):
    if arch == 'bmshj2018-hyperprior':
        arch_name = 'hyperprior'; print(arch_name)
    elif arch == 'elic2022-official':
        arch_name = 'elic'; print(arch_name)
    trun_flag = 'False'

    if 'train' in pipeline_config:
        compressai_train(train_data_root, data_root, \
                        trun_flag, transform_type, samples, bit_depth, \
                        train_model_type, train_task, \
                        arch, arch_name, lambda_value, epochs, save_period, learning_rate, batch_size, patch_size, pretrained_model)
    
    if 'test' in pipeline_config:
        compressai_test(test_data_root, data_root, \
                        trun_flag, transform_type, samples, bit_depth, \
                        train_task, \
                        arch, arch_name, lambda_value, epochs, learning_rate, batch_size, patch_size)

def compressai_test_multiple(pipeline_config, train_data_root, test_data_root, data_root, \
                            transform_type, samples, bit_depth, \
                            train_model_type, train_task, \
                            arch, lambda_value, epochs, save_period, learning_rate, batch_size, patch_size, pretrained_model):
    if arch == 'bmshj2018-hyperprior':
        arch_name = 'hyperprior'; print(arch_name)
    elif arch == 'elic2022-official':
        arch_name = 'elic'; print(arch_name)
    trun_flag = 'False'

    if arch_name == 'hyperprior':
        if train_task == 'csr':
            lambda_value_all = [0.0005, 0.0008, 0.0017, 0.0019, 0.002, 0.0025, 0.003, 0.0035, 0.006]
            epochs_all = [400, 400, 400, 400, 400, 400, 400, 400, 400]; 
            batch_size_all = [128, 128, 128, 128, 128, 128, 128, 128, 128]
            learning_rate = "0.0001";  patch_size = "64-1024"   # height first, width later
        elif train_task == 'seg':
            lambda_value_all = [0.0007,	0.0008, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.006, 0.007, 0.01, 0.015]
            epochs_all = [200, 200, 200, 600, 600, 900, 600, 600, 1000, 1000, 1200, 1000]
            batch_size_all = [128, 128,	128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
            patch_size = "256-256"; learning_rate = "0.0001"
        elif train_task == 'tti':
            lambda_value_all = [0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.05]
            epochs_all = [600, 600, 600, 600, 600, 600, 600, 600, 600]
            batch_size_all = [32, 32, 32, 32, 32, 32, 32, 32, 32]
            patch_size = "512-512"; learning_rate = '0.0001'
        elif train_task == 'hybrid':
            lambda_value_all = [0.0005,	0.0008,	0.001, 0.0013, 0.0015, 0.0018, 0.0019, 0.002, 0.0021, 0.0023, 0.0025, 0.0028, 0.003, 0.004, 0.005, 0.006, 0.007, 0.01, 0.02]
            epochs_all = [600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600]
            batch_size_all = [360, 180, 360, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 360, 180, 180, 360, 180]
            patch_size = "256-256"; learning_rate = '0.0001'
    elif arch_name == 'elic':
        if train_task == 'hybrid':
            lambda_value_all = [0.0001,	0.0003,	0.0005,	0.0008,	0.001, 0.0015, 0.0019, 0.0021, 0.0025, 0.003, 0.004, 0.005, 0.007, 0.008, 0.01, 0.015, 0.02]
            epochs_all = [600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600]
            batch_size_all = [120, 60, 120, 60, 60, 60, 60, 60, 60, 60, 60, 120, 60, 60, 60, 60, 60]
            patch_size = "256-256"; learning_rate = '0.0001'
    
    for idx, lambda_value in enumerate(lambda_value_all):
        epochs = epochs_all[idx]
        batch_size = batch_size_all[idx]

        compressai_test(test_data_root, data_root, \
                            trun_flag, transform_type, samples, bit_depth, \
                            train_task, \
                            arch, arch_name, lambda_value, epochs, learning_rate, batch_size, patch_size)

def argument_parsing():
    parser = argparse.ArgumentParser(description="Train Evaluation Pipeline")
    parser.add_argument('--pipeline_config', type=str, default='train_test', help='pipeline_config')
    parser.add_argument('--arch', type=str, default='bmshj2018-hyperprior', help='arch')
    parser.add_argument('--train_model_type', type=str, default='hybrid', help='train_model_type')
    parser.add_argument('--train_task', type=str, default='hybrid', help='train_task')
    parser.add_argument('--transform_type', type=str, default='kmeans', help='transform_type')
    parser.add_argument('--samples', type=int, default=10, help='samples')
    parser.add_argument('--bit_depth', type=int, default=8, help='bit_depth')
    parser.add_argument('--lambda_value', type=float, default=0.01, help='lambda')
    parser.add_argument('--epochs', type=int, default=500, help='epochs')
    parser.add_argument('--save_period', type=int, default=20, help='save_period')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning_rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--patch_size', type=str, default='256-256', help='patch_size')
    parser.add_argument('--pretrained_model', type=str, default='None', help='pretrained_model')
    
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = argument_parsing()
    pipeline_config = args.pipeline_config
    arch = args.arch
    train_model_type = args.train_model_type
    train_task = args.train_task
    transform_type = args.transform_type
    samples = args.samples
    bit_depth = args.bit_depth
    lambda_value = args.lambda_value
    epochs = args.epochs
    save_period = args.save_period
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    patch_size = args.patch_size
    pretrained_model = args.pretrained_model
    if pretrained_model: print(f'Pretrained model: {pretrained_model}')

    data_root = "/gdata1/gaocs/Data_DTUFC"
    train_data_root = "/gdata1/gaocs/FCM_LM_Train_Data"
    test_data_root = "/gdata1/gaocs/FCM_LM_Test_Dataset"

    # compressai_pipeline(pipeline_config, train_data_root, test_data_root, data_root, \
    #                     transform_type, samples, bit_depth, \
    #                     train_model_type, train_task, \
    #                     arch, lambda_value, epochs, save_period, learning_rate, batch_size, patch_size, pretrained_model)

    compressai_test_multiple(pipeline_config, train_data_root, test_data_root, data_root, \
                            transform_type, samples, bit_depth, \
                            train_model_type, train_task, \
                            arch, lambda_value, epochs, save_period, learning_rate, batch_size, patch_size, pretrained_model)