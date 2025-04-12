import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
from tokenizations.bpe_tokenizer import get_encoder

GPT2_BASE_DIR=str(os.getenv('GPT2_BASE_DIR'))

def build_files(data_path, tokenized_data_path, num_pieces, full_tokenizer, min_length):
    """
    用于将原始文本数据转换为适合GPT2模型训练的格式
    * data_path: 原始数据路径
    * tokenized_data_path: 分词后的数据存放路径
    * num_pieces: 将训练语料分成多少份, 如100份, 文章总数为1000, 那么每份就有10个文章
    * full_tokenizer: 分词器
    * min_length: 最短用于训练的文章的长度, 如128, 则长度小于128的文章不参与训练
    """
    with open(data_path, 'r', encoding='utf8') as f:
        print('reading lines')
        lines = json.load(f)  # 是一个列表
        lines = [line.replace('\n', ' [SEP] ') for line in lines]  # 用[SEP]表示换行, 段落之间使用SEP表示段落结束, line表示的是一篇文章
    all_len = len(lines)
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    for i in tqdm(range(num_pieces)):
        sublines = lines[all_len // num_pieces * i: all_len // num_pieces * (i + 1)]  # 分割成num_pieces份
        if i == num_pieces - 1:
            sublines.extend(lines[all_len // num_pieces * (i + 1):])  # 把尾部例子添加到最后一个piece
        sublines = [full_tokenizer.tokenize(line) for line in sublines if len(line) > min_length]  # 进行tokenization, 只考虑长度超过min_length的句子
        sublines = [full_tokenizer.convert_tokens_to_ids(line) for line in sublines]  #  将token转换为id
        full_line = []
        for subline in sublines:  # 每一个subline都是一篇文章
            full_line.append(full_tokenizer.convert_tokens_to_ids('[MASK]'))  # 文章开头添加MASK表示文章开始
            full_line.extend(subline)
            full_line.append(full_tokenizer.convert_tokens_to_ids('[CLS]'))  # 文章之间添加CLS表示文章结束
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'w') as f:  # 每组文章写入一个文件
            for id in full_line:
                f.write(str(id) + ' ')
    print('finish')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--raw_data_path', default='data/train.json', type=str, required=False, help='原始训练语料')
    parser.add_argument('--tokenized_data_path', default='data/tokenized/', type=str, required=False,
                        help='tokenized语料存放位置')
    parser.add_argument('--raw', action='store_true', help='是否先做tokenize')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练循环')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss, 设置为gradient accumulation的整数倍')
    parser.add_argument('--stride', default=768, type=int, required=False, help='训练时取训练数据的窗口步长')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--fp16', action='store_true', help='混合精度')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--num_pieces', default=100, type=int, required=False, help='将训练语料分成多少份')
    parser.add_argument('--min_length', default=128, type=int, required=False, help='最短收录文章长度')
    parser.add_argument('--output_dir', default='model/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型训练起点路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    parser.add_argument('--bpe_token', action='store_true', help='subword')
    parser.add_argument('--encoder_json', default="tokenizations/encoder.json", type=str, help="encoder.json")
    parser.add_argument('--vocab_bpe', default="tokenizations/vocab.bpe", type=str, help="vocab.bpe")

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    if args.segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert  # 以词为单位的分词方式, 如"我喜欢编程"会被分成"我", "喜欢", "编程"
    else:
        from tokenizations import tokenization_bert  # 以字符为代为的分词方式, 如"我喜欢编程"会被分成"我", "喜", "欢", "编", "程"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡

    model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)  # 加载模型配置
    print('config:\n' + model_config.to_json_string())

    n_ctx = model_config.n_ctx  # 训练时的上下文长度. 即模型一次可以处理的最大token数量
    if args.bpe_token:
        full_tokenizer = get_encoder(args.encoder_json, args.vocab_bpe)
    else:
        full_tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    full_tokenizer.max_len = 999999  # 表示将分词器的最大长度设置为999999, 确保分词器在处理长文本的时候不会因为长度限制而截断
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    raw_data_path = args.raw_data_path
    tokenized_data_path = args.tokenized_data_path
    raw = args.raw  # 选择是否从零开始构建数据集
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    warmup_steps = args.warmup_steps
    log_step = args.log_step
    stride = args.stride
    gradient_accumulation = args.gradient_accumulation
    fp16 = args.fp16  # 不支持半精度的显卡请勿打开
    fp16_opt_level = args.fp16_opt_level
    max_grad_norm = args.max_grad_norm
    num_pieces = args.num_pieces
    min_length = args.min_length
    output_dir = args.output_dir
    tb_writer = SummaryWriter(log_dir=args.writer_dir)
    assert log_step % gradient_accumulation == 0

    """
    * raw_data_path: 原始数据路径
    * tokenized_data_path: 分词后的数据存放路径
    * raw: 是否从零开始构建适配GPT2格式的数据集
    * epochs: 训练轮数
    * batch_size: 训练batch size
    * lr: 学习率
    * warmup_steps: warm up步数
    * log_step: 多少步汇报一次loss
    * n_ctx: 训练时的上下文长度(窗口长度)
    * stride: 窗口之间的间隔. 假设模型的输入窗口长度为 1024 tokens, stride 设置为 512, 则每次新窗口的起始位置相对于上一个窗口向前移动 512 tokens, 从而使得窗口之间存在 50% 的重叠
    * gradient_accumulation: 梯度积累, 将训练数据分成多个 mini-batch, 每个 mini-batch 分别进行前向计算和梯度反向传播, 但不立即更新模型参数, 将每个 mini-batch 计算出的梯度累加起来, 直到累计达到一定次数(由 gradient_accumulation 参数指定), 再一次性更新模型权重
    * fp16: 混合精度
    * fp16_opt_level: 混合精度优化级别
    * max_grad_norm: 梯度裁剪
    * num_pieces: 将训练语料分成多少份, 如100份, 文章总数为1000, 那么每份就有10个文章
    * min_length: 最短用于训练的文章的长度, 如128, 则长度小于128的文章不参与训练
    * output_dir: 模型输出路径
    * tb_writer: Tensorboard路径
    """

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if raw:
        print('building files')
        build_files(data_path=raw_data_path, tokenized_data_path=tokenized_data_path, num_pieces=num_pieces,
                    full_tokenizer=full_tokenizer, min_length=min_length)
        print('files built')

    if not args.pretrained_model:
        model = transformers.modeling_gpt2.GPT2LMHeadModel(config=model_config)
    else:
        model = transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    model.train()
    model.to(device)

    num_parameters = 0  # 模型参数数量
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('number of parameters: {}'.format(num_parameters))

    multi_gpu = False  # 是否使用多GPU训练
    full_len = 0  # 统计token id的总数量
    print('calculating total steps')
    for i in tqdm(range(num_pieces)):
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
            full_len += len([int(item) for item in f.read().strip().split()])
    total_steps = int(full_len / stride * epochs / batch_size / gradient_accumulation)
    print('total steps = {}'.format(total_steps))

    optimizer = transformers.AdamW(model.parameters(), lr=lr, correct_bias=True)
    scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps,
                                                          t_total=total_steps)
    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)  # 混合精度训练

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])  # 将模型wrap一下用于多GPU训练
        multi_gpu = True
    print('starting training')
    overall_step = 0
    running_loss = 0
    for epoch in range(epochs):
        print('epoch {}'.format(epoch + 1))
        now = datetime.now()
        print('time: {}'.format(now))
        x = np.linspace(0, num_pieces - 1, num_pieces, dtype=np.int32)  # x表示的是第x组文章
        random.shuffle(x)  # 随机打乱文章组的顺序
        piece_num = 0  # 表示当前训练到第几组文章
        for i in x:
            with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
                line = f.read().strip()
            tokens = line.split()
            tokens = [int(token) for token in tokens]  # 这组文章所有tokens
            start_point = 0  # 表示当前窗口的起始token
            samples = []  # 存储当前组文章的所有窗口
            while start_point < len(tokens) - n_ctx:  # 窗口的起始token小于这组文章所有tokens的长度减去窗口长度
                samples.append(tokens[start_point: start_point + n_ctx])  # 将当前窗口的tokens添加到samples中
                start_point += stride  # 移动窗口
            if start_point < len(tokens):  # 补足最后一个窗口, 大小不够n_ctx的窗口
                samples.append(tokens[len(tokens)-n_ctx:])
            random.shuffle(samples)  # 随机打乱窗口顺序
            for step in range(len(samples) // batch_size):  # 最后一个窗口丢弃

                #  prepare data
                batch = samples[step * batch_size: (step + 1) * batch_size]  # 取出当前batch
                batch_inputs = []  # 存储当前batch的tokens
                for ids in batch:
                    int_ids = [int(x) for x in ids]
                    batch_inputs.append(int_ids)  # 将当前batch的tokens添加到batch_inputs中
                batch_inputs = torch.tensor(batch_inputs).long().to(device)  # 将batch_inputs转换为tensor并转移到GPU上

                #  forward pass
                outputs = model.forward(input_ids=batch_inputs, labels=batch_inputs)  # 正向传播, GT是当前batch的tokens
                loss, logits = outputs[:2]  # logits是每个位置上所有词汇的预测分数(未经过softmax), loss是当前batch的损失

                #  get loss
                if multi_gpu:
                    loss = loss.mean()  # 多GPU训练时, loss需要取平均
                if gradient_accumulation > 1:
                    loss = loss / gradient_accumulation  # 如果不对损失值进行归一化, 那么每次更新模型参数时, 梯度会变得非常大, 导致模型不收敛

                #  loss backward
                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:  # 缩放的目的是为了避免 fp16 数值范围较小可能导致的下溢(underflow)问题. 简单来说它能表达的最小非零整数比FP32要大很多, 当计算得到的梯度非常小的时候, 它们可能会被四舍五入为0, 这就是所谓的下溢现象, 导致训练停滞或收敛速度变慢. Loss Scaling的核心思想是, 在反向传播之前将损失乘以一个较大的标量, 使得计算得到的梯度数值成比例地放大, 从而进入FP16的数值范围. 这一步骤在混合精度训练中非常重要.
                        scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)  # amp.master_params(optimizer) 获取的是以 32 位浮点形式存储的模型参数
                        """
                        混合精度训练的步骤如下(以apex为例):
                        1. 前向传播: FP32参数会被cast为FP16执行计算
                        2. 反向传播: 计算得到的梯度在FP16下可能很小, 会进行梯度缩放以避免下溢
                        3. 反向传播得到的梯度经过梯度反缩放, 更新模型FP32参数
                        4. 更新后的FP32模型参数会被cast为FP16, 以便于下一次前向传播
                        """
                else:
                    loss.backward()  # 反向传播
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # 进行梯度裁剪, 防止梯度爆炸

                #  optimizer step
                if (overall_step + 1) % gradient_accumulation == 0:  # 如果当前batch是gradient accumulation的整数倍, 则更新模型参数
                    running_loss += loss.item()  # 累加loss
                    optimizer.step()  # 更新模型参数
                    optimizer.zero_grad()  # 清空梯度
                    scheduler.step()  # 更新学习率
                if (overall_step + 1) % log_step == 0:
                    tb_writer.add_scalar('loss', loss.item() * gradient_accumulation, overall_step)
                    print('now time: {}:{}. Step {} of piece {} of epoch {}, loss {}'.format(
                        datetime.now().hour,
                        datetime.now().minute,
                        step + 1,
                        piece_num,
                        epoch + 1,
                        running_loss * gradient_accumulation / (log_step / gradient_accumulation)))
                    running_loss = 0
                overall_step += 1  # 已经训练的batch的个数
            piece_num += 1  # 训练到第几组文章

        print('saving model for epoch {}'.format(epoch + 1))
        if not os.path.exists(output_dir + 'model_epoch{}'.format(epoch + 1)):
            os.mkdir(output_dir + 'model_epoch{}'.format(epoch + 1))
        model_to_save = model.module if hasattr(model, 'module') else model  # 如果使用了多GPU训练, 需要去掉外层的包装
        model_to_save.save_pretrained(output_dir + 'model_epoch{}'.format(epoch + 1))
        # torch.save(scheduler.state_dict(), output_dir + 'model_epoch{}/scheduler.pt'.format(epoch + 1))
        # torch.save(optimizer.state_dict(), output_dir + 'model_epoch{}/optimizer.pt'.format(epoch + 1))
        print('epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('time: {}'.format(then))
        print('time for one epoch: {}'.format(then - now))

    print('training finished')
    if not os.path.exists(output_dir + 'final_model'):
        os.mkdir(output_dir + 'final_model')
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir + 'final_model')
    # torch.save(scheduler.state_dict(), output_dir + 'final_model/scheduler.pt')
    # torch.save(optimizer.state_dict(), output_dir + 'final_model/optimizer.pt')


if __name__ == '__main__':
    main()
