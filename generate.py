import torch
import torch.nn.functional as F
import os
import argparse
from tqdm import trange
from transformers import GPT2LMHeadModel


def is_word(word):
    for item in list(word):
        if item not in 'qwertyuiopasdfghjklzxcvbnm':
            return False
    return True


def _is_chinese_char(char):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(char)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, context, length, n_ctx, tokenizer, temperature=1.0, top_k=30, top_p=0.0, repitition_penalty=1.0,
                    device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    with torch.no_grad():
        for _ in trange(length):
            inputs = {'input_ids': generated[0][-(n_ctx - 1):].unsqueeze(0)}
            outputs = model(
                **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :]
            for id in set(generated):
                next_token_logits[id] /= repitition_penalty
            next_token_logits = next_token_logits / temperature
            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated.tolist()[0]


def fast_sample_sequence(model, context, length, temperature=1.0, top_k=30, top_p=0.0, device='cpu'):
    """推理, 快速生成文本

    步骤:
    1. 首先使用past存储了除了最后一个token外的所有token的编码状态
    2. 然后将最后一个token单独保存在prev中
    3. 在生成循环的时候, 模型使用prev和past来预测下一个token
    4. 预测出的新token会成为新的prev

    参数:
    * context: 如[8790, 139, 156, 12642, 8179]
    * length: 生成文本的长度
    """
    inputs = torch.LongTensor(context).view(1, -1).to(device)  # 将向量变为列向量
    if len(context) > 1:
        _, past = model(inputs[:, :-1], None)[:2]  # inputs[:, :-1]是将inputs中除了最后一个token之外的tokens作为模型的输入; _表示的是logits, past表示的是模型的隐藏状态. 注意, 这和训练的时候返回的参数是不一样的. past在这里是一个包含10个元素的tuple, 表示10个隐藏层的状态. 这个状态就是之前的上下文信息.
        prev = inputs[:, -1].view(1, -1)
    else:
        past = None
        prev = inputs
    generate = [] + context  # 当前的文本tokens
    with torch.no_grad():
        for i in trange(length):
            output = model(prev, past=past)  # 模型的输出是一个tuple, 其中第一个元素是logits, 第二个元素是当前的past(隐藏层状态)
            output, past = output[:2]  # 取出logits和past
            output = output[-1].squeeze(0) / temperature  # 将logits除以温度, 使得生成的文本更加随机
            filtered_logits = top_k_top_p_filtering(output, top_k=top_k, top_p=top_p)  # 过滤logits, 只保留top_k和top_p的token
            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)  # 对filtered_logits进行softmax操作, 将logits转换为概率分布, 然后从得到的概率分布中进行采样, 采样的过程基于概率分布的随机选择, 概率越高的词被选中的可能性越大.
            generate.append(next_token.item())  # 更新当前的文本tokens
            prev = next_token.view(1, 1)  # 预测出的新token成为新的prev
    return generate


# 通过命令行参数--fast_pattern, 指定模式
def generate(n_ctx, model, context, length, tokenizer, temperature=1, top_k=0, top_p=0.0, repitition_penalty=1.0, device='cpu',
             is_fast_pattern=False):
    if is_fast_pattern:  # 使用更快的方式产生文本
        return fast_sample_sequence(model, context, length, temperature=temperature, top_k=top_k, top_p=top_p,
                                    device=device)
    else:
        return sample_sequence(model, context, length, n_ctx, tokenizer=tokenizer, temperature=temperature, top_k=top_k, top_p=top_p,
                               repitition_penalty=repitition_penalty, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='生成设备')
    parser.add_argument('--length', default=-1, type=int, required=False, help='生成长度')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='生成的batch size')
    parser.add_argument('--nsamples', default=10, type=int, required=False, help='生成几个样本')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成温度')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高几选一')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
                        help='模型参数')
    parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False, help='词表路径')
    parser.add_argument('--model_path', default='model/final_model', type=str, required=False, help='模型路径')
    parser.add_argument('--prefix', default='萧炎', type=str, required=False, help='生成文章的开头')
    parser.add_argument('--no_wordpiece', action='store_true', help='不做word piece切词')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    parser.add_argument('--fast_pattern', action='store_true', help='采用更加快的方式生成文本')
    parser.add_argument('--save_samples', action='store_true', help='保存产生的样本')
    parser.add_argument('--save_samples_path', default='.', type=str, required=False, help="保存样本的路径")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False)

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    if args.segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    length = args.length
    batch_size = args.batch_size
    nsamples = args.nsamples
    temperature = args.temperature
    topk = args.topk
    topp = args.topp
    repetition_penalty = args.repetition_penalty
    """
    * length: 生成文本的长度, 即生成多少个 token 作为输出.
    * batch_size: 每次生成或训练时所使用的批量大小, 表示一次生成多少个样本或一次处理多少个样本.
    * nsamples: 需要生成的样本总数, 程序会根据批量大小分批生成, 总数达到 nsamples 后结束.
    * temperature: 控制生成时采样随机性的温度参数. 较低的温度(如 0.7)会使生成更加确定(重复率较高), 而较高的温度(如 1.0 以上)会增加采样的随机性, 从而提高输出的多样性.
    * topk: top-k 采样策略中的 k 值, 即在生成每个 token 时, 只从概率最高的 k 个候选 token 中选择, 从而限制候选范围, 提高文本质量.
    * topp: nucleus sampling(顶核采样)策略中的 p 值, 即保留累计概率达到 p 的最小候选集合后, 从中采样下一个 token, 这种方法在动态调整候选 token 数量上比 top-k 更灵活.
    * repetition_penalty: 重复惩罚因子, 用于降低已生成 token 重复出现的概率, 确保生成的文本更加多样化.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    n_ctx = model.config.n_ctx

    if length == -1:
        length = model.config.n_ctx
    if args.save_samples:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
        samples_file = open(args.save_samples_path + '/samples.txt', 'w', encoding='utf8')
    while True:
        raw_text = args.prefix  # 输入的prompt
        context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_text))  # 将prompt转为token id
        generated = 0  # 表示当前的批次
        for _ in range(nsamples // batch_size):
            out = generate(  # 推理
                n_ctx=n_ctx,
                model=model,
                context=context_tokens,
                length=length,
                is_fast_pattern=args.fast_pattern, tokenizer=tokenizer,
                temperature=temperature, top_k=topk, top_p=topp, repitition_penalty=repetition_penalty, device=device
            )
            for i in range(batch_size):
                generated += 1
                text = tokenizer.convert_ids_to_tokens(out)
                for i, item in enumerate(text[:-1]):  # 确保英文前后有空格
                    if is_word(item) and is_word(text[i + 1]):
                        text[i] = item + ' '
                for i, item in enumerate(text):
                    if item == '[MASK]':
                        text[i] = ''
                    elif item == '[CLS]':
                        text[i] = '\n\n'
                    elif item == '[SEP]':
                        text[i] = '\n'
                info = "=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + "\n"
                print(info)
                text = ''.join(text).replace('##', '').strip()
                print(text)
                if args.save_samples:
                    samples_file.write(info)
                    samples_file.write(text)
                    samples_file.write('\n')
                    samples_file.write('=' * 90)
                    samples_file.write('\n' * 2)
        print("=" * 80)
        if generated == nsamples:
            # close file when finish writing.
            if args.save_samples:
                samples_file.close()
            break


if __name__ == '__main__':
    main()
