import torch
import logging
from tqdm import tqdm

from transformers import AutoTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from rouge_score.rouge_scorer import RougeScorer
from nltk import bleu, meteor
import numpy as np

from data_process.openrule_dataset_processed import *


logging.basicConfig(
    filename='logs/openrule.log',
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)
config = {
    'G_MODEL_NAME': 'C:/Users/86183/Desktop/RLHF/checkpoints/generate/',
    'E_MODEL_NAME': 'C:/Users/86183/Desktop/RLHF/checkpoints/extract/',
}

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# Reward model
tokenizer = AutoTokenizer.from_pretrained('./checkpoints/rewardmodel_best/')
R_model = torch.load('./checkpoints/rewardmodel_best/model.pt')
R_model.to(device).eval()

# 文本生成模型
G_gpt2_model = GPT2LMHeadModel.from_pretrained(config['G_MODEL_NAME'])
G_gpt2_tokenizer = AutoTokenizer.from_pretrained(config['G_MODEL_NAME'])
G_gpt2_tokenizer.eos_token = G_gpt2_tokenizer.pad_token
G_gpt2_model.to(device)

#规则抽取模型
E_gpt2_model = GPT2LMHeadModel.from_pretrained(config['E_MODEL_NAME'])
E_gpt2_tokenizer = AutoTokenizer.from_pretrained(config['E_MODEL_NAME'])
E_gpt2_tokenizer.eos_token = E_gpt2_tokenizer.pad_token
E_gpt2_model.to(device)


gen_len = 512
gen_kwargs = {
    "min_length":-1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    # "pad_token_id": G_gpt2_tokenizer.eos_token_id
}

ext_len = 200
ext_kwargs = {
    "min_length": -1,
    "top_k": 50,
    "top_p": 0.8,
    "do_sample": True,
    "pad_token_id": E_gpt2_tokenizer.eos_token_id
}

def inference(prompt: str, model, tokenizer, mode: str):
    """
    根据prompt生成内容。
    """
    inputs = tokenizer(prompt, return_tensors='pt')
    if mode == 'generate':
        response = model.generate(inputs['input_ids'].to(device),
                                  max_new_tokens=gen_len, **gen_kwargs)
        r = response.squeeze()[-gen_len:]
        r_str = tokenizer.decode(r)
        # 构建正则表达式，查找特定子字符串后的所有字符
        pattern = re.compile("then what other relationships can we derive between A and B\?" + r'(.*)$', re.DOTALL)
        # 使用 findall 方法查找所有匹配项
        substrings_list = pattern.findall(r_str)
        # 如果有匹配项，则将第一个匹配项的第一个分组中的子字符串列表合并为单个字符串
        if substrings_list:
            result = ' '.join(substrings_list[0].split())
        else:
            result = r_str
    elif mode == 'extract':
        response = model.generate(inputs['input_ids'].to(device),
                                  max_new_tokens=ext_len, **ext_kwargs)
        r = response.squeeze()[-ext_len:]
        r_str = tokenizer.decode(r)
        result = r_str
    return result

def infer_processed(resp):
    rel1 = r'\((.*?)\)'
    rel1_result = re.findall(rel1, resp)
    re_result = []
    for word in rel1_result:
        if word != '' and 'A' in word and 'B' in word:
            re_result.append(word)
    re_result = sorted(set(re_result), key=re_result.index)
    return re_result

def get_chain(txt):
    l = []
    for i in txt:
        l.append(i[0])
    return l


def infer_rule(chain_rel,A_type, B_type, max_step):
    reward_sentences = []
    premise_rels = ','.join(chain_rel)
    G_prompt = f'If A is a {A_type},B is a {B_type},and {premise_rels},then what other relationships can we derive between A and B?'
    passage1 = inference(G_prompt, G_gpt2_model, G_gpt2_tokenizer, mode='generate')
    E_prompt = "Please extract relationships from the given passage:'{}'<|endoftext|>".format(passage1)
    passage2 = inference(E_prompt, E_gpt2_model, E_gpt2_tokenizer, mode='extract')
    select_re = infer_processed(passage2)
    if len(select_re) == 0:
        return
    for word in select_re:
        reward_sentences.append(f"If A is a {A_type},B is a {B_type},and {premise_rels}," + ' we can get ' + word)
    inputs = tokenizer(
        reward_sentences,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    r = R_model(**inputs)
    r_list = r.tolist()

    appended_re = select_re[r_list.index(max(r_list))]
    chain_rel.append(appended_re)
    if len(chain_rel) <= max_step:
        infer_rule(chain_rel,A_type, B_type, max_step)
    else:
        return

scorer = RougeScorer(['rougeL'], use_stemmer=True)
# 计算Rouge指标
def rouge(references, hy_chain):
    scores = []
    for reference in references:
        scores.append(
            scorer.score(
                reference,
                hy_chain)['rougeL'][2]
        )

    return max(scores)

#计算self-BLEU-2
def self_bleu(hy_chain):
    bleus = []
    for i in range(len(hy_chain)):
        bleus.append(bleu(
            hy_chain[:i] + hy_chain[i + 1:],
            hy_chain[i],
            weights=(0.5, 0.5)))

    ret = np.mean(bleus)
    return ret

def print_metric(self, task, metrics):
    logger.info("Task: {}".format(str(task)))
    for k, v in metrics.items():
        logger.info("{}: {}".format(k, str(np.mean(v))))

#Dataloader
LOADER = get_data_loader('C:\\Users\\86183\\Desktop\\reasoning_dataset\\csv_data\\openrule.txt')

with torch.no_grad():
    metrics = {
        "bleu-4": [],
        "bleu-3": [],
        "bleu-2": [],
        "bleu-1": [],
        "METEOR": [],
        "ROUGE-L": [],
        "self-BLEU-2": [],
    }

    with tqdm(total=LOADER.__len__()) as pbar:
    #txt[0]:root_re  txt[1]:rule_chain  txt[2]:A_type  txt[3]:B_type
        for idx, txt in enumerate(LOADER):
            pbar.update(1)
            inputs = re.sub("<A>|<B>", "<mask>", txt[0][0])
            references = [relation.replace('<A>', '<mask>').replace('<B>', '<mask>').lower().strip()
                          for relation in get_chain(txt[1])] #数据集的rule chain
            max_step = len(txt[1])+1
            hy_chain = [txt[0][0].replace('<A>', 'A').replace('<B>', 'B')]   #生成的rule chain
            infer_rule(hy_chain,txt[2][0],txt[3][0],max_step)
            hy_chain  = [relation.replace('A', '<mask>').replace('B', '<mask>').lower().strip()
                          for relation in hy_chain[1:]]
            logger.info("***********Input************")
            logger.info(inputs)
            logger.info("*********hy_chain*********")
            for i, hypo in enumerate(hy_chain):
                logger.info(hypo)

            logger.info("****************************")
            logger.info("*********References*********")
            logger.info(references)
            logger.info("****************************")

            # 统计评估指标
            if len(hy_chain) == 0:
                for k in metrics.keys():
                    if k != 'self-BLEU-2':
                        metrics[k].append(0.)

            else:
                for hypo in hy_chain:
                    try:
                        metrics['bleu-4'].append(
                            bleu(
                                [reference.split() for reference in references],
                                hypo.split(),
                                weights=(0.25, 0.25, 0.25, 0.25)
                            )
                        )
                    except Exception:
                        logger.warning("Skip bleu-4 in example: {}".format(inputs))
                        pass

                    try:
                        metrics['bleu-3'].append(
                            bleu(
                                [reference.split() for reference in references],
                                hypo.split(),
                                weights=(1 / 3,) * 3
                            )
                        )
                    except Exception:
                        logger.warning("Skip bleu-3 in example: {}".format(inputs))
                        pass

                    try:
                        metrics['bleu-2'].append(
                            bleu(
                                [reference.split() for reference in references],
                                hypo.split(),
                                weights=(0.5, 0.5)
                            )
                        )
                    except Exception:
                        logger.warning("Skip bleu-2 in example: {}".format(inputs))
                        pass

                    try:
                        metrics['bleu-1'].append(
                            bleu(
                                [reference.split() for reference in references],
                                hypo.split(),
                                weights=(1.0,)
                            )
                        )
                    except Exception:
                        logger.warning("Skip bleu-1 in example: {}".format(inputs))
                        pass

                    try:
                        metrics['METEOR'].append(
                            meteor(
                                references,
                                hypo,
                            )
                        )
                    except:
                        logger.warning("Skip METEOR in example: {}".format(inputs))
                        pass

                    try:
                        metrics['ROUGE-L'].append(
                            rouge(
                                references,
                                hypo,
                            )
                        )
                    except:
                        logger.warning("Skip ROUGE-L in example: {}".format(inputs))
                        pass
                # try:
                metrics['self-BLEU-2'].append(
                    self_bleu(
                        hy_chain,
                    )
                )
                # except:
                #     logger.warning("Skip self-bleu-2 in example: {}.".format(inputs))
                #     pass
            # break
        for k, v in metrics.items():
            logger.info("{}: {}".format(k, str(np.mean(v))))




