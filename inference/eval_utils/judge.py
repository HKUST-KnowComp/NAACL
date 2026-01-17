import re
import string
import random
from collections import Counter, defaultdict

def normalize_answer(s):
    """标准化文本：转小写、移除标点、删除冠词、规整空格"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, ground_truth):
    """
    计算预测文本与真实文本之间的F1分数
    Args:
        prediction (str): 模型预测文本
        ground_truth (str): 真实参考文本
    
    Returns:
        float: F1分数 (范围0.0-1.0)
    """
    # 文本标准化处理
    normalized_pred = normalize_answer(prediction)
    normalized_gt = normalize_answer(ground_truth)
    
    # 处理空文本情况
    if not normalized_pred or not normalized_gt:
        return 0.0
    
    # 分词并计算词频
    pred_tokens = normalized_pred.split()
    gt_tokens = normalized_gt.split()
    common_tokens = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common_tokens.values())
    
    # 无共同词时返回0
    if num_same == 0:
        return 0.0
    
    # 计算精确率和召回率
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    
    # 计算F1分数（调和平均数）
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def f1_judge(pred, truth, threshold=0.8, return_score=False):
    if not pred or not truth:
        return 0.0 if return_score else False
    
    if not return_score:
        # Handle boolean case
        if isinstance(truth, bool) or isinstance(truth[0], bool):
            pred = pred.lower().strip()
            truth_bool = truth if isinstance(truth, bool) else truth[0]
            if 'yes' in pred:
                pred_bool = True
            elif 'no' in pred:
                pred_bool = False
            else:
                return False
            return pred_bool == truth_bool
            
        if isinstance(truth, str):
            pred = pred.lower().strip()
            truth = truth.lower().strip()
            f1_score = compute_f1(pred, truth)

            if truth in pred or f1_score >= threshold:
                return True

        elif isinstance(truth, list):
            correct = False
            for answer in truth:
                if answer == "N/A":
                    continue
                pred = pred.lower().strip()
                answer = answer.lower().strip()
                f1_score = compute_f1(pred, answer)
                if answer in pred or f1_score >= threshold:
                    correct = True
            return correct
        
        return False
    
    else:
        # Handle boolean case
        if isinstance(truth, bool) or isinstance(truth[0], bool):
            pred = pred.lower().strip()
            truth_bool = truth if isinstance(truth, bool) else truth[0]
            if 'yes' in pred:
                pred_bool = True
            elif 'no' in pred:
                pred_bool = False
            else:
                return 0.0
            return 1.0 if pred_bool == truth_bool else 0.0
            
        if isinstance(truth, str):
            pred = pred.lower().strip()
            truth = truth.lower().strip()
            f1_score = compute_f1(pred, truth)
            return f1_score

        elif isinstance(truth, list):
            f1_scores = []
            for answer in truth:
                pred = pred.lower().strip()
                answer = answer.lower().strip()
                f1_score = compute_f1(pred, answer)
                f1_scores.append(f1_score)
            return max(f1_scores)
        
        return 0.0

def is_same_answer(pred1, pred2):
    return normalize_answer(pred1) == normalize_answer(pred2)

def aggregate_answers(answer_conf_pairs):
    """
    从多个答案-置信度对中选择最终答案
    
    Args:
        answer_conf_pairs: list of tuples, 格式为 [(ans1, conf1), (ans2, conf2), ...]
    
    Returns:
        tuple: (final_answer, final_confidence) 最终选择的答案和置信度
    """
    if not answer_conf_pairs:
        return None, 0.0
    
    # 使用字典存储每个答案组的统计信息
    # key: 代表答案（使用第一个出现的答案作为代表）
    # value: {'count': 频率, 'total_conf': 总置信度, 'representative': 代表答案}
    answer_groups = {}
    
    # 将答案分组
    for ans, conf in answer_conf_pairs:
        # 查找是否已有相同的答案组
        found_group = False
        for group_key in answer_groups:
            if is_same_answer(ans, group_key):
                answer_groups[group_key]['count'] += 1
                answer_groups[group_key]['total_conf'] += conf
                found_group = True
                break
        
        # 如果没有找到相同的组，创建新组
        if not found_group:
            answer_groups[ans] = {
                'count': 1,
                'total_conf': conf,
                'representative': ans
            }
    
    # 计算每个组的平均置信度
    group_stats = []
    for group_key, stats in answer_groups.items():
        avg_conf = stats['total_conf'] / stats['count']
        group_stats.append({
            'answer': stats['representative'],
            'count': stats['count'],
            'avg_conf': avg_conf
        })
    
    # 按频率降序排序，频率相同则按置信度降序排序
    group_stats.sort(key=lambda x: (-x['count'], -x['avg_conf']))
    
    # 找出频率最高的组
    max_count = group_stats[0]['count']
    max_count_groups = [g for g in group_stats if g['count'] == max_count]
    
    # 如果有多个组频率相同，取置信度最高的
    if len(max_count_groups) > 1:
        max_conf = max_count_groups[0]['avg_conf']
        max_conf_groups = [g for g in max_count_groups if g['avg_conf'] == max_conf]
        
        # 如果置信度也相同，随机选择一个
        if len(max_conf_groups) > 1:
            selected = random.choice(max_conf_groups)
        else:
            selected = max_conf_groups[0]
    else:
        selected = max_count_groups[0]
    
    return selected['answer'], selected['avg_conf']

