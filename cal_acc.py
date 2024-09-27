
import pandas as pd

answer_path = './answer_list.csv'
answer_values = pd.read_csv(answer_path).values.tolist()

prob_path = './work/result/result.csv'
prob_values = pd.read_csv(prob_path).values.tolist()

def list_to_dict(a):
  b = dict()
  for k in a:
    b[k[0]] = k[1]
  return b
  
def cal_acc(prob_list, answer_dict):
  n = len(prob_list)
  acc_num = 0
  for _prob in prob_list:
    if answer_dict[_prob[0]] == _prob[1]:
      acc_num += 1
      
  return acc_num / n

answer_dict = list_to_dict(answer_values)
acc = cal_acc(prob_values, answer_dict)

print('acc', acc)