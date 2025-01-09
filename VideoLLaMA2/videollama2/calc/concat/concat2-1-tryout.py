import json

file_path = "/home/jisoo/AAAI2025/VideoLLaMA2/outputs/threshold"

GENERATED = [
    
    f"{file_path}/THIS_FINAL_CAPFIRST_T2_1_conv_dpo_with_gt_3_t2.json",  #Caption 2
    f"{file_path}/THIS_FINAL_TIMEFIRST_T2_1_conv_dpo_with_gt_3_t2.json", # Timestamp 3
    
   f"{file_path}/THIS_FINAL_CAPFIRST_2_conv_dpo_gt_wo_gt.json", # Timestamp 0  
    f"{file_path}/THIS_FINAL_TIMEFIRST_2_conv_dpo_gt_wo_gt.json", # Caption 1 
    
    f"{file_path}/THIS_FINAL_CAPFIRST_3_conv_dpo_gt_wo_gt.json", # Timestamp 4
    f"{file_path}/THIS_FINAL_TIMEFIRST_3_conv_dpo_gt_wo_gt.json" # Caption 5
]

GT_CONDITIONED = [ 
    f"{file_path}/THIS_FINAL_T2_CAPFIRST_2_conv_dpo_with_gt_2_t2.json",  #caption 0
    f"{file_path}/THIS_FINAL_T2_TIMEFIRST_2_conv_dpo_with_gt_2_t2.json", # Timestamp 1
    
    f"{file_path}/THIS_FINAL_CAPFIRST_4_conv_dpo_with_gt_3.json", # Timestamp 2
    f"{file_path}/THIS_FINAL_TIMEFIRST_4_conv_dpo_with_gt_3.json", # Caption 3
    
    f"{file_path}/THIS_FINAL_CAPFIRST_5_conv_dpo_with_gt_3.json", # Timestamp 4
    f"{file_path}/THIS_FINAL_TIMEFIRST_5_conv_dpo_with_gt_3.json", # Caption 5
]

data_g = [json.load(open(path)) for path in GENERATED]
data_c = [json.load(open(path)) for path in GT_CONDITIONED]

_len = [len(d['vid']) for d in data_g+data_c]
print(_len)
print(sum(_len))

final_dict = {k:[] for k in ['prompt', 'chosen', 'rejected', 'vid', 'logits']}

for genF, condF in zip(data_g, data_c):
    
    for k in ['prompt', 'chosen', 'rejected', 'vid', 'logits']:
        final_dict[k].extend(genF[k])
        final_dict[k].extend(condF[k])

breakpoint()

# T2 First
# final_dict  = {k:data_c[3][k] + data_g[3][k] + data_c[3][k] + data_c[5][k] + data_g[1][k] + data_g[5][k] + data_g[0][k] + data_g[4][k] + data_c[2][k] + data_c[4][k] + data_g[2][k] +  data_c[2][k] for k in ['prompt', 'chosen', 'rejected', 'vid', 'logits']}
print([len(v) for i,v in final_dict.items()])


breakpoint()
# with open( f'/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-concat-wo-gt-btw-generated-{eval_threshold}.json', 'w') as f: f.write(json.dumps(final_dict, indent=2))
with open( f'/home/jisoo/AAAI2025/VideoLLaMA2/outputs/threshold/JOINT-FINAL-CAP2TIME-T2-threshold.json', 'w') as f: f.write(json.dumps(final_dict, indent=2))

breakpoint()

# /home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-multi-not-5000/conv_dpo_gt_wo_gt_eval10.0.json
# /home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-multi-not-5000/conv_dpo_gt_wo_gt.json
# /home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-multi-not-5000