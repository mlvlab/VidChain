import json

file_path = "/home/jisoo/AAAI2025/VideoLLaMA2/outputs/ALL"

GENERATED = [
    
    f"{file_path}/THIS_FINAL_T2_1.json",  #Caption 2
#    f"{file_path}/THIS_FINAL_2.json", # Timestamp 0  
    # f"{file_path}/THIS_FINAL_3.json", # Timestamp 4
]

GT_CONDITIONED = [ 
    f"{file_path}/THIS_FINAL_T2_2.json",  #caption 0
    # f"{file_path}/THIS_FINAL_4.json", # Timestamp 2
    # f"{file_path}/THIS_FINAL_5.json", # Timestamp 4
]

data_g = [json.load(open(path)) for path in GENERATED]
data_c = [json.load(open(path)) for path in GT_CONDITIONED]

_len = [len(d['vid']) for d in data_g+data_c]
print(_len)
print(sum(_len))

final_dict = {k:[] for k in ['prompt', 'chosen', 'rejected', 'vid', 'logits']}

for genF, condF in zip(data_g, data_c):
    
    for k in ['prompt', 'chosen', 'rejected', 'vid', 'logits']:
        # final_dict[k].extend(genF[k])
        final_dict[k].extend(condF[k])

breakpoint()

# T2 First
# final_dict  = {k:data_c[3][k] + data_g[3][k] + data_c[3][k] + data_c[5][k] + data_g[1][k] + data_g[5][k] + data_g[0][k] + data_g[4][k] + data_c[2][k] + data_c[4][k] + data_g[2][k] +  data_c[2][k] for k in ['prompt', 'chosen', 'rejected', 'vid', 'logits']}
print([len(v) for i,v in final_dict.items()])


breakpoint()
# with open( f'/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-concat-wo-gt-btw-generated-{eval_threshold}.json', 'w') as f: f.write(json.dumps(final_dict, indent=2))
with open( f'/home/jisoo/AAAI2025/VideoLLaMA2/outputs/ALL/final-GT-CONDITIONED-T2.json', 'w') as f: f.write(json.dumps(final_dict, indent=2))

breakpoint()

# /home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-multi-not-5000/conv_dpo_gt_wo_gt_eval10.0.json
# /home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-multi-not-5000/conv_dpo_gt_wo_gt.json
# /home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-multi-not-5000