import json

# eval_threshold = 10

'''
### GENERATED 
/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-all/FINAL_T2_1_conv_dpo_with_gt_2_eval5.0_5.0_gt_5.0_5.0_t2.json
/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-all/FINAL_2_conv_dpo_gt_wo_gt_eval5.0_5.0_gt_5.0_5.0.json
/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/Final-generate-samples/FINAL_3_conv_dpo_gt_wo_gt_eval5.0_5.0_gt_5.0_5.0.json

### GT CONDITIONED
/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-multi/FINAL_T2_2_conv_dpo_with_gt_2_eval10.0_5.0_gt_5.0_5.0_t2.json
/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples/FINAL_4_conv_dpo_with_gt_3_eval5.0_5.0_gt_5.0_5.0.json
/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-multi-not-5000/FINAL_5_conv_dpo_with_gt_3_eval5.0_5.0_gt_5.0_5.0.json
'''


GENERATED = [
    "/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-all/FINAL_T2_1_conv_dpo_with_gt_2_eval5.0_5.0_gt_5.0_5.0_t2.json", 
    "/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-all/FINAL_2_conv_dpo_gt_wo_gt_eval5.0_5.0_gt_5.0_5.0.json", 
    "/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/Final-generate-samples/FINAL_3_conv_dpo_gt_wo_gt_eval5.0_5.0_gt_5.0_5.0.json"
]

GT_CONDITIONED = [
    "/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-multi/FINAL_T2_2_conv_dpo_with_gt_2_eval10.0_5.0_gt_5.0_5.0_t2.json",
    "/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples/FINAL_4_conv_dpo_with_gt_3_eval5.0_5.0_gt_5.0_5.0.json",
    "/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-multi-not-5000/FINAL_5_conv_dpo_with_gt_3_eval5.0_5.0_gt_5.0_5.0.json"
]



data_g = [json.load(open(path)) for path in GENERATED]
data_c = [json.load(open(path)) for path in GT_CONDITIONED]

print([len(d['vid']) for d in data_g+data_c])
# joint_data = [data_g[2]+data_c[2]+data_g[1]+data_c[1]+data_g[0]+data_c[0]]
# final_dict  = {k:data_c[2][k]+data_g[2][k]+data_g[1][k]+data_c[1][k]+data_g[0][k]+data_c[0][k] for k in ['prompt', 'chosen', 'rejected', 'vid', 'logits']}
final_dict  = {k:data_g[2][k]+data_g[1][k]+data_c[2][k]+data_c[1][k]+data_c[0][k]+data_g[0][k] for k in ['prompt', 'chosen', 'rejected', 'vid', 'logits']}

# final_dict  = {k:t3_gt_generated[k] for k in ['prompt', 'chosen', 'rejected', 'vid', 'logits']}

# final_dict = {}
# final_dict['prompt'] = t3_1['prompt'] + t3_2['prompt'] + t2['prompt']
# final_dict['chosen'] = t3_1['chosen'] + t3_2['chosen'] + t2['chosen']
# final_dict['rejected'] = t3_1['rejected'] + t3_2['rejected'] + t2['rejected']
# final_dict['vid'] = t3_1['vid'] + t3_2['vid'] + t2['vid']
# final_dict['logits'] = t3_1['logits'] + t3_2['logits'] + t2['logits']


# /home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-multi-not-5000/conv_dpo_gt_gt_2.json
# VideoLLaMA2/outputs/log/Final-generate-samples/conv_dpo_gt_wo_gt.json
# /home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/Final-generate-samples/capfirst_score_conv_dpo1.json
print([len(v) for i,v in final_dict.items()])


breakpoint()
breakpoint()
# with open( f'/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-concat-wo-gt-btw-generated-{eval_threshold}.json', 'w') as f: f.write(json.dumps(final_dict, indent=2))
with open( f'/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-concat-JOINT-FINAL-ORDER.json', 'w') as f: f.write(json.dumps(final_dict, indent=2))

breakpoint()

# /home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-multi-not-5000/conv_dpo_gt_wo_gt_eval10.0.json
# /home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-multi-not-5000/conv_dpo_gt_wo_gt.json
# /home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-multi-not-5000