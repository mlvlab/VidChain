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
   "/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-all/THIS_FINAL_CAPFIRST_2_conv_dpo_gt_wo_gt_eval5.0_5.0_gt_5.0_5.0.json", # Timestamp 0  
    "/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-all/THIS_FINAL_TIMEFIRST_2_conv_dpo_gt_wo_gt_eval5.0_5.0_gt_5.0_5.0.json", # Caption 1 
    
    "/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-all/THIS_FINAL_CAPFIRST_T2_1_conv_dpo_with_gt_2_eval5.0_5.0_gt_5.0_5.0_t2.json",  #Caption 2
    "/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-all/THIS_FINAL_TIMEFIRST_T2_1_conv_dpo_with_gt_2_eval5.0_5.0_gt_5.0_5.0_t2.json", # Timestamp 3
    
    "/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/Final-generate-samples/THIS_FINAL_CAPFIRST_3_conv_dpo_gt_wo_gt_eval5.0_5.0_gt_5.0_5.0.json", # Timestamp 4
    "/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/Final-generate-samples/THIS_FINAL_TIMEFIRST_3_conv_dpo_gt_wo_gt_eval5.0_5.0_gt_5.0_5.0.json" # Caption 5
]

GT_CONDITIONED = [ 
    "/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-multi/THIS_FINAL_T2_CAPFIRST_2_conv_dpo_with_gt_2_eval10.0_5.0_gt_5.0_5.0_t2.json",  #caption 0
    "/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-multi/THIS_FINAL_T2_TIMEFIRST_2_conv_dpo_with_gt_2_eval10.0_5.0_gt_5.0_5.0_t2.json", # Timestamp 1
    
    "/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples/THIS_FINAL_CAPFIRST_4_conv_dpo_with_gt_3_eval5.0_5.0_gt_5.0_5.0.json", # Timestamp 2
    "/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples/THIS_FINAL_TIMEFIRST_4_conv_dpo_with_gt_3_eval5.0_5.0_gt_5.0_5.0.json", # Caption 3
    
    "/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-multi-not-5000/THIS_FINAL_CAPFIRST_5_conv_dpo_with_gt_3_eval5.0_5.0_gt_5.0_5.0.json", # Timestamp 4
    "/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-multi-not-5000/THIS_FINAL_TIMEFIRST_5_conv_dpo_with_gt_3_eval5.0_5.0_gt_5.0_5.0.json", # Caption 5
]



data_g = [json.load(open(path)) for path in GENERATED]
data_c = [json.load(open(path)) for path in GT_CONDITIONED]

print([len(d['vid']) for d in data_g+data_c])
# joint_data = [data_g[2]+data_c[2]+data_g[1]+data_c[1]+data_g[0]+data_c[0]]
# final_dict  = {k:data_c[2][k]+data_g[2][k]+data_g[1][k]+data_c[1][k]+data_g[0][k]+data_c[0][k] for k in ['prompt', 'chosen', 'rejected', 'vid', 'logits']}
# "/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-all/THIS_FINAL_CAPFIRST_T2_1_conv_dpo_with_gt_2_eval5.0_5.0_gt_5.0_5.0_t2.json", 
final_dict  = {k:data_c[3][k] + data_g[3][k] + data_c[3][k] + data_c[5][k] + data_g[1][k] + data_g[5][k] + data_g[0][k] + data_g[4][k] + data_c[2][k] + data_c[4][k] + data_g[2][k] +  data_c[2][k] for k in ['prompt', 'chosen', 'rejected', 'vid', 'logits']}
print([len(v) for i,v in final_dict.items()])


breakpoint()
breakpoint()
# with open( f'/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-concat-wo-gt-btw-generated-{eval_threshold}.json', 'w') as f: f.write(json.dumps(final_dict, indent=2))
with open( f'/home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-concat-JOINT-FINAL-CAP2TIME-T2.json', 'w') as f: f.write(json.dumps(final_dict, indent=2))

breakpoint()

# /home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-multi-not-5000/conv_dpo_gt_wo_gt_eval10.0.json
# /home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-multi-not-5000/conv_dpo_gt_wo_gt.json
# /home/jisoo/AAAI2025/VideoLLaMA2/outputs/log/generate-samples-multi-not-5000