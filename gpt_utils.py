import torch

def generate_gpt_ans(ques, tokenizer, model):
    prompt = f"<|startoftext|><|question|>{ques}<|answer|>"
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    sample_outputs = model.generate(
                                generated, 
                                #bos_token_id=random.randint(1,30000),
                                do_sample=True,   
                                top_k=50, 
                                max_length = 300,
                                top_p=0.95, 
                                num_return_sequences=1
                                )
    output = sample_outputs[0]      
    ans = tokenizer.decode(output, skip_special_tokens=True)   
    ans_token_index = ans.index("<|answer|>")
    # parse out the ans only
    ans = ans[ans_token_index+10:]
    return ans
