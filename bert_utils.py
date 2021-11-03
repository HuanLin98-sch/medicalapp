from transformers import BertTokenizer

import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

def create_mbert_tokens(title):
    # print(f"preparing embedding for {title}")
    tokens = {'input_ids': [], 'attention_mask': []}
    new_tokens = tokenizer.encode_plus(title, max_length=32, truncation=True,
                                            padding='max_length', return_tensors='pt')

    # tokens['input_ids'] = new_tokens['input_ids'][0]
    # tokens['attention_mask'] = new_tokens['attention_mask'][0]
    tokens['input_ids'].append(new_tokens['input_ids'][0])
    tokens['attention_mask'].append(new_tokens['attention_mask'][0])

    # reformat list of tensors into single tensor
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
    # print(tokens["attention_mask"])
    return tokens['input_ids'], tokens['attention_mask']

def bert_embed_gen(title, model):
    model = model.bert
    tokens = create_mbert_tokens(title)
    token_mapping = {
        "input_ids": tokens[0],
        "attention_mask": tokens[1]
    }
    # print(tokens)
    # passing the tokens into model to get the pre-trained embeddings
    outputs = model(**token_mapping)
    # print(outputs)
    # outputs = self.model(new_tokens['input_ids'][0], attention_mask=new_tokens['attention_mask'][0])
    embeddings = outputs.last_hidden_state
    # print(embeddings)

    # resizing the attention mask to the correct dimensions
    attention_mask = token_mapping['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()

    # applying the mask onto the embeddings
    masked_embeddings = embeddings * mask

    # converting the multuple token embeddings into a single embedding (ie sentence embedding)
    # this is done by taking the mean of the different token embeddings
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask

    mean_pooled = mean_pooled.detach().numpy()
    
    return mean_pooled