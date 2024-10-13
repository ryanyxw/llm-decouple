import numpy as np

def tokenize_with_hate_loss_span_masking(line, toxic_threshold, safe_threshold, tokenizer):
    """
    tokenizes the text with loss mask using toxic classification of spans of the text
    :param text: either a string or a dictionary with a key "text"
    :param tokenizer: hf tokenizer
    :return:
    """
    text = line["text"][0]
    spans_with_toxic_level = line["toxic_spans"][0]

    #we first split the text into sentences and record whether each is toxic or not
    splitted_text = []
    span_label_mask = []
    prev_end = 0
    for span in spans_with_toxic_level:
        # we assign the label of in-between text to the next span (to keep consistent tokenization and whitespace)
        splitted_text.append(text[min(prev_end, int(span[0])):int(span[1])])
        if span[2] > toxic_threshold:
            span_label_mask.append(3)
        elif span[2] > safe_threshold:
            span_label_mask.append(2)
        else:
            span_label_mask.append(1)
        prev_end = int(span[1])

    if prev_end < len(text):
        splitted_text.append(text[prev_end:])
        span_label_mask.append(1)

    # if the text is all toxic or only the last span is toxic, we skip the text
    if sum(span_label_mask) == 0 or (span_label_mask[-1] == 0 and sum(span_label_mask) == (len(span_label_mask)-1)):
        return {"input_ids": [],
                "loss_mask": [],
                "attention_mask": []}

    input_ids_list = tokenizer(splitted_text, add_special_tokens=False, return_tensors="np")["input_ids"]

    concatenated_input_ids = []
    concatenated_loss_mask = []

    for input_ids, label_mask in zip(input_ids_list, span_label_mask):
        concatenated_input_ids.extend(input_ids.tolist())
        concatenated_loss_mask.extend([label_mask] * len(input_ids))

    return {"input_ids": [concatenated_input_ids],
            "loss_mask": [concatenated_loss_mask],
            "attention_mask": [[1] * len(concatenated_input_ids)]}

def tokenize_with_hate_loss_masking(text, tokenizer, percentage_backprop=0, bad_words=[]):
    """
    tokenizes the text with loss mask, but ensures that the bad words are not going to be backpropped
    :param text: either a string or a dictionary with a key "text"
    :param tokenizer: hf tokenizer
    :param percentage_backprop: double
    :param bad_words: list of strings
    :return:
    """
    #check if text is a dictionary (when we directly apply hf map function
    if not isinstance(text, str):
        text = text["text"]

    input_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="np")[0]

    #create the loss mask. 1 is where we are training on that token
    loss_mask = np.zeros_like(input_ids, dtype=int)
    loss_mask[:int(len(input_ids) * percentage_backprop)] = 1
    np.random.shuffle(loss_mask)

    backprop_token_pos = np.argwhere(loss_mask).flatten()
    backprop_tokens = tokenizer.batch_decode(input_ids[backprop_token_pos])

    for i in range(len(backprop_tokens)):
        if backprop_tokens[i].lower().strip() in bad_words:
            loss_mask[backprop_token_pos[i]] = 0

    return {"input_ids": input_ids.tolist(),
            "loss_mask": loss_mask.tolist(),
            "attention_mask": [1] * len(input_ids)}



