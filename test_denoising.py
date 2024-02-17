from transformers import AutoTokenizer

from text_denoising import DataCollatorForUL2

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer.padding_side = "right"
    for i in range(500):
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [f"[MASK-{500 - i - 1}]"]})
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[REBERT]"]})
    sink_token = tokenizer.encode("[REBERT]", add_special_tokens=False)[0]
    if not tokenizer.pad_token_id:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    print(f"Final Vocab Size: {len(tokenizer)}")
    collate_fn = DataCollatorForUL2(tokenizer, sentinel_map=lambda x: sink_token - x, decoder_start_token_id=sink_token)

    batch = [
        '''One day, a little girl named Lily found a needle in her room. She knew it was difficult to play with it because it was sharp. Lily wanted to share the needle with her mom, so she could sew a button on her shirt. Lily went to her mom and said, "Mom, I found this needle. Can you share it with me and sew my shirt?" Her mom smiled and said, "Yes, Lily, we can share the needle and fix your shirt." Together, they shared the needle and sewed the button on Lily's shirt. It was not difficult for them because they were sharing and helping each other. After they finished, Lily thanked her mom for sharing the needle and fixing her shirt. They both felt happy because they had shared and worked together.''',
    ] * 10
    encode = collate_fn([{'input_ids': tokenizer(r)['input_ids']} for r in batch])
    print(tokenizer.decode(tokenizer(batch[0])['input_ids']))
    print('-----')
    for input_ids, token_ids, label_ids in zip(encode['input_ids'], encode['decoder_input_ids'], encode['labels']):
        print('---------')
        print("Input")
        print(tokenizer.decode(input_ids))
        print("Decoder Input")
        print(tokenizer.decode(token_ids))
        print("Label")
        print(tokenizer.decode(label_ids[label_ids != -100]))
        print('---------')
