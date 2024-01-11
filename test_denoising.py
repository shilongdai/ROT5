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
        '''Once upon a time, there was a little car named Beep. Beep loved to go fast and play in the sun. Beep was a healthy car because he always had good fuel. Good fuel made Beep happy and strong. One day, Beep was driving in the park when he saw a big tree. The tree had many leaves that were falling. Beep liked how the leaves fall and wanted to play with them. Beep drove under the tree and watched the leaves fall on him. He laughed and beeped his horn. Beep played with the falling leaves all day. When it was time to go home, Beep knew he needed more fuel. He went to the fuel place and got more healthy fuel. Now, Beep was ready to go fast and play again the next day. And Beep lived happily ever after.''',
        '''One day, a little fish named Fin was swimming near the shore. He saw a big crab and wanted to be friends. "Hi, I am Fin. Do you want to play?" asked the little fish. The crab looked at Fin and said, "No, I don't want to play. I am cold and I don't feel fine." Fin felt sad but wanted to help the crab feel better. He swam away and thought of a plan. He remembered that the sun could make things warm. So, Fin swam to the top of the water and called to the sun, "Please, sun, help my new friend feel fine and not freeze!" The sun heard Fin's call and shone its warm light on the shore. The crab started to feel better and not so cold. He saw Fin and said, "Thank you, little fish, for making me feel fine. I don't feel like I will freeze now. Let's play together!" And so, Fin and the crab played and became good friends.''',
        '''Once upon a time, in a land full of trees, there was a little cherry tree. The cherry tree was very sad because it did not have any friends. All the other trees were big and strong, but the cherry tree was small and weak. The cherry tree was envious of the big trees. One day, the cherry tree felt a tickle in its branches. It was a little spring wind. The wind told the cherry tree not to be sad. The wind said, "You are special because you have sweet cherries that everyone loves." The cherry tree started to feel a little better. As time went on, the cherry tree grew more and more cherries. All the animals in the land came to eat the cherries and play under the cherry tree. The cherry tree was happy because it had many friends now. The cherry tree learned that being different can be a good thing. And they all lived happily ever after.'''
    ]
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
