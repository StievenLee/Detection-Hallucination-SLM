# tes.py
# from src.utils.data_loader import load_facqa, load_wrete, load_bioasq, load_trivia_qa

# print('=== FacQA ===')
# for s in load_facqa('data/raw/facqa/train_preprocess.csv', n=3):
#     print('Q:', s['question'][:80])
#     print('A:', s['answer'])
#     print()

# print('=== WReTE ===')
# for s in load_wrete('data/raw/wrete/train_preprocess.csv', n=3):
#     print('Q:', s['question'][:80])
#     print('A:', s['answer'])
#     print()

# print('=== BioASQ ===')
# for s in load_bioasq(n=3):
#     print('Q:', s['question'][:80])
#     print('A:', s['answer'])
#     print()



from src.utils.data_loader import _normalize, is_correct

sample = {
    'answer': 'th monocarboxylate transporter 8 (mct8) mutation is implicated in the th resistance syndrome',
    'answer_aliases': []
}
# Simulasi jawaban model
preds = ['MCT8', 'monocarboxylate transporter 8', 'mct8 is the transporter', 'Paris']
for p in preds:
    print(f'pred={p!r:40} → correct={is_correct(p, sample)}')
