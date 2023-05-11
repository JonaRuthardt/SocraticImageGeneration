import pandas as pd

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice

spice = Spice()
blue = Bleu(4)
meteor = Meteor()
rouge = Rouge()

scorers = [spice, blue, meteor, rouge]

results_df = pd.read_csv(r'data/results/coco_small_similarity_termination/results.csv', index_col=0)

gts = {}
res = {}
for idx, row in results_df.iterrows():
    gts[row['prompt_id']] = [row['original_prompt']]
    res[row['prompt_id']] = [row['optimized_caption']]

results = {}
for scorer in scorers:
    avg_score, scores = scorer.compute_score(gts, res)
    results[scorer.method()] = avg_score

