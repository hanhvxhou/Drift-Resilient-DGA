import pandas as pd
from src.utils.dga_taxonomy import is_word_based

df = pd.read_csv("data/processed/benchmark/D24.csv")
dga = df[df.label == 1]
word_count = dga["family"].apply(is_word_based).sum()
char_count = len(dga) - word_count
print(f"D24: char-based={char_count}, word-based={word_count}")
print(f"Word-based families: {dga[dga.family.apply(is_word_based)].family.unique()}")