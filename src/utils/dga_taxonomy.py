"""DGA type taxonomy - classify families into char-based vs word-based."""
WORD_BASED_FAMILIES = {"matsnu","suppobox","gozi","pizd","banjori","bigviktor","ngioweb","nymaim2"}

def is_word_based(family: str) -> bool:
    return family.lower().strip() in WORD_BASED_FAMILIES

def classify_family(family: str) -> str:
    if family.lower().strip() == "benign": return "benign"
    return "word-based" if is_word_based(family) else "char-based"

def split_by_dga_type(df, label_col="label", family_col="family"):
    is_benign = df[label_col] == 0
    is_dga    = df[label_col] == 1
    is_word   = df[family_col].apply(is_word_based) & is_dga
    is_char   = ~df[family_col].apply(is_word_based) & is_dga
    return df[is_char | is_benign].copy(), df[is_word | is_benign].copy()
