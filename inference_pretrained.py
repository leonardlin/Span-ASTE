# Use pretrained SpanModel weights for prediction
import sys
sys.path.append("aste")
from pathlib import Path
from data_utils import Data, Sentence, SplitEnum
from wrapper import SpanModel

def predict_sentence(text: str, model: SpanModel) -> Sentence:
    path_in = "tmp/temp_in.txt"
    path_out = "tmp/temp_out.txt"
    sent = Sentence(tokens=text.split(), triples=[], pos=[], is_labeled=False, weight=1, id=0)
    data = Data(root=Path(), data_split=SplitEnum.test, sentences=[sent])
    data.save_to_path(path_in)
    model.predict(path_in, path_out)
    data = Data.load_from_full_path(path_out)
    return data.sentences[0]

text = "Did not enjoy the new Windows 8 and touchscreen functions ."
model = SpanModel(save_dir="pretrained_14lap", random_seed=0)
sent = predict_sentence(text, model)

for t in sent.triples:
    target = " ".join(sent.tokens[t.t_start:t.t_end+1])
    opinion = " ".join(sent.tokens[t.o_start:t.o_end+1])
    print()
    print(dict(target=target, opinion=opinion, sentiment=t.label))