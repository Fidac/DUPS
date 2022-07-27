from pathlib import Path

from scripts.utils import Collection
from src.ehealth20 import MAJA2020

# === Check on src.ehealth20 script, the 'run' function in the 'MAJA2020' class, to find the
# === directories where the trained models are going to be loaded from, as well as their configs

# === It is important the the config file matches exactly with the dumped model, otherwise
# === there is going to be missmatches in layer sizes and the models structure

# Path to the sentences collection in the eHealth input format
collection_path = Path("")

# Directory to dump the results
dump_dir = Path("")

# Loading the collection
collection = Collection()
collection.load(collection_path)


# Models ablation configuration
taskA_ablation = {
    "bert_embedding": True,
    "word_embedding": False,
    "chars_info": True,
    "postag": True,
    "dependency": False,
}

taskB_ablation = {
    "bert_embedding": True,
    "word_embedding": False,
    "chars_info": True,
    "postag": True,
    "dependency": True,
    "entity_type": True,
    "entity_tag": True
}

# Loading and running the pipeline
algorithm = MAJA2020(taskA_ablation, taskB_ablation)
collection = algorithm.run(collection, taskA=True, taskB=True)

# Dumping the results
dump_path = dump_dir / "scenario.txt"
collection.dump(dump_path)
