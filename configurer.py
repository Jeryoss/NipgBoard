import sys
import json
import os

def main(argv):
  workspace = argv[0]
  config = {"default": {"name": "DEFAULT", "embedding_folder": "<default_embedding_folder>", "image_folder": "<image_folder>"}, "training": {"name": "TRAINING", "embedding_folder": "<training_embedding_folder>", "algorithm_path": os.path.join(workspace,"algorithms"), "algorithm": {"file": "kira", "callable": "run", "arguments": [], "keyword_arguments": {"epoch": "10"}}}}
  with open('cnf.json', 'w') as outfile:
     json.dump(config, outfile, indent=4)

if __name__ == "__main__":
    main(sys.argv[1:])
