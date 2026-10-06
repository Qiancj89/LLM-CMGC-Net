import argparse
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
args = parser.parse_args()
checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
state = checkpoint["model_state_dict"]
checkpoint["model_state_dict"] = {
    key: value for key, value in state.items() if not key.startswith("text_encoder.")
}
temporary = args.checkpoint.with_suffix(".tmp")
torch.save(checkpoint, temporary)
temporary.replace(args.checkpoint)
