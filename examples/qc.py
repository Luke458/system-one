"""Run the retail QC smoke example."""
import json
from system_one import DecisionModel
from system_one.qc import CASES, FIELDS

if __name__ == "__main__":
    model = DecisionModel.from_pretrained()
    result = model.decide(CASES[0]["context"], FIELDS)
    print(json.dumps({k: v.to_dict() for k, v in result.items()}, indent=2))
