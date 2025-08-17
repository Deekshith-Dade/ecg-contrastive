import argparse
import torch
from models import ECG_TemporalNet1D, ModelGroup
import parameters


parser = argparse.ArgumentParser(description="ECG PreTrained Models")
parser.add_argument('--arch', type=str, choices=['TemporalNet', 'ModelGroup_TemporalNet'], default='TemporalNet')


model_path = dict(
    TemporalNet='checkpoints/temporalnet.pth.tar',
    ModelGroup_TemporalNet='checkpoints/lead_groupings_temporalnet.pth.tar'
)


def create_model(args, baseline, finetune=False):
    print("=> Creating Model")
    if args.arch == "TemporalNet":
        model = ECG_TemporalNet1D(**parameters.TemporalParams_1D, classification=True)
    else:
        lead_groups = [
            [0,1,6,7],
            [2,3,4,5]
        ]
        model = ModelGroup(arch=args.arch, parameters=parameters.TemporalParams_1D, lead_groups=lead_groups, classification=True)

    if baseline:
        print(f"Returning Baseline Model")
        return model

    checkpoint = torch.load(model_path[args.arch], map_location="cpu")
    state_dict = checkpoint['state_dict']

    if args.arch == "TemporalNet":
        for k in list(state_dict.keys()):
            if k.startswith("module.") and not k.startswith("module.finalLayer."):
                state_dict[k[len("module."):]] = state_dict[k]
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        missing_keys = {'finalLayer.0.weight', 'finalLayer.0.bias'}
        assert set(msg.missing_keys) == missing_keys
        
        for name, param in model.named_parameters():
            if not name.startswith("finalLayer"):
                param.requires_grad = finetune
    else:
        model_g1_state_dict = state_dict['model_g1']
        model_g2_state_dict = state_dict['model_g2']

        msg = model.model_g1.load_state_dict(model_g1_state_dict, strict=False)
        if (len(msg.missing_keys) != 0): print(f"There are {len(msg.missing_keys)} missing keys")
        
        for name, param in model.model_g1.named_parameters():
            if not name.startswith("finalLayer"):
                param.requires_grad = finetune

        msg = model.model_g2.load_state_dict(model_g2_state_dict, strict=False)
        if (len(msg.missing_keys) != 0): print(f"There are {len(msg.missing_keys)} missing keys")

        
        for name, param in model.model_g2.named_parameters():
            if not name.startswith("finalLayer"):
                param.requires_grad = finetune
            
        print(f"Pre-Trained Model Loaded from {model_path[args.arch]}")

    return model


def main():
    
    args = parser.parse_args()
    
    ecg = torch.randn(16,  8, 2500)  # Example input tensor
    model = create_model(args, baseline=False, finetune=True)
    model.eval()

    with torch.no_grad():
        output = model(ecg)
        print(f"Model Output: {output}")

if __name__ == "__main__":
    main()