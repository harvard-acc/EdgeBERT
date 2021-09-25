# Copyright 2020-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Once a model has been fine-pruned, the weights that are masked during the forward pass can be pruned once for all.
For instance, once the a model from the :class:`~emmental.MaskedBertForSequenceClassification` is trained, it can be saved (and then loaded)
as a standard :class:`~transformers.BertForSequenceClassification`.
"""

import argparse
import os
import shutil

import torch

# from .binarizer import MagnitudeBinarizer, ThresholdBinarizer, TopKBinarizer
#
"""
Binarizers take a (real value) matrice as input and produce a binary (values in {0,1}) mask of the same shape.
"""

from torch import autograd

class ThresholdBinarizer(autograd.Function):
    """
    Thresholdd binarizer.
    Computes a binary mask M from a real value matrix S such that `M_{i,j} = 1` if and only if `S_{i,j} > \tau`
    where `\tau` is a real value threshold.

    Implementation is inspired from:
        https://github.com/arunmallya/piggyback
        Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights
        Arun Mallya, Dillon Davis, Svetlana Lazebnik
    """

    @staticmethod
    def forward(ctx, inputs: torch.tensor, threshold: float, sigmoid: bool):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input matrix from which the binarizer computes the binary mask.
            threshold (`float`)
                The threshold value (in R).
            sigmoid (`bool`)
                If set to ``True``, we apply the sigmoid function to the `inputs` matrix before comparing to `threshold`.
                In this case, `threshold` should be a value between 0 and 1.
        Returns:
            mask (`torch.FloatTensor`)
                Binary matrix of the same size as `inputs` acting as a mask (1 - the associated weight is
                retained, 0 - the associated weight is pruned).
        """
        nb_elems = inputs.numel()
        nb_min = int(0.005 * nb_elems) + 1
        if sigmoid:
            mask = (torch.sigmoid(inputs) > threshold).type(inputs.type())
        else:
            mask = (inputs > threshold).type(inputs.type())
        if mask.sum() < nb_min:
            # We limit the pruning so that at least 0.5% (half a percent) of the weights are remaining
            k_threshold = inputs.flatten().kthvalue(max(nb_elems - nb_min, 1)).values
            mask = (inputs > k_threshold).type(inputs.type())
        return mask

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None, None


class TopKBinarizer(autograd.Function):
    """
    Top-k Binarizer.
    Computes a binary mask M from a real value matrix S such that `M_{i,j} = 1` if and only if `S_{i,j}`
    is among the k% highest values of S.

    Implementation is inspired from:
        https://github.com/allenai/hidden-networks
        What's hidden in a randomly weighted neural network?
        Vivek Ramanujan*, Mitchell Wortsman*, Aniruddha Kembhavi, Ali Farhadi, Mohammad Rastegari
    """

    @staticmethod
    def forward(ctx, inputs: torch.tensor, threshold: float):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input matrix from which the binarizer computes the binary mask.
            threshold (`float`)
                The percentage of weights to keep (the rest is pruned).
                `threshold` is a float between 0 and 1.
        Returns:
            mask (`torch.FloatTensor`)
                Binary matrix of the same size as `inputs` acting as a mask (1 - the associated weight is
                retained, 0 - the associated weight is pruned).
        """
        # Get the subnetwork by sorting the inputs and using the top threshold %
        mask = inputs.clone()
        _, idx = inputs.flatten().sort(descending=True)
        j = int(threshold * inputs.numel())

        # flat_out and mask access the same memory.
        flat_out = mask.flatten()
        flat_out[idx[j:]] = 0
        flat_out[idx[:j]] = 1
        return mask

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None


class MagnitudeBinarizer(object):
    """
    Magnitude Binarizer.
    Computes a binary mask M from a real value matrix S such that `M_{i,j} = 1` if and only if `S_{i,j}`
    is among the k% highest values of |S| (absolute value).

    Implementation is inspired from https://github.com/NervanaSystems/distiller/blob/2291fdcc2ea642a98d4e20629acb5a9e2e04b4e6/distiller/pruning/automated_gradual_pruner.py#L24
    """

    @staticmethod
    def apply(inputs: torch.tensor, threshold: float):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input matrix from which the binarizer computes the binary mask.
                This input marix is typically the weight matrix.
            threshold (`float`)
                The percentage of weights to keep (the rest is pruned).
                `threshold` is a float between 0 and 1.
        Returns:
            mask (`torch.FloatTensor`)
                Binary matrix of the same size as `inputs` acting as a mask (1 - the associated weight is
                retained, 0 - the associated weight is pruned).
        """
        # Get the subnetwork by sorting the inputs and using the top threshold %
        mask = inputs.clone()
        _, idx = inputs.abs().flatten().sort(descending=True)
        j = int(threshold * inputs.numel())

        # flat_out and mask access the same memory.
        flat_out = mask.flatten()
        flat_out[idx[j:]] = 0
        flat_out[idx[:j]] = 1
        return mask


def main(args):
    pruning_method = args.pruning_method
    threshold = args.threshold

    model_name_or_path = args.model_name_or_path.rstrip("/")
    target_model_path = args.target_model_path

    print(f"Load fine-pruned model from {model_name_or_path}")
    model = torch.load(os.path.join(model_name_or_path, "pytorch_model.bin"))
    pruned_model = {}
    
    for name, tensor in model.items():
        print(name)    
 
    for name, tensor in model.items():
        #temporary fix
        if "albert.encoder.highway.0.pooler.dense.weight" in name:
            pruned_model["albert.encoder.highway.0.pooler.weight"] = tensor
            print(f"Copied layer {name}")
        elif "albert.encoder.highway.0.pooler.dense.bias" in name:
            pruned_model["albert.encoder.highway.0.pooler.bias"] = tensor
            print(f"Copied layer {name}")
        #actual 
        elif "embedding" in name or "LayerNorm" in name or "layer_norm" in name or "pooler" in name:
            pruned_model[name] = tensor
            print(f"Copied layer {name}")
        elif "highway" in name: #highway params
            pruned_model[name] = tensor
            print(f"Copied layer {name}")
        elif "classifier" in name or "qa_output" in name:
            pruned_model[name] = tensor
            print(f"Copied layer {name}")
        elif "bias" in name:
            pruned_model[name] = tensor
            print(f"Copied layer {name}")
        elif "adaptive_span" in name:
            pruned_model[name] = tensor
            print(f"Copied layer {name}")
        else:
            if pruning_method == "magnitude":
                mask = MagnitudeBinarizer.apply(inputs=tensor, threshold=threshold)
                pruned_model[name] = tensor * mask
                print(f"Pruned layer {name}")
            elif pruning_method == "topK":
                if "mask_scores" in name:
                    continue
                prefix_ = name[:-6]
                scores = model[f"{prefix_}mask_scores"]
                mask = TopKBinarizer.apply(scores, threshold)
                pruned_model[name] = tensor * mask
                print(f"Pruned layer {name}")
            elif pruning_method == "sigmoied_threshold":
                if "mask_scores" in name:
                    continue
                prefix_ = name[:-6]
                scores = model[f"{prefix_}mask_scores"]
                mask = ThresholdBinarizer.apply(scores, threshold, True)
                pruned_model[name] = tensor * mask
                print(f"Pruned layer {name}")
            elif pruning_method == "l0":
                if "mask_scores" in name:
                    continue
                prefix_ = name[:-6]
                scores = model[f"{prefix_}mask_scores"]
                l, r = -0.1, 1.1
                s = torch.sigmoid(scores)
                s_bar = s * (r - l) + l
                mask = s_bar.clamp(min=0.0, max=1.0)
                pruned_model[name] = tensor * mask
                print(f"Pruned layer {name}")
            else:
                raise ValueError("Unknown pruning method")

    if target_model_path is None:
        target_model_path = os.path.join(
            os.path.dirname(model_name_or_path), f"bertarized_{os.path.basename(model_name_or_path)}"
        )

    if not os.path.isdir(target_model_path):
        shutil.copytree(model_name_or_path, target_model_path)
        print(f"\nCreated folder {target_model_path}")

    torch.save(pruned_model, os.path.join(target_model_path, "pytorch_model.bin"))
    print("\nPruned model saved! See you later!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pruning_method",
        choices=["l0", "magnitude", "topK", "sigmoied_threshold"],
        type=str,
        required=True,
        help="Pruning Method (l0 = L0 regularization, magnitude = Magnitude pruning, topK = Movement pruning, sigmoied_threshold = Soft movement pruning)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=False,
        help="For `magnitude` and `topK`, it is the level of remaining weights (in %) in the fine-pruned model."
        "For `sigmoied_threshold`, it is the threshold \tau against which the (sigmoied) scores are compared."
        "Not needed for `l0`",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Folder containing the model that was previously fine-pruned",
    )
    parser.add_argument(
        "--target_model_path",
        default=None,
        type=str,
        required=False,
        help="Folder containing the model that was previously fine-pruned",
    )

    args = parser.parse_args()

    main(args)
