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
Count remaining (non-zero) weights in the encoder (i.e. the transformer layers).
Sparsity and remaining weights levels are equivalent: sparsity % = 100 - remaining weights %.
"""
import argparse
import os

import torch
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


def main(args):
    serialization_dir = args.serialization_dir
    pruning_method = args.pruning_method
    threshold = args.threshold

    st = torch.load(os.path.join(serialization_dir, "pytorch_model.bin"), map_location="cpu")

    remaining_count = 0  # Number of remaining (not pruned) params in the encoder
    encoder_count = 0  # Number of params in the encoder

    print("name".ljust(60, " "), "Remaining Weights %", "Remaning Weight")
    for name, param in st.items():
        if "encoder" not in name:
            continue

        if "mask_scores" in name:
            if pruning_method == "topK":
                mask_ones = TopKBinarizer.apply(param, threshold).sum().item()
            elif pruning_method == "sigmoied_threshold":
                mask_ones = ThresholdBinarizer.apply(param, threshold, True).sum().item()
            elif pruning_method == "l0":
                l, r = -0.1, 1.1
                s = torch.sigmoid(param)
                s_bar = s * (r - l) + l
                mask = s_bar.clamp(min=0.0, max=1.0)
                mask_ones = (mask > 0.0).sum().item()
            else:
                raise ValueError("Unknown pruning method")
            remaining_count += mask_ones
            print(name.ljust(60, " "), str(round(100 * mask_ones / param.numel(), 3)).ljust(20, " "), str(mask_ones))
        else:
            encoder_count += param.numel()
            if "bias" in name or "LayerNorm" in name:
                remaining_count += param.numel()

    print("")
    print("Remaining Weights (global) %: ", 100 * remaining_count / encoder_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pruning_method",
        choices=["l0", "topK", "sigmoied_threshold"],
        type=str,
        required=True,
        help="Pruning Method (l0 = L0 regularization, topK = Movement pruning, sigmoied_threshold = Soft movement pruning)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=False,
        help="For `topK`, it is the level of remaining weights (in %) in the fine-pruned model."
        "For `sigmoied_threshold`, it is the threshold \tau against which the (sigmoied) scores are compared."
        "Not needed for `l0`",
    )
    parser.add_argument(
        "--serialization_dir",
        type=str,
        required=True,
        help="Folder containing the model that was previously fine-pruned",
    )

    args = parser.parse_args()

    main(args)
