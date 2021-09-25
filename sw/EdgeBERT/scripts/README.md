# Training and Evaluating EdgeBERT models

This example reproduces the results from the paper (in Table 3) on the SST-2 task.

1. Run download_glue.sh to clone https://github.com/nyu-mll/GLUE-baselines.git and download the required datasets.

2. Run 1_train_teacher_sst2.sh to train a teacher model.

3. Run 2_train_sst2.sh to train the full model.

4. Run 3_bertarize_sst2.sh to prune the full model.

5. Run 4_eval_sst2_ee.sh to evaluate the full model with early exit.

6. Copy the lookup table (csv file) produced by the entropypredictor.ipynb script into this directory.

7. Run 5_eval_sst2_ep.sh to evaluate the full model with entropy prediction.

8. (Optional) Run 6_eval_sst2_ep_predlayer.sh to identify the average predicted exit layer used by DVFS.
