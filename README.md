<div align="center">
<h1>
ToMAP: Training Opponent-Aware<br>
LLM Persuaders with Theory of Mind
</h1>
</div>

<div align="center">
<h3>
Peixuan Han, Zijia Liu, Jiaxuan You
</h3>
</div>


<p align="center">
📃<a href="https://arxiv.org/pdf/2505.22961" target="_blank">Paper</a> • 🤗<a href="https://huggingface.co/HakHan/Qwen2.5-3B-Instruct-ToMAP" target="_blank">Model</a>
</p>


# About

![](figures/main_fig.png)

Theory of Mind Augmented Persuader (**ToMAP**) is a novel persuader training schema that incorporates theory of mind information, enabling the model to analyse the opponent's current thoughts, and develop more effective, targeted persuasion strategy. ToMAP enables language models of 3B size to obtain impressive persuasion capability, outperforming much larger LLMs.


# Repo Structure 

### Persuasion Setup
Refer to `verl/env_feedback/argument_graph.py`.

### RL Workflow
Refer to `verl/trainer/main_ppo.py` and `verl/trainer/ppo/ray_trainer.py`.
The original single-turn rollout is replaced by the multi-turn rollout in `verl/llm_agent/generation.py`.
The implementation is relatively inefficient and may benefit from optimization. Suggestions for improvement are welcome.

### Reward Design
Refer to `verl/utils/rewards.py` and `verl/trainer/main_ppo.py RewardManager`.

### Hparams
Refer to `verl/trainer/config/ppo_trainer.yaml`.

Specifically, you should always set `trainer.is_debate=True` when running persuasion tasks.

# Preperation

Steps marked with **\*** are required. Other steps involve preprocessing already completed by us, and are only necessary if reproduction from scratch is desired.

### Install Dependencies*

+ `python=3.9` and `vllm==0.6.3` are required for this repository.
+ It is recommended to use the pip package manager. Run the following commands to install all requirements:
```
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e . # verl
```
+ In addition, **ensure that system variables are configured according to your environment prior to using any of the bash scripts below, which are marked with "###"**.

### Load the Persuadee*

We use vllm to deploy the persuadee (by default Qwen2.5-7B-Instruct): `scripts/load_server.sh`.

For the attitude predictor, a BGE-M3 encoder should also be deployed (it is lightweight and requires minimal GPU memory): `scripts/load_encoder_server.sh`. This requires **vllm >= 0.8.4**, so a separate environment may be necessary for deployment.

**Ensure that the API server is running when conducting experiments.** Failure to do so may result in generic error messages from Ray, such as `RuntimeError: Failed to unpickle serialized exception`.

You may configure the port number as needed. The defaults are: 1279 for QWen-7B, 1568 for LLaMa-8B, 2184 for Phi-4, and 1450 for BGE-M3.

We support `external_persuadee`, but the interface is currently not user-friendly.

### Prepare Data
First, prepare a list of topics named `statements.json`, formatted as:
```
{
    "Topic 1",
    "Topic 2",
    ...
}
```
Use the following scripts to generate claims for both sides in the debate.

```
python data_gen/process_debate_datasets.py --base_dir [BASE_DIR]
python data_gen/debate.py --base_dir [BASE_DIR]
```
Preprocessed data is also provided in the `data` directory. Key files are `[dataset]/[train/test].[parquet/jsonl]`.

### Obtain Counterclaims
The training process does not appear to impact the persuader’s prediction of counterclaims. Consequently, all counterclaims have been preprocessed for efficiency.

The preprocessed counterclaims are available in the `data` directory. Ten counterclaims are collected per topic, although only three are used during training and evaluation. Key files are `[dataset]/[train/test]_argument_tree.pkl`.

### Obtain Initial Attitudes

Initial attitudes of the persuadees are collected for efficiency purposes. To regenerate them, run `scripts/build_tree.sh`.

This step may also be omitted, as the training/evaluation script will automatically perform it if required.

Attitudes for the three persuadees used in the main experiment are available in the `data` directory. Key files include `[dataset]/[persuadee]/[train/test].pkl` (trees with confidence values) and `[dataset]/[persuadee]/[train/test]_initial_attitude.json` (a human-readable version).

### Train the Attitude Predictor
+ Use `scripts/train_predictor.sh` to train the attitude predictor.

+ The checkpoint will be released at the time of publication.

# Persuader Training

Please refer to `scripts/train.sh`.

In particular, `tom_style` and `max_width` are important hyperparameters influencing the **theory of mind setting**:
+ For ToMAP, set `tom_style=black_external` and `max_width=3`, which indicates that 3 counterclaims are generated and an external attitude predictor is employed to assess the persuadee's attitude.

+ For the base model, set `tom_style=black_skip` and `max_width=0`. 

+ `tom_style=black_skip` and `max_width=3` constitutes the ablation setting "ToMAP (w/o att)", where 3 counterclaims are generated but no attitude prediction is provided.

+ Other `tom_style` values are available for ablation studies. Notably, `tom_style=white` refers to using the persuadee's actual attitude.

For further customization of hyperparameters, refer to `verl/trainer/config/ppo_trainer.yaml`.

### Training Plots

![](figures/training_plot.png)

## Evaluation
Please refer to `scripts/validate.sh`.

The script facilitates serialized evaluation across multiple tasks, persuadees, and persuaders.

**Due to the size of the CMV and args.me corpora, only 20% of the CMV validation data and 50% of the args.me validation data are used.** The statistics reported in the paper reflect this truncation.

Each validation result is saved in the following format:
```
"pos": "Pizza should contain pineapple.",
"neg": "Pizza should not contain pineapple.",
"turns": [
    "...(by Alice)",
    "...(by Bob)",
    ...
    ],
"thoughts": [
    "...(by Alice)",
    "...(by Bob)",
    ...
    ],
"reward": xxx
```

### Eval Results
![](figures/eval_results.png)

## Cite this paper
This repo is based on [TinyZero](https://github.com/Jiayi-Pan/TinyZero). We removed unrelated parts from the original repo.


If you find this repo or the paper useful, please cite:
```
@article{han2025tomap,
      title={ToMAP: Training Opponent-Aware LLM Persuaders with Theory of Mind}, 
      author={Peixuan Han and Zijia Liu and Jiaxuan You},
      year={2025},
      journal={arXiv preprint arXiv:2505.22961},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2505.22961}, 
}
```

Reach out to [Peixuan Han](mailto:ph16@illinois.edu) for any questions.
