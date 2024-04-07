# Gradio App for HF space
import random
from datetime import datetime, timedelta

# Input:
# Model choice

# Output:
# DF of predictions

import gradio as gr
import huggingface_hub
import pandas as pd
import torch.cuda
from transformers import LogitsProcessorList, MaxLengthCriteria, StoppingCriteriaList

import model.vocab as v
import model.model as mod
import huggingface_hub as hub
import yaml

from model.data import timestamp_space_calculation



@torch.no_grad()
def perform_action(input_df, option, prediction_rows):
    # Validate the METRIC_NAMEs
    for idx, value in input_df["METRIC_NAME"].items():
        if value not in mn_vals_set:
            err_string = f"METRIC_NAME in row {idx} must be in: " + "~".join(mn_vals_set)
            print(err_string.replace("~", "\n"))
            raise gr.Error(err_string.replace("~", " - "))

    input_df["PAT_ID"] = input_df["PAT_ID"].astype(int)

    if input_df["PAT_ID"].max() > pat_ct:
        raise gr.Error(f"Patient IDs must be lower than {pat_ct}")

    # Take the input DF and tokenize
    tokens: torch.Tensor = tk.encode(input_df)

    # Pad to length
    eos_index = len(tokens)
    n = al_model.config.n_positions
    input = torch.nn.functional.pad(
        input=tokens,
        pad=(0, n - len(tokens)),
        value=0,  # EOS token
    )
    input = input.unsqueeze(0).long()


    if option == options[0]:
        labels = torch.nn.functional.pad(
            input=torch.cat((tokens, torch.tensor([0]))),
            pad=(0, n - 1 - len(tokens)),
            value=-100
        )
        labels = labels.unsqueeze(0).long()
        output = al_model(input.to(device), labels=labels.to(device), return_dict=True)
        loss = output.loss.cpu().numpy()

        # Construct the loss df
        loss_rows = [
            {"METRIC_NAME": "-", "PAT_ID": "-", "ACCESS_TIME": "-"}
        ]
        for i in range(1, input_df.shape[0]):
            metric_loss = loss[0, i - 1]
            patient_loss = loss[1, i]
            time_loss = loss[2, i]
            row = {
                "METRIC_NAME": metric_loss, "PAT_ID": patient_loss, "ACCESS_TIME": time_loss
            }
            loss_rows.append(row)
        return pd.DataFrame(loss_rows)

    elif option == options[1]:
        sc = StoppingCriteriaList([
            MaxLengthCriteria(max_length=len(tokens) + len(vocab.field_names(include_special=False) * prediction_rows)),
        ])
        outputs = al_model.sample(
            input[:, :len(tokens)].to(device),
            stopping_criteria=sc,
            logits_processor=logits_processor
        )
        output_tokens = outputs[0, len(tokens):(len(tokens) + 3 * prediction_rows)].cpu().numpy()
        predictions: pd.DataFrame = tk.decode(output_tokens)
        return predictions

def gen_random_df(row_c):
    rows = []
    last_time = start_date
    for i in range(row_c):
        last_time = last_time + timedelta(seconds=random.choice(timestamp_spaces) + random.gauss(0, 10))
        next_row = {
            "METRIC_NAME": random.choice(mn_vals),
            "PAT_ID": random.randint(0, pat_ct),
            "ACCESS_TIME": last_time
        }
        rows.append(next_row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Load the vocab
    vocab_file = huggingface_hub.hf_hub_download("bcwarner/audit-icu-gpt2-25_3M", "vocab.pkl")
    vocab = v.EHRVocab(vocab_path=vocab_file)

    # Pull up our config and load the model
    config = yaml.safe_load(open("config_hf.yaml"))
    al_model = mod.EHRAuditGPT2.from_pretrained("bcwarner/audit-icu-gpt2-25_3M", vocab)
    al_model.loss.reduction = "none"
    al_model.generation_config.pad_token_id = al_model.generation_config.eos_token_id

    # Load the tokenizer
    timestamp_spaces = timestamp_space_calculation(list(config["timestamp_bins"].values()))
    tk = v.EHRAuditTokenizer(vocab, timestamp_spaces_cal=timestamp_spaces)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    options = ["Cross-Entropy", "Next Action w/ Sample Search"]
    start_date = datetime.now()
    mn_vals = list(vocab.field_tokens["METRIC_NAME"].keys())
    mn_vals_set = set(sorted(mn_vals))
    pat_ct = config["patient_id_max"]

    logits_processor = LogitsProcessorList(
        [
            v.EHRAuditLogitsProcessor(vocab=vocab),
        ]
    )

    demo = gr.Interface(
        perform_action,
        [
            gr.Dataframe(
                headers=["METRIC_NAME", "PAT_ID", "ACCESS_TIME"],
                datatype=["str", "number", "date"],
                row_count=5,
                value=gen_random_df(5),
                col_count=(3, "fixed"),
                interactive=True,
                label="Audit Log Input"
            ),
            gr.Dropdown(options, value=0, label="Action"),
            gr.Number(precision=0, label="Rows to Predict"),
        ],
        gr.Dataframe(
            label="Output"
        ),
        description="Demo of [`bcwarner/audit-icu-gpt2-25_3M`](https://huggingface.co/bcwarner/audit-icu-gpt2-25_3M) for Epic EHR audit log generation/cross-entropy performance." + \
                    " Notice: This demo is purely for research purposes only and does not constitute medical advice."
    )

    demo.launch()
