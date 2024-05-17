# LLM Tasks Toolbox

Miscellaneous Skypilot tasks for Gen-AI workflow.

## Download Hugging Face model and upload to S3 bucket

This task downloads model from Hugging Face and upload to S3.

Create a env file like this. Set Hugging Face model name as `MODEL_ID`.

```shell
cat <<-EOF > download-llama-3-8b-instruct.env
MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct
S3_URL=<S3_URL>
HF_USERNAME=<HF_USERNAME>
HF_TOKEN=<HF_TOKEN>
AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
#AWS_SESSION_TOKEN=<AWS_SESSION_TOKEN>
EOF
```

Then start Skypilot task as follows.

```shell
sky launch --env-file download-llama-3-8b-instruct.env hf_model_downloader.yaml 
```

In case you reuse existing Skypilot cluster, give `-c` flag in `sky launch`.

```shell
sky launch -c <CLUSTER_NAME> 
```


## TensorRT-LLM ahead-of-time compliation

For `Mistral-7B-Instruct-v0.1`.

```shell
sky launch \
  --env S3_URL=s3://models.tne.ai/trtllm/Mistral-7B-Instruct-v0.1 \
  --env OPTION_MODEL_ID=s3://models.tne.ai/hf/mistralai/ \
  --env OPTION_DTYPE=bf16 \
  --env OPTION_TENSOR_PARALLE_DEGREE=1 \
  --env OPTION_MAX_NUM_TOKENS=50000 \
  --env OPTION_MAX_INPUT_LEN=4096 \
  --env OPTION_MAX_OUTPUT_LEN=4096 \
  --env OPTION_MAX_ROLLING_BATCH_SIZE=256 \
  trtllm_aot_g5_2xlarge.yaml
```
