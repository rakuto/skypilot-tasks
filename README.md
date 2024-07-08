# Skypilot tasks for ML applications

Miscellaneous Skypilot tasks for ML applications.

## Download Hugging Face model on S3

The task to download a model from Hugging Face and then upload to S3 bucket.  
Create `.env` file with following variables, make sure you update variables for your needs.

```shell
cd download_model

cat <<-EOF >> .env 
MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct 
S3_URL=s3://model-repo/hf/meta-llama/Meta-Llama-3-8B-Instruct
HF_USERNAME=<YOUR_HF_NAME>
HF_TOKEN=<YOUR_HF_TOKEN>
EOF
```

Using `m6i.2xlarge` instance by default. Please update `resources.instrance_type` in case you need more beefy instance.

```shell
sky launch --env-file .env task.yaml 
```

In case you reuse existing Skypilot cluster, give `-c` flag in `sky launch`.

```shell
sky launch -c $CLUSTER_NAME task.yaml
```

## Quantizing model

A task quantizing a text-generation model. 

```shell
cat <<-EOF >> .env 
MODEL_ID=nvidia/Llama3-ChatQA-1.5-8B 
# or MODEL_ID=s3://model-repo/hf/Llama3-ChatQA-1.5-8B
S3_UPLOAD_URL=s3://model-repo/hf/Llama3-ChatQA-1.5-8B-GPTQ-4bit
HF_REPO_ID=<YOUR_HF_REPO_ID>
HF_TOKEN=<YOUR_HF_TOKEN>
EOF
```

Launch Managed Spot jobs that any spot preemptions are automatically 
handled by SkyPilot without user intervention.

```shell
sky jobs launch --env-file .env gptq.yaml
```

## TensorRT-LLM ahead-of-time compliation

For `Mistral-7B-Instruct-v0.1`.

```shell
sky launch \
  --env S3_URL=s3://model-repo/trtllm/Mistral-7B-Instruct-v0.1 \
  --env OPTION_MODEL_ID=s3://model-repo/hf/mistralai/Mistral-7B-Instruct-v0.1 \
  --env OPTION_DTYPE=bf16 \
  --env OPTION_TENSOR_PARALLE_DEGREE=1 \
  --env OPTION_MAX_NUM_TOKENS=50000 \
  --env OPTION_MAX_INPUT_LEN=4096 \
  --env OPTION_MAX_OUTPUT_LEN=4096 \
  --env OPTION_MAX_ROLLING_BATCH_SIZE=256 \
  trtllm_aot_g5_2xlarge.yaml
```
