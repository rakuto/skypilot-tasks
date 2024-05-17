import os
import boto3
from transformers import AutoModel, AutoModelForCausalLM, GPTQConfig, AutoTokenizer

s3_client = boto3.client("s3")


def upload_to_s3(dir: str, output: str) -> None:
    for root, dirs, files in os.walk(dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            rel_path = os.path.relpath(local_path, dir)
            s3_path = os.path.join(output, rel_path)

            print('Uploading {} to {}'.format(local_path, s3_path))
            s3_client.upload_file(local_path, s3_path)


def quantize() -> None:
    model_id = os.getenv("MODEL_ID")
    gptq_bits = int(os.getenv("GPTQ_BITS", "4"))
    gptq_dataset = os.getenv("GPTQ_DATASET", "c4")
    trust_remote_code = os.getenv("TRUST_REMOTE_CODE", "False").lower() == "true"
    s3_output_url = os.getenv("S3_UPLOAD_URL", None)
    hf_hub_repo_id = os.getenv("HF_REPO_ID", None)

    print(f"""\
Quantization settings:
  algorithm: GPTQ
  model: {model_id}
  bits: {gptq_bits}  
  dataset: {gptq_dataset}
  s3_output: {s3_output_url}
  hf_hub_repo_id: {hf_hub_repo_id}
""")

    if model_id.startswith("s3://"):
        model_name = "./" + model_id.split("/")[-1]
        command = "aws sync {} {}".format(model_id, model_name)
        print(command)
        os.system(command)
        model_id = model_name

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    gptq_config = GPTQConfig(
        bits=int(gptq_bits),
        dataset=gptq_dataset,
        tokenizer=tokenizer,
    )

    print("Loading quantized model")
    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='auto',
        quantization_config=gptq_config,
        trust_remote_code=trust_remote_code,
    )

    out_dir = model_id.split("/")[-1]
    quantized_model.to("cpu")
    quantized_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    if hf_hub_repo_id:
        print("Upload to Hugging Facce Hub {}".format(hf_hub_repo_id))
        quantized_model.push_to_hub(hf_hub_repo_id)
        tokenizer.push_to_hub(hf_hub_repo_id)

    if s3_output_url:
        print("Upload to S3 {}".format(s3_output_url))
        upload_to_s3(out_dir, s3_output_url)


if __name__ == '__main__':
    quantize()
