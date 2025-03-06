import boto3
import json

prompt_data = """
Act as a Shakespeare and write a poem on generative AI
"""

bedrock = boto3.client(service_name="bedrock-runtime",region_name="us-east-1")

payload={
    "prompt":prompt_data,
    "max_gen_len": 512,
    "temperature": 0.5,
    "top_p": 0.9
}

body = json.dumps(payload)
model_id = "us.meta.llama3-3-70b-instruct-v1:0"
response = bedrock.invoke_model(body=body,modelId = model_id)

response_body = json.loads(response["body"].read())
response_text = response_body['generation']
print(response_text)