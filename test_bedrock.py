# =============================================================================
# test_bedrock.py
# =============================================================================
# Quick test to verify Bedrock connectivity and model access.
# Run with: python test_bedrock.py
# =============================================================================

import os
import json
import traceback

# Set proxy (same as config.py)
os.environ["AWS_REGION"] = "us-east-1"
os.environ["http_proxy"] = (
    "vpce-04cc59f15d545ec85-b2aczhbv."
    "vpce-svc-0aef210793c6f5cd7.eu-west-1.vpce.amazonaws.com:3128"
)
os.environ["https_proxy"] = os.environ["http_proxy"]
os.environ["HTTP_proxy"] = os.environ["http_proxy"]
os.environ["HTTPS_proxy"] = os.environ["http_proxy"]

import boto3

# List of model IDs to test
MODELS_TO_TEST = [
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
]

REGION = "us-east-1"


def test_model(client, model_id):
    """Try to call a model and report success or failure."""
    print("\n--- Testing: {} ---".format(model_id))
    try:
        response = client.converse(
            modelId=model_id,
            system=[{"text": "You are a helpful assistant. Reply in one short sentence."}],
            messages=[
                {
                    "role": "user",
                    "content": [{"text": "Say hello"}],
                }
            ],
            inferenceConfig={
                "maxTokens": 50,
                "temperature": 0.0,
                "topP": 1.0,
            },
        )
        # Extract response text
        message = response["output"]["message"]
        parts = message.get("content", [])
        text = " ".join(p.get("text", "") for p in parts if "text" in p)
        print("  SUCCESS: {}".format(text.strip()))
        return True
    except Exception as exc:
        print("  FAILED: {} - {}".format(type(exc).__name__, str(exc)))
        return False


def main():
    print("=" * 60)
    print("BEDROCK CONNECTIVITY TEST")
    print("Region: {}".format(REGION))
    print("=" * 60)

    # Step 1: Create the client
    print("\nStep 1: Creating bedrock-runtime client...")
    try:
        client = boto3.client("bedrock-runtime", region_name=REGION)
        print("  Client created successfully.")
    except Exception as exc:
        print("  FAILED to create client: {}".format(exc))
        traceback.print_exc()
        return

    # Step 2: Try listing foundation models to verify basic connectivity
    print("\nStep 2: Testing basic connectivity...")
    try:
        bedrock_mgmt = boto3.client("bedrock", region_name=REGION)
        models = bedrock_mgmt.list_foundation_models(
            byProvider="Anthropic",
            byOutputModality="TEXT",
        )
        model_ids = [m["modelId"] for m in models.get("modelSummaries", [])]
        print("  Available Anthropic models:")
        for mid in model_ids:
            print("    - {}".format(mid))
    except Exception as exc:
        print("  Could not list models (this may be normal): {}".format(exc))

    # Step 3: Test each model ID
    print("\nStep 3: Testing model IDs with converse API...")
    working_models = []
    for model_id in MODELS_TO_TEST:
        success = test_model(client, model_id)
        if success:
            working_models.append(model_id)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if working_models:
        print("Working models:")
        for m in working_models:
            print("  - {}".format(m))
    else:
        print("NO MODELS WORKED.")
        print("Check:")
        print("  1. AWS credentials are configured")
        print("  2. IAM role has bedrock:InvokeModel permission")
        print("  3. Models are enabled in the Bedrock console")
        print("  4. Proxy/VPC endpoint is correct")
    print("=" * 60)


if __name__ == "__main__":
    main()