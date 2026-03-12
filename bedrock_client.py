# =============================================================================
# bedrock_client.py
# =============================================================================
# Thin wrapper around the AWS Bedrock Runtime converse API.
# Handles the constraint that some models do not allow both temperature
# and top_p to be specified simultaneously.
# =============================================================================

import traceback

import boto3

from config import Config


class BedrockClient:
    """
    Manages a single boto3 bedrock-runtime client and exposes a simple
    call() method that wraps the converse API.
    """

    def __init__(self, region=Config.AWS_REGION):
        # Create the boto3 client once and reuse it
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def call(
        self,
        model_id,
        system_prompt,
        user_text,
        max_tokens=2048,
        temperature=0.0,
    ):
        """
        Send a single user message to the specified model and return
        the assistant's text response.

        IMPORTANT: Claude 4.x models do not allow both temperature and
        top_p to be specified at the same time. We only send temperature.
        """
        # Log the call for debugging (visible in the terminal)
        print("=" * 60)
        print("[BedrockClient] Calling model: {}".format(model_id))
        print("[BedrockClient] User text length: {} chars".format(len(user_text)))
        print("[BedrockClient] Max tokens: {}, Temperature: {}".format(
            max_tokens, temperature
        ))
        print("=" * 60)

        # Build inference config with ONLY temperature (not top_p)
        # Claude 4.x models reject requests that include both.
        inference_config = {
            "maxTokens": max_tokens,
            "temperature": temperature,
        }

        try:
            response = self._client.converse(
                modelId=model_id,
                system=[{"text": system_prompt}],
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": user_text}],
                    }
                ],
                inferenceConfig=inference_config,
            )

            # Extract text blocks from the response message
            message = response["output"]["message"]
            parts = message.get("content", [])
            text_parts = [
                part.get("text", "") for part in parts if "text" in part
            ]
            result = " ".join(text_parts).strip()

            # Log success
            print("[BedrockClient] SUCCESS - Response length: {} chars".format(
                len(result)
            ))
            print("[BedrockClient] First 200 chars: {}".format(result[:200]))
            print("=" * 60)

            return result

        except Exception as exc:
            # Log the full error for debugging
            print("[BedrockClient] ERROR calling model: {}".format(model_id))
            print("[BedrockClient] Error type: {}".format(type(exc).__name__))
            print("[BedrockClient] Error message: {}".format(str(exc)))
            traceback.print_exc()
            print("=" * 60)
            # Re-raise so the caller can handle it
            raise