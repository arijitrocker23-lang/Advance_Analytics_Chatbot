# =============================================================================
# data_loader.py
# =============================================================================
# Responsible for loading CSV data from local file uploads and from Amazon S3.
# Returns a pandas DataFrame along with metadata about the loaded file.
# =============================================================================

import io
from typing import Tuple
from urllib.parse import urlparse

import boto3
import pandas as pd

from config import Config
from utils import detect_date_columns


class S3Helper:
    """
    Lightweight helper for downloading CSV files from Amazon S3.
    """

    def __init__(self, region: str = Config.AWS_REGION):
        self._client = boto3.client("s3", region_name=region)

    @staticmethod
    def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
        """
        Split an s3://bucket/key URI into its bucket and key components.
        Raises ValueError if the URI is malformed.
        """
        parsed = urlparse(s3_uri)
        if parsed.scheme != "s3":
            raise ValueError("S3 URI must start with 's3://'")
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not bucket or not key:
            raise ValueError("Invalid S3 URI format; expected s3://bucket/key")
        return bucket, key

    def get_metadata(self, s3_uri: str) -> dict:
        """Return size and last-modified metadata for the S3 object."""
        bucket, key = self.parse_s3_uri(s3_uri)
        resp = self._client.head_object(Bucket=bucket, Key=key)
        return {
            "bucket": bucket,
            "key": key,
            "size_bytes": resp["ContentLength"],
            "size_mb": resp["ContentLength"] / (1024 * 1024),
            "last_modified": resp["LastModified"].isoformat(),
        }

    def download(self, s3_uri: str) -> bytes:
        """Download the S3 object and return its raw bytes."""
        bucket, key = self.parse_s3_uri(s3_uri)
        resp = self._client.get_object(Bucket=bucket, Key=key)
        return resp["Body"].read()


def load_csv_from_bytes(
    content: bytes,
    file_name: str,
) -> Tuple[bool, str, pd.DataFrame]:
    """
    Attempt to parse *content* as a CSV using several common encodings.

    Returns
    -------
    (success, message, dataframe)
        On failure the DataFrame will be empty.
    """
    buffer = io.BytesIO(content)
    for encoding in ("utf-8", "latin-1", "iso-8859-1", "cp1252"):
        try:
            buffer.seek(0)
            df = pd.read_csv(buffer, encoding=encoding, low_memory=False)

            # Auto-convert detected date columns to datetime dtype
            for col in detect_date_columns(df):
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass

            msg = (
                f"Loaded {file_name}: "
                f"{len(df):,} rows, {len(df.columns)} columns"
            )
            return True, msg, df
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            return False, f"Error loading CSV: {exc}", pd.DataFrame()

    return False, "Could not decode CSV with common encodings.", pd.DataFrame()