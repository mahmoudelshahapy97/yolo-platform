"""S3 / MinIO object storage (boto3 in a thread; uploads stream from a spooled file)."""

import asyncio
from typing import BinaryIO, Optional
from urllib.parse import urlsplit

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


class Storage:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str, region: str,
                 public_endpoint: Optional[str] = None, public_prefix: str = "") -> None:
        self.bucket = bucket
        self.public_prefix = public_prefix.rstrip("/")
        kwargs = dict(aws_access_key_id=access_key or None, aws_secret_access_key=secret_key or None,
                      region_name=region, config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                                                        retries={"max_attempts": 3}))
        self._s3 = boto3.client("s3", endpoint_url=endpoint or None, **kwargs)
        # Presigned links are signed for the host the client will use.
        self._signer = boto3.client("s3", endpoint_url=public_endpoint or endpoint or None, **kwargs)

    async def ensure_bucket(self) -> None:
        def _ensure():
            try:
                self._s3.head_bucket(Bucket=self.bucket)
            except ClientError:
                self._s3.create_bucket(Bucket=self.bucket)
        await asyncio.to_thread(_ensure)

    async def ready(self) -> bool:
        try:
            await asyncio.to_thread(self._s3.head_bucket, Bucket=self.bucket)
            return True
        except Exception:
            return False

    async def upload_fileobj(self, fileobj: BinaryIO, key: str, content_type: str) -> None:
        await asyncio.to_thread(self._s3.upload_fileobj, fileobj, self.bucket, key, ExtraArgs={"ContentType": content_type})

    async def presign(self, key: str, expires: int = 3600) -> str:
        url = await asyncio.to_thread(self._signer.generate_presigned_url, "get_object",
                                      Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires)
        if self.public_prefix:
            # Signed for the internal endpoint; the proxy forwards with that Host header.
            parts = urlsplit(url)
            return f"{self.public_prefix}{parts.path}?{parts.query}"
        return url
