#%%
import os
import sys
import subprocess
import re
from configparser import ConfigParser
from collections import OrderedDict
from pathlib import Path

import boto3
import botocore


def get_credentials():
    try:
        config = ConfigParser()
        config.read(os.getenv("HOME") + "/.aws/credentials")
        return (
            config.get("default", "aws_access_key_id"),
            config.get("default", "aws_secret_access_key"),
        )
    except:
        ACCESS = os.getenv("AWS_ACCESS_KEY_ID")
        SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not ACCESS and SECRET:
        raise AttributeError("No AWS credentials found.")
    return (ACCESS, SECRET)

def parse_path(s3_datapath):
    """ return bucket and prefix from full path. """
    bucket_path = str(s3_datapath).split('//')[1]
    parts = bucket_path.split('/')
    bucket = parts[0]
    prefix = '/'.join(parts[1:4])
    return bucket, prefix

def s3_client():
    ACCESS, SECRET = get_credentials()
    return boto3.client('s3', aws_access_key_id=ACCESS, aws_secret_access_key=SECRET)

def get_matching_s3_objects(bucket, prefix='', suffix=''):
    """
    Generate objects in an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch objects whose key starts with
        this prefix (optional).
    :param suffix: Only fetch objects whose keys end with
        this suffix (optional).
    """
    s3 = s3_client()
    kwargs = {'Bucket': bucket}

    # If the prefix is a single string (not a tuple of strings), we can
    # do the filtering directly in the S3 API.
    if isinstance(prefix, str):
        kwargs['Prefix'] = prefix

    while True:

        # The S3 API response is a large blob of metadata.
        # 'Contents' contains information about the listed objects.
        resp = s3.list_objects_v2(**kwargs)

        try:
            contents = resp['Contents']
        except KeyError:
            print("No contents found.")
            return

        for obj in contents:
            key = obj['Key']
            if key.startswith(prefix) and key.endswith(suffix):
                yield key

        # The S3 API is paginated, returning up to 1000 keys at a time.
        # Pass the continuation token into the next response, until we
        # reach the final page (when this field is missing).
        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break

def s3_download_graph(bucket, prefix, local):
    """
    given an s3 directory,
    copies in that directory to local.
    """
    parent = Path(local).parent
    if not parent.is_dir():
        parent.mkdir(parents=True, exist_ok=True)
    s3 = s3_client()
    s3.download_file(bucket, prefix, local)