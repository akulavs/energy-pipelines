"""
Generic filesystem I/O utilities.

Accepts any URI recognised by pyarrow.fs.FileSystem.from_uri():
  s3://bucket/key    → S3FileSystem
  /local/path        → LocalFileSystem
  file:///local/path → LocalFileSystem
"""

import pyarrow.fs


def write_bytes(uri: str, data: bytes) -> None:
    fs, path = pyarrow.fs.FileSystem.from_uri(uri)
    parent = path.rsplit("/", 1)[0]
    fs.create_dir(parent, recursive=True)
    with fs.open_output_stream(path) as f:
        f.write(data)


def read_bytes(uri: str) -> bytes:
    fs, path = pyarrow.fs.FileSystem.from_uri(uri)
    with fs.open_input_stream(path) as f:
        return f.read()
