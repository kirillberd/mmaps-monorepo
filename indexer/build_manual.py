import os
import io
import json
import time
import hashlib
import mimetypes
import importlib
from dataclasses import dataclass
from typing import Iterator, Optional, List, Dict, Any, Callable, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# -------------------- utils --------------------
def _require(module_name: str, pip_name: Optional[str] = None) -> None:
    """
    Small helper to provide a clear error if optional deps are missing.
    """
    try:
        importlib.import_module(module_name)
    except ImportError as e:
        name = pip_name or module_name
        raise RuntimeError(
            f"Missing dependency '{module_name}'. Install it with:\n  pip install {name}"
        ) from e


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "y", "on")


# -------------------- configs --------------------
@dataclass(frozen=True)
class DatasetConfig:
    name: str = "open-images-v6"
    split: str = "train"
    max_samples: int = 100
    shuffle: bool = True
    seed: int = 42


@dataclass(frozen=True)
class MinioConfig:
    endpoint: str = "localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    secure: bool = False

    bucket: str = "images"
    object_prefix: str = "raw/"

    public_base_url: Optional[str] = "http://localhost:9000"

    make_bucket_public: bool = True
    skip_if_exists: bool = True


@dataclass(frozen=True)
class IndexingConfig:
    batch_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    parquet_path: str = "image_embeddings.parquet"

    store_embedding_as_binary: bool = True
    stable_uid_by_content_hash: bool = True


@dataclass(frozen=True)
class OpenSearchConfig:
    enabled: bool = True

    host: str = "localhost"
    port: int = 9200
    index: str = "image-embeddings"

    # HNSW params
    m: int = 16
    ef_construction: int = 200
    ef_search: int = 64

    # bulk ingest
    batch: int = 1000

    # client options
    timeout: int = 600
    http_compress: bool = True

    # For cosine-like behavior with innerproduct
    ensure_l2_normalized: bool = True


# -------------------- abstractions --------------------
class DatasetProvider:
    def iter_filepaths(self) -> Iterator[str]:
        raise NotImplementedError


class ObjectStore:
    def ensure_bucket(self) -> None:
        raise NotImplementedError

    def put_bytes(
        self, key: str, data: bytes, content_type: Optional[str] = None
    ) -> None:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def url_for(self, key: str) -> str:
        raise NotImplementedError


class Embedder:
    @property
    def embedding_dim(self) -> int:
        raise NotImplementedError

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def encode_batch(self, batch: torch.Tensor) -> np.ndarray:
        raise NotImplementedError


# -------------------- FiftyOne provider --------------------
class FiftyOneZooProvider(DatasetProvider):
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        _require("fiftyone", "fiftyone")
        import fiftyone.zoo as foz  # type: ignore

        self._dataset = foz.load_zoo_dataset(
            cfg.name,
            split=cfg.split,
            max_samples=cfg.max_samples,
            shuffle=cfg.shuffle,
            seed=cfg.seed,
            label_types=[],
            progress=True,
        )

    def iter_filepaths(self) -> Iterator[str]:
        for sample in self._dataset.iter_samples():
            yield sample.filepath


# -------------------- MinIO store --------------------
class MinioObjectStore(ObjectStore):
    def __init__(self, cfg: MinioConfig):
        self.cfg = cfg
        _require("minio", "minio")
        from minio import Minio  # type: ignore

        self.client = Minio(
            endpoint=cfg.endpoint,
            access_key=cfg.access_key,
            secret_key=cfg.secret_key,
            secure=cfg.secure,
        )

    def ensure_bucket(self) -> None:
        bucket = self.cfg.bucket
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)

        if self.cfg.make_bucket_public:
            policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"AWS": ["*"]},
                        "Action": ["s3:GetObject"],
                        "Resource": [f"arn:aws:s3:::{bucket}/*"],
                    }
                ],
            }
            try:
                self.client.set_bucket_policy(bucket, json.dumps(policy))
            except Exception as e:
                print(f"[WARN] Could not set public bucket policy automatically: {e}")

    def _full_key(self, key: str) -> str:
        prefix = self.cfg.object_prefix or ""
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        return f"{prefix}{key}"

    def exists(self, key: str) -> bool:
        try:
            self.client.stat_object(self.cfg.bucket, self._full_key(key))
            return True
        except Exception:
            return False

    def put_bytes(
        self, key: str, data: bytes, content_type: Optional[str] = None
    ) -> None:
        full_key = self._full_key(key)

        if self.cfg.skip_if_exists and self.exists(key):
            return

        bio = io.BytesIO(data)
        self.client.put_object(
            bucket_name=self.cfg.bucket,
            object_name=full_key,
            data=bio,
            length=len(data),
            content_type=content_type or "application/octet-stream",
        )

    def url_for(self, key: str) -> str:
        full_key = self._full_key(key)

        if self.cfg.public_base_url:
            base = self.cfg.public_base_url.rstrip("/")
            return f"{base}/{self.cfg.bucket}/{full_key}"

        return f"s3://{self.cfg.bucket}/{full_key}"


# -------------------- EfficientNet-B0 embedding model --------------------
class EfficientNetB0Embed(nn.Module):
    def __init__(
        self,
        weights: EfficientNet_B0_Weights = EfficientNet_B0_Weights.DEFAULT,
        l2_normalize: bool = True,
    ):
        super().__init__()
        base = efficientnet_b0(weights=weights)
        base.eval()

        self.features = base.features
        self.avgpool = base.avgpool
        self.l2_normalize = l2_normalize

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)  # [B, 1280, H', W']
        x = self.avgpool(x)  # [B, 1280, 1, 1]
        x = torch.flatten(x, 1)  # [B, 1280]

        if self.l2_normalize:
            eps = 1e-12
            norm = torch.sqrt(torch.clamp((x * x).sum(dim=1, keepdim=True), min=eps))
            x = x / norm

        return x


# -------------------- Embedder wrapper --------------------
class TorchModelEmbedder(Embedder):
    """
    Wrap ANY torch.nn.Module that maps (B,3,H,W) -> (B,D) or (B,D,1,1).
    """

    def __init__(
        self,
        model: nn.Module,
        device: str,
        preprocess: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        dummy_input_size: int = 224,
    ):
        self.device = device
        self.model = model.to(device).eval()

        if preprocess is None:
            self._pre = T.Compose(
                [
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self._pre = preprocess

        with torch.no_grad():
            x = torch.zeros(1, 3, dummy_input_size, dummy_input_size, device=device)
            y = self.model(x)
            y = self._postprocess_output(y)
            self._dim = int(y.shape[-1])

    @staticmethod
    def _postprocess_output(y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 4:
            y = y.flatten(1)
        return y

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self._pre(img)

    @torch.no_grad()
    def encode_batch(self, batch: torch.Tensor) -> np.ndarray:
        batch = batch.to(self.device, non_blocking=True)
        y = self.model(batch)
        y = self._postprocess_output(y)
        return y.detach().float().cpu().numpy()


def build_default_model(device: str) -> TorchModelEmbedder:
    """
    EfficientNet-B0 -> 1280-dim embeddings (+ L2-нормировка в модели)
    """
    weights = EfficientNet_B0_Weights.DEFAULT
    model = EfficientNetB0Embed(weights=weights, l2_normalize=True)
    preprocess = weights.transforms()

    return TorchModelEmbedder(
        model=model,
        device=device,
        preprocess=preprocess,
        dummy_input_size=224,
    )


# -------------------- UID + helpers --------------------
def compute_uid(image_bytes: bytes, stable: bool = True) -> str:
    if stable:
        return hashlib.sha1(image_bytes).hexdigest()
    import uuid

    return uuid.uuid4().hex


def uid_to_int64(uid: str) -> int:
    """
    Stable 63-bit positive integer id derived from uid (expected hex).
    Useful to keep OpenSearch _id/int benchmark-style compatibility.
    """
    try:
        x = int(uid[:16], 16)  # 64 bits
    except ValueError:
        x = int(hashlib.sha1(uid.encode("utf-8")).hexdigest()[:16], 16)
    return int(x & ((1 << 63) - 1))


def guess_content_type(filepath: str) -> str:
    ct, _ = mimetypes.guess_type(filepath)
    return ct or "application/octet-stream"


def object_key_from_uid(uid: str, filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ("", ".bin"):
        ext = ".jpg"
    return f"{uid}{ext}"


# -------------------- Parquet sink (batched) --------------------
class ParquetSink:
    def __init__(self, path: str, store_embedding_as_binary: bool, embedding_dim: int):
        self.path = path
        self.store_embedding_as_binary = store_embedding_as_binary
        self.embedding_dim = embedding_dim

        _require("pyarrow", "pyarrow")
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        self.pa = pa
        self.pq = pq

        self._writer = None
        self._schema = self._build_schema()

    def _build_schema(self):
        pa = self.pa
        fields = [
            pa.field("id", pa.string()),  # uid hex (sha1)
            pa.field("url", pa.string()),
        ]
        if self.store_embedding_as_binary:
            fields += [
                pa.field("embedding", pa.binary()),
                pa.field("embedding_dim", pa.int32()),
                pa.field("embedding_dtype", pa.string()),
            ]
        else:
            fields += [
                pa.field("embedding", pa.list_(pa.float32())),
            ]
        return pa.schema(fields)

    def _ensure_writer(self):
        if self._writer is None:
            self._writer = self.pq.ParquetWriter(
                where=self.path,
                schema=self._schema,
                compression="zstd",
                use_dictionary=True,
            )

    def write_batch(
        self, ids: List[str], urls: List[str], embeddings: np.ndarray
    ) -> None:
        assert len(ids) == len(urls) == embeddings.shape[0]

        pa = self.pa
        self._ensure_writer()

        cols: Dict[str, Any] = {
            "id": pa.array(ids, type=pa.string()),
            "url": pa.array(urls, type=pa.string()),
        }

        emb32 = np.asarray(embeddings, dtype=np.float32)

        if self.store_embedding_as_binary:
            blobs = [row.tobytes() for row in emb32]  # each row -> bytes
            cols["embedding"] = pa.array(blobs, type=pa.binary())
            cols["embedding_dim"] = pa.array(
                [emb32.shape[1]] * emb32.shape[0], type=pa.int32()
            )
            cols["embedding_dtype"] = pa.array(
                ["float32"] * emb32.shape[0], type=pa.string()
            )
        else:
            cols["embedding"] = pa.array(emb32.tolist(), type=pa.list_(pa.float32()))

        table = pa.table(cols, schema=self._schema)
        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


# -------------------- OpenSearch index builder --------------------
def _parquet_num_rows(parquet_path: str) -> int:
    _require("pyarrow", "pyarrow")
    import pyarrow.parquet as pq  # type: ignore

    pf = pq.ParquetFile(parquet_path)
    return int(pf.metadata.num_rows)


def _dtype_from_str(s: str):
    s = (s or "").lower()
    if s in ("float16", "fp16", "half"):
        return np.float16
    return np.float32


def iter_parquet_embedding_batches(
    parquet_path: str,
    batch_size: int,
    store_embedding_as_binary: bool,
    expected_dim: Optional[int] = None,
) -> Iterator[Tuple[List[str], List[str], np.ndarray]]:
    """
    Yields (ids, urls, embeddings[np.float32]) in batches.
    Supports both binary-embedding parquet and list<float> embedding parquet.
    """
    _require("pyarrow", "pyarrow")
    import pyarrow.dataset as ds  # type: ignore

    if store_embedding_as_binary:
        columns = ["id", "url", "embedding", "embedding_dim", "embedding_dtype"]
    else:
        columns = ["id", "url", "embedding"]

    dataset = ds.dataset(parquet_path, format="parquet")
    scanner = dataset.scanner(columns=columns, batch_size=batch_size)

    for rb in scanner.to_batches():
        ids = rb.column(rb.schema.get_field_index("id")).to_pylist()
        urls = rb.column(rb.schema.get_field_index("url")).to_pylist()

        if store_embedding_as_binary:
            blobs = rb.column(rb.schema.get_field_index("embedding")).to_pylist()

            dim_idx = rb.schema.get_field_index("embedding_dim")
            dtype_idx = rb.schema.get_field_index("embedding_dtype")

            dim = (
                int(rb.column(dim_idx)[0].as_py())
                if dim_idx != -1 and len(blobs)
                else 0
            )
            dtype_str = (
                rb.column(dtype_idx)[0].as_py()
                if dtype_idx != -1 and len(blobs)
                else "float32"
            )
            dtype = _dtype_from_str(dtype_str)

            if expected_dim is not None and dim != expected_dim and dim != 0:
                raise RuntimeError(
                    f"Parquet embedding_dim={dim} but expected_dim={expected_dim}. "
                    f"Check that parquet was produced by the same model/config."
                )

            if len(blobs) == 0:
                emb = np.zeros((0, expected_dim or dim), dtype=np.float32)
            else:
                used_dim = (
                    expected_dim or dim or (len(blobs[0]) // np.dtype(dtype).itemsize)
                )
                emb = np.empty((len(blobs), used_dim), dtype=np.float32)
                for i, blob in enumerate(blobs):
                    v = np.frombuffer(blob, dtype=dtype, count=used_dim)
                    if v.dtype != np.float32:
                        v = v.astype(np.float32, copy=False)
                    emb[i] = v
        else:
            emb_list = rb.column(rb.schema.get_field_index("embedding")).to_pylist()
            emb = np.asarray(emb_list, dtype=np.float32)

            if expected_dim is not None and emb.shape[1] != expected_dim:
                raise RuntimeError(
                    f"Parquet embedding dim={emb.shape[1]} but expected_dim={expected_dim}."
                )

        yield ids, urls, emb


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


class OpenSearchKnnIndexer:
    """
    Builds OpenSearch k-NN index (HNSW, FAISS engine, innerproduct space)
    and ingests vectors from parquet. Recreates index from scratch.
    """

    def __init__(self, cfg: OpenSearchConfig):
        self.cfg = cfg
        self.client = None

    def setup(self, dim: int) -> None:
        _require("opensearchpy", "opensearch-py")
        from opensearchpy import OpenSearch  # type: ignore

        self.client = OpenSearch(
            hosts=[{"host": self.cfg.host, "port": self.cfg.port}],
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False,
            http_compress=bool(self.cfg.http_compress),
            timeout=int(self.cfg.timeout),
        )

        # OpenSearch k-NN: innerproduct + L2-normalized vectors ~= cosine similarity
        body = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "vec": {
                        "type": "knn_vector",
                        "dimension": int(dim),
                        "method": {
                            "name": "hnsw",
                            "engine": "faiss",
                            "space_type": "innerproduct",
                            "parameters": {
                                "m": int(self.cfg.m),
                                "ef_construction": int(self.cfg.ef_construction),
                                "ef_search": int(self.cfg.ef_search),
                            },
                        },
                    },
                    "id": {"type": "long"},  # numeric id (derived from uid)
                    "uid": {"type": "keyword"},  # original sha1 hex
                    "url": {"type": "keyword"},
                }
            },
        }

        assert self.client is not None
        if self.client.indices.exists(index=self.cfg.index):
            self.client.indices.delete(index=self.cfg.index)
        self.client.indices.create(index=self.cfg.index, body=body)

    def ingest_parquet(
        self,
        parquet_path: str,
        store_embedding_as_binary: bool,
        embedding_dim: int,
    ) -> None:
        _require("opensearchpy", "opensearch-py")
        from opensearchpy import helpers  # type: ignore

        assert self.client is not None

        total = _parquet_num_rows(parquet_path)
        pbar = tqdm(total=total, desc="OpenSearch ingest", unit="vec")

        for ids, urls, xb in iter_parquet_embedding_batches(
            parquet_path=parquet_path,
            batch_size=int(self.cfg.batch),
            store_embedding_as_binary=store_embedding_as_binary,
            expected_dim=embedding_dim,
        ):
            if xb.shape[0] == 0:
                continue

            if self.cfg.ensure_l2_normalized:
                xb = l2_normalize_rows(xb)

            def gen_actions():
                for i in range(len(ids)):
                    uid = str(ids[i])
                    did = uid_to_int64(uid)
                    yield {
                        "_index": self.cfg.index,
                        "_id": int(did),
                        "_source": {
                            "id": int(did),
                            "uid": uid,
                            "url": str(urls[i]),
                            "vec": xb[i].tolist(),
                        },
                    }

            helpers.bulk(
                self.client,
                gen_actions(),
                chunk_size=int(self.cfg.batch),
                refresh=False,
                timeout=int(self.cfg.timeout),
            )

            pbar.update(len(ids))

        pbar.close()
        self.client.indices.refresh(index=self.cfg.index)

    def cleanup(self) -> None:
        if self.client is None:
            return
        try:
            self.client.indices.delete(index=self.cfg.index, ignore=[400, 404])
        except Exception:
            pass


# -------------------- main pipeline --------------------
def index_images(
    provider: DatasetProvider,
    store: ObjectStore,
    embedder: Embedder,
    cfg: IndexingConfig,
) -> None:
    store.ensure_bucket()

    sink = ParquetSink(
        path=cfg.parquet_path,
        store_embedding_as_binary=cfg.store_embedding_as_binary,
        embedding_dim=embedder.embedding_dim,
    )

    batch_tensors: List[torch.Tensor] = []
    batch_ids: List[str] = []
    batch_urls: List[str] = []

    def flush():
        nonlocal batch_tensors, batch_ids, batch_urls
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors, dim=0)
        emb = embedder.encode_batch(batch)  # (B,D) numpy
        sink.write_batch(batch_ids, batch_urls, emb)

        batch_tensors = []
        batch_ids = []
        batch_urls = []

    n_ok = 0
    n_fail = 0

    for fp in tqdm(provider.iter_filepaths(), desc="Indexing images", unit="img"):
        try:
            with open(fp, "rb") as f:
                b = f.read()

            uid = compute_uid(b, stable=cfg.stable_uid_by_content_hash)
            key = object_key_from_uid(uid, fp)

            store.put_bytes(key=key, data=b, content_type=guess_content_type(fp))
            url = store.url_for(key)

            img = Image.open(io.BytesIO(b))
            x = embedder.preprocess(img)

            batch_tensors.append(x)
            batch_ids.append(uid)
            batch_urls.append(url)

            if len(batch_tensors) >= cfg.batch_size:
                flush()

            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[WARN] Failed on {fp}: {e}")

    flush()
    sink.close()

    print(f"Done. OK={n_ok}, FAIL={n_fail}")
    print(f"Parquet written to: {cfg.parquet_path}")


def main():
    ds_cfg = DatasetConfig(
        name=os.getenv("DATASET_NAME", "open-images-v6"),
        split=os.getenv("DATASET_SPLIT", "train"),
        max_samples=int(os.getenv("DATASET_MAX_SAMPLES", "50000")),
        shuffle=_env_bool("DATASET_SHUFFLE", "true"),
        seed=int(os.getenv("DATASET_SEED", "42")),
    )

    minio_cfg = MinioConfig(
        endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", None),
        secure=_env_bool("MINIO_SECURE", "false"),
        bucket=os.getenv("MINIO_BUCKET", "images"),
        object_prefix=os.getenv("MINIO_PREFIX", "raw/"),
        public_base_url=os.getenv("MINIO_PUBLIC_BASE_URL", "http://localhost:9000"),
        make_bucket_public=_env_bool("MINIO_MAKE_PUBLIC", "true"),
        skip_if_exists=_env_bool("MINIO_SKIP_IF_EXISTS", "true"),
    )

    idx_cfg = IndexingConfig(
        batch_size=int(os.getenv("BATCH_SIZE", "64")),
        device=os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        parquet_path=os.getenv("PARQUET_PATH", "image_embeddings.parquet"),
        store_embedding_as_binary=_env_bool("EMB_BINARY", "false"),
        stable_uid_by_content_hash=_env_bool("UID_STABLE", "true"),
    )

    os_cfg = OpenSearchConfig(
        enabled=_env_bool("OPENSEARCH_ENABLE", "true"),
        host=os.getenv("OPENSEARCH_HOST", "localhost"),
        port=int(os.getenv("OPENSEARCH_PORT", "9200")),
        index=os.getenv("OPENSEARCH_INDEX", "image-embeddings"),
        m=int(os.getenv("OPENSEARCH_M", "16")),
        ef_construction=int(os.getenv("OPENSEARCH_EF_CONSTRUCTION", "200")),
        ef_search=int(os.getenv("OPENSEARCH_EF_SEARCH", "64")),
        batch=int(os.getenv("OPENSEARCH_BATCH", "1000")),
        timeout=int(os.getenv("OPENSEARCH_TIMEOUT", "600")),
        http_compress=_env_bool("OPENSEARCH_HTTP_COMPRESS", "true"),
        ensure_l2_normalized=_env_bool("OPENSEARCH_ENSURE_L2", "false"),
    )

    provider = FiftyOneZooProvider(ds_cfg)
    store = MinioObjectStore(minio_cfg)
    embedder = build_default_model(idx_cfg.device)

    t0 = time.time()
    index_images(provider=provider, store=store, embedder=embedder, cfg=idx_cfg)
    t1 = time.time()
    print(f"Embedding+upload stage time: {t1 - t0:.2f}s")

    # ---- NEW: read parquet + build OpenSearch index from scratch ----
    if os_cfg.enabled:
        print(
            f"Building OpenSearch index '{os_cfg.index}' at {os_cfg.host}:{os_cfg.port} "
            f"from parquet: {idx_cfg.parquet_path}"
        )
        idxr = OpenSearchKnnIndexer(os_cfg)
        idxr.setup(dim=embedder.embedding_dim)  # deletes existing + creates fresh index
        t2 = time.time()
        idxr.ingest_parquet(
            parquet_path=idx_cfg.parquet_path,
            store_embedding_as_binary=idx_cfg.store_embedding_as_binary,
            embedding_dim=embedder.embedding_dim,
        )
        t3 = time.time()
        print(f"OpenSearch ingest stage time: {t3 - t2:.2f}s")
        print("OpenSearch index is ready")


if __name__ == "__main__":
    main()
