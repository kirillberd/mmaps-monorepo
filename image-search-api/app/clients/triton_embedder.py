from __future__ import annotations

import numpy as np

try:
    import tritonclient.http as triton_http
except Exception:  # noqa: BLE001
    triton_http = None  # type: ignore


class TritonClientNotInstalledError(RuntimeError):
    pass


class TritonInferenceError(RuntimeError):
    pass


class TritonEmbedder:
    """Synchronous Triton client for the "efficientnet" embedder model.

    Expected model signature (as in your model config):
    - input:  name=images,     dtype=FP32, shape=[1, 3, 224, 224]
    - output: name=embeddings, dtype=FP32, shape=[1, 1280]
    """

    def __init__(
        self,
        url: str,
        model_name: str = "efficientnet",
        input_name: str = "images",
        output_name: str = "embeddings",
        model_version: str | None = None,
        timeout_s: float = 10.0,
        ssl: bool = False,
    ) -> None:
        if triton_http is None:
            raise TritonClientNotInstalledError(
                "tritonclient is not installed. Add dependency: tritonclient[http]",
            )

        self.url = url
        self.model_name = model_name
        self.model_version = model_version
        self.input_name = input_name
        self.output_name = output_name
        self.timeout_s = float(timeout_s)
        self.ssl = bool(ssl)

        self._client = triton_http.InferenceServerClient(
            url=self.url, ssl=self.ssl, verbose=False
        )

    def embed(self, images: np.ndarray) -> np.ndarray:
        """Returns embeddings with shape [1, 1280]."""

        if images.dtype != np.float32:
            images = images.astype(np.float32)
        images = np.ascontiguousarray(images)

        try:
            inp = triton_http.InferInput(self.input_name, images.shape, "FP32")
            inp.set_data_from_numpy(images, binary_data=True)

            out = triton_http.InferRequestedOutput(self.output_name, binary_data=True)

            result = self._client.infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=[inp],
                outputs=[out],
            )
        except Exception as e:  # noqa: BLE001
            raise TritonInferenceError(f"Triton inference failed: {e}") from e

        embeddings = result.as_numpy(self.output_name)
        if embeddings is None:
            raise TritonInferenceError(
                f"Triton response does not contain output '{self.output_name}'",
            )
        return np.ascontiguousarray(embeddings, dtype=np.float32)
