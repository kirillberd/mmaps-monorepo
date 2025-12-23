from dependency_injector import containers, providers

from app.clients.triton_embedder import TritonEmbedder
from app.repositories.opensearch_repository import OpenSearchRepository
from app.services.image_search_service import ImageSearchService
from app.services.preprocessing import ImagePreprocessor


class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

    # Core building blocks
    preprocessor = providers.Singleton(
        ImagePreprocessor,
        image_size=config.preprocess.image_size,
        mean=config.preprocess.mean,
        std=config.preprocess.std,
    )

    embedder = providers.Singleton(
        TritonEmbedder,
        url=config.triton.url,
        model_name=config.triton.model_name,
        model_version=config.triton.model_version,
        input_name=config.triton.input_name,
        output_name=config.triton.output_name,
        timeout_s=config.triton.timeout_s,
        ssl=config.triton.ssl,
    )

    opensearch_repository = providers.Singleton(
        OpenSearchRepository,
        host=config.opensearch.host,
        port=config.opensearch.port,
        index=config.opensearch.index,
        username=config.opensearch.username,
        password=config.opensearch.password,
        use_ssl=config.opensearch.use_ssl,
        verify_certs=config.opensearch.verify_certs,
        http_compress=config.opensearch.http_compress,
        timeout_s=config.opensearch.timeout_s,
        request_timeout_s=config.opensearch.request_timeout_s,
        ef_search=config.opensearch.ef_search,
        source_includes=config.opensearch.source_includes,
    )

    # Orchestrator
    image_search_service = providers.Singleton(
        ImageSearchService,
        preprocessor=preprocessor,
        embedder=embedder,
        repository=opensearch_repository,
        l2_normalize_embeddings=config.search.l2_normalize,
    )
