# Optional: build a single image containing the webpage and the trained model.
# The default Compose launcher uses the bundled runtime archive and mounted files.
FROM ghcr.io/ggml-org/llama.cpp:server-b11429@sha256:2f8ebc2dde83c4bd7e5c6d61efa57965e0d5a79d29802c246c4593740d9c04c1
COPY web /web
COPY models /models
COPY licenses /licenses
ENTRYPOINT ["/app/llama-server"]
CMD ["--model", "/models/Llama-GuidelineKG-1B-Q8_0.gguf", "--alias", "guidelinekg", "--host", "0.0.0.0", "--port", "8080", "--path", "/web", "--ctx-size", "4096", "--parallel", "1", "--threads", "4", "--threads-batch", "4", "--n-gpu-layers", "0", "--predict", "512", "--temp", "0", "--repeat-penalty", "1.3", "--jinja"]
HEALTHCHECK --interval=5s --timeout=3s --start-period=30s --retries=12 CMD curl --fail --silent http://127.0.0.1:8080/health || exit 1
