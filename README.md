# SGLang-Omni

> **Boson AI fork.** This branch of sglang-omni is maintained by Boson AI to integrate Boson AI's multimodal models (e.g., Higgs TTS) into the SGLang-Omni pipeline framework. Upstream: [sgl-project/sglang-omni](https://github.com/sgl-project/sglang-omni).

SGLang-Omni is an ecosystem project for SGLang.
Omni models refer to models that have multi-modal inputs and multi-modal outputs.
These models typically consist of multiple stages, making SGLang's LLM-specific architecture no longer suitable.
Therefore, SGLang-Omni is designed to provide the ability to orchestrate multi-stage pipeline with high performance and real-time API support.
Our core features include:

- Native Integration with SGLang for performance
- Multi-Stage Pipeline Framework for Omni Models
- OpenAI-Compatible Server with Real-Time API support

## Quick Start

We will host the documentation upon open-sourcing. For now, you can find the documentation in the [docs](./docs) folder.

- [Get Started](./docs/get_started/installation.md)
- [Developer Reference](./docs/developer_reference/architecture.md)
- [Benchmarks](./docs/benchmarks/relay.md)
- [Examples](./examples/README.md)
