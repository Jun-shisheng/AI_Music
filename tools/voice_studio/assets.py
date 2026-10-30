"""Public asset lock. Sizes and LFS SHA-256 came from the pinned official HF metadata."""
SOURCE_REPOSITORY="https://github.com/Soul-AILab/SoulX-Singer.git"
SOURCE_REVISION="81aeb3ae772c70093c3de74dc23c92d983801ae4"
ASSETS={
    "Soul-AILab/SoulX-Singer":dict(revision="40493ad90286056c7a9095035164434a79daa8c9",directory="SoulX-Singer",files={
        "model-svc.pt":dict(size=2793965154,sha256="38e72d9e15be6eef3cc937daaf81421a7b5b546816786530bb9aa346cee5d43c"),
        "model.pt":dict(size=2818092278,sha256="447eaf41f91a6b6659d55e9ec3c9b809221724fb8592aebaec35a23751a5b500"),
        "config.yaml":dict(size=579,git_blob_sha1="b2964b328783451e6239e78e1e2558fcfe404ffb")}),
    "openai/whisper-base":dict(revision="e37978b90ca9030d5170a5c07aadb050351a65bb",directory="whisper-base",files={
        "model.safetensors":dict(size=290403936,sha256="07cadb9f25677c8d50df603e66a98fbd842cce45047139baeb16e6219a1e807b"),
        "config.json":dict(size=1983,git_blob_sha1="708cca09e7e8a9364e444efe8b4f9c2691fc5f51"),
        "preprocessor_config.json":dict(size=184990,git_blob_sha1="c2048dfa9fd94a052e62e908d2c4dfb18534b4d2")}),
}
