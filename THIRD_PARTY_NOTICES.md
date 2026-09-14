# Third-party notices

TinyLibrary uses nanoVLM and the BabyLM evaluation suite as external dependencies; their source code is not vendored here and remains under each project's own license.

`pipeline/llava_format/sanity_check.py` contains modified conversation-pair checking logic from TinyLLaVA Factory, commit `74883f79083ec9b14ab8df1d47e342ce1a3aa029`, originally located in `tinyllava/data/template/base.py`. That file is distributed under the Apache License 2.0 rather than the repository's default MIT license. A copy is provided in [`licenses/TinyLLaVA-Apache-2.0.txt`](licenses/TinyLLaVA-Apache-2.0.txt).
