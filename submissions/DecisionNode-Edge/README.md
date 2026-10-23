# DecisionNode Edge and Edge Mini (vision)

[DecisionNode Edge](https://huggingface.co/bynn-intelligence-com/decisionnode-edge) and [DecisionNode Edge Mini](https://huggingface.co/bynn-intelligence-com/decisionnode-edge-mini) are open-weight decision models that answer on the `/v1/systemone` interface, for text and images. Each model repository ships its runtime (`runtime/`, the `decisionnode-edge` Python package) and its server.

| | Edge | Edge Mini |
|---|---:|---:|
| Public vision index (0.3) | **59.13** | **49.10** |
| Rows | 12,210, complete, 0 missing | 12,210, complete, 0 missing |
| Request time, one request at a time | median 62.5 ms · mean 104.1 ms · p80 141.6 ms | median 51.2 ms · mean 90.7 ms · p80 119.0 ms |
| Results | [runs/decisionnode-edge-vision](https://huggingface.co/datasets/bynn-intelligence-com/decisionnode-edge-decision-index/tree/69f0d176da6ac542d3ac673afccbe4d72339af53/runs/decisionnode-edge-vision) (compact: no suite text) | [runs/decisionnode-edge-mini-vision](https://huggingface.co/datasets/bynn-intelligence-com/decisionnode-edge-decision-index/tree/69f0d176da6ac542d3ac673afccbe4d72339af53/runs/decisionnode-edge-mini-vision) (compact) |

Hardware: one NVIDIA RTX PRO 6000 Blackwell (96 GB), one server process.

```sh
hf download bynn-intelligence-com/decisionnode-edge --revision 7706b508ecc443dd6d7e97440406349c90e826fa --local-dir decisionnode-edge
pip install "./decisionnode-edge/runtime[vllm,images]"
python -m decisionnode_edge serve decisionnode-edge --host 127.0.0.1 --port 8790 --api-key <key file>
DECISION_INDEX_API_KEY=<key> python -m decision_index vision run --dir work/vision --engine decide \
    --option base_url=http://127.0.0.1:8790 --option protocol=decisions --option path=/v1/systemone \
    --model decisionnode-edge --workers 1 --out runs/decisionnode-edge-vision
```

Edge Mini is the same with `bynn-intelligence-com/decisionnode-edge-mini` at `0e77dfdcfd7417a4e22d229d7ad6f8e34ac4ff13` and `--model decisionnode-edge-mini`.

Settings: the server's defaults, nothing set per benchmark; the kit's runner and scorer unmodified (`25c6ec7`). The suite was rebuilt from the pinned sources with Pillow 12.3.0 and Arial Bold, as `docs/vision.md` asks, and `vision verify` is ok. The results files are the kit's run files in the compact form (`payload` and `raw_output` left out, as `--compact` writes them); they rescore to the numbers above.
