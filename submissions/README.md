# Model submissions

Public results submitted for maintainer review. Private scores and standardized latency are measured by the maintainers.

| Model | Tracks | Public indices | Results | Engine / revision | Hardware | Settings and limits |
|---|---|---|---|---|---|---|
| [VisionJEV 1.0.0 (4B)](https://huggingface.co/tinnel123/VisionJEV-4B/tree/a5d1b24c42f546bdd8e05b34c28f46e86aff51e4) | **vision**, text (0.3) | Vision 73.03369467702923; Text 57.78 | [Complete runs](https://huggingface.co/datasets/tinnel123/VisionJEV-4B-results/tree/a57f283b0b98721395cd548498a508dd93241167/runs) | Native joint decision renderer; pinned source in [submission](visionjev-1.0.0-4b/README.md) | 8 × MLU590-M9DG (independent request workers) | BF16 computation, full 32 layers, T=1.5539092221462472; [exact configuration](visionjev-1.0.0-4b/README.md) |
