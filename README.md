# ComfyUI Kirox Nano Banana Pro Node

This custom node sends a ComfyUI image plus an auto-generated prompt to Kirox Nano Banana Pro, waits for completion, downloads the result, and returns it to ComfyUI as an IMAGE.

## Install

Copy this folder into:

```text
ComfyUI/custom_nodes/ComfyUI-Kirox-NanoBanana
```

Then restart ComfyUI.

## Required Keys

Kirox needs a public image URL. For fully automatic "drop image and run" usage, this node uploads the reference image to a GitHub repo first.

Put these in a `.env` file in the ComfyUI root folder, or fill the node fields manually:

```text
KIROX_API_KEY=sk-your-kirox-key
GITHUB_TOKEN=github_pat_or_classic_token_with_repo_contents_write
```

The node defaults to this repo:

```text
tianjiaxing670-del/kirox-images
```

## Workflow Wiring

Use:

- image: resized reference image
- prompt: auto-generated prompt text
- output image: connect to SaveImage

Face policy:

- preserve: keep the same face visibility as the source
- hidden: source has no visible face, so output must not reveal a face
- visible: source face is visible, so output can show it

