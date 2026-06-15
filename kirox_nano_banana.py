import base64
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


KIROX_API_BASE = "https://api.kirox.ai/v1/images"
KIROX_MODEL = "nano-banana-pro"


FACE_POLICY_PROMPTS = {
    "preserve": (
        "Preserve the face visibility from the reference image. If the reference face is visible, keep the same visible face. "
        "If the reference face is hidden, covered, turned away, or not clearly visible, do not invent or reveal a face."
    ),
    "hidden": (
        "Important identity rule: the reference image does not show the person's face clearly. The generated image must not "
        "show or invent a visible face. Keep the face hidden by phone, hair, camera angle, back view, looking down, or turned away. "
        "No front-facing face, no new facial features, no eye contact with the camera."
    ),
    "visible": (
        "The reference image shows the person's face. Keep the same visible identity, facial structure, hairstyle, and realistic expression."
    ),
}


def load_env_file() -> None:
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
        Path.home() / ".kirox.env",
    ]
    for path in candidates:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def image_tensor_to_png_bytes(image: torch.Tensor) -> bytes:
    if image.ndim == 4:
        image = image[0]
    array = image.detach().cpu().numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(array)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def png_bytes_to_image_tensor(data: bytes) -> torch.Tensor:
    pil_image = Image.open(io.BytesIO(data)).convert("RGB")
    array = np.asarray(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]


def request_json(method: str, url: str, headers: dict, body: Optional[dict] = None, timeout: int = 120) -> dict:
    payload = None
    final_headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    }
    final_headers.update(headers)
    if body is not None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers=final_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def upload_png_to_github(
    png_bytes: bytes,
    github_token: str,
    github_repo: str,
    github_branch: str,
    github_path_prefix: str,
) -> str:
    if not github_token:
        raise RuntimeError("Missing github_token. Kirox needs a public image URL, so automatic upload requires a GitHub token.")
    if "/" not in github_repo:
        raise RuntimeError("github_repo must look like owner/repo, for example tianjiaxing670-del/kirox-images.")

    prefix = github_path_prefix.strip().strip("/") or "kirox-inputs"
    filename = f"{int(time.time())}-{uuid.uuid4().hex}.png"
    path = f"{prefix}/{filename}"
    encoded_path = urllib.parse.quote(path, safe="/")
    api_url = f"https://api.github.com/repos/{github_repo.strip()}/contents/{encoded_path}"

    body = {
        "message": f"upload kirox input {filename}",
        "content": base64.b64encode(png_bytes).decode("ascii"),
        "branch": github_branch.strip() or "main",
    }
    data = request_json(
        "PUT",
        api_url,
        {
            "Authorization": f"Bearer {github_token.strip()}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        body,
        timeout=180,
    )
    content = data.get("content") or {}
    download_url = content.get("download_url")
    if download_url:
        return download_url

    owner_repo = github_repo.strip()
    branch = urllib.parse.quote(github_branch.strip() or "main", safe="")
    return f"https://raw.githubusercontent.com/{owner_repo}/{branch}/{encoded_path}"


def submit_kirox(api_key: str, image_url: str, prompt: str, aspect_ratio: str, image_size: str) -> str:
    body = {
        "model": KIROX_MODEL,
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "image_size": image_size,
        "urls": [image_url],
    }
    data = request_json(
        "POST",
        f"{KIROX_API_BASE}/generations",
        {"Authorization": f"Bearer {api_key.strip()}"},
        body,
        timeout=120,
    )
    task_id = data.get("id")
    if not task_id:
        raise RuntimeError(f"Kirox did not return a task id: {data}")
    return task_id


def poll_kirox(api_key: str, task_id: str, poll_interval: int, timeout_seconds: int) -> str:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        data = request_json(
            "GET",
            f"{KIROX_API_BASE}/{urllib.parse.quote(task_id)}",
            {"Authorization": f"Bearer {api_key.strip()}"},
            timeout=120,
        )
        status = data.get("status")
        if status == "completed":
            results = data.get("results") or []
            if not results or not results[0].get("url"):
                raise RuntimeError(f"Kirox completed but returned no image URL: {data}")
            return results[0]["url"]
        if status == "failed":
            raise RuntimeError(f"Kirox task failed: {data}")
        time.sleep(max(1, int(poll_interval)))
    raise TimeoutError(f"Kirox task timed out: {task_id}")


def download_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=240) as response:
        return response.read()


class KiroxNanoBananaImagePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "image_size": (["4K", "2K", "1K"], {"default": "4K"}),
                "aspect_ratio": (["auto", "1:1", "9:16", "16:9", "3:4", "4:3", "2:3", "3:2", "4:5", "5:4"], {"default": "auto"}),
                "face_policy": (["preserve", "hidden", "visible"], {"default": "preserve"}),
                "poll_interval": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
                "timeout_seconds": ("INT", {"default": 900, "min": 60, "max": 3600, "step": 30}),
            },
            "optional": {
                "kirox_api_key": ("STRING", {"default": "", "multiline": False}),
                "image_url": ("STRING", {"default": "", "multiline": False}),
                "github_token": ("STRING", {"default": "", "multiline": False}),
                "github_repo": ("STRING", {"default": "tianjiaxing670-del/kirox-images", "multiline": False}),
                "github_branch": ("STRING", {"default": "main", "multiline": False}),
                "github_path_prefix": ("STRING", {"default": "kirox-inputs", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "result_url")
    FUNCTION = "generate"
    CATEGORY = "Kirox"

    def generate(
        self,
        image,
        prompt,
        image_size,
        aspect_ratio,
        face_policy,
        poll_interval,
        timeout_seconds,
        kirox_api_key="",
        image_url="",
        github_token="",
        github_repo="tianjiaxing670-del/kirox-images",
        github_branch="main",
        github_path_prefix="kirox-inputs",
    ):
        load_env_file()
        api_key = kirox_api_key.strip() or os.environ.get("KIROX_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing Kirox API key. Fill kirox_api_key or set KIROX_API_KEY in .env.")

        final_prompt = (prompt or "").strip()
        if not final_prompt:
            final_prompt = (
                "Create a realistic image variation based on the reference image. Keep the same person, outfit, "
                "environment, lighting, camera style, and image quality. Change only the body pose naturally."
            )
        final_prompt = f"{final_prompt}\n\n{FACE_POLICY_PROMPTS[face_policy]}"

        public_image_url = (image_url or "").strip()
        if not public_image_url:
            png_bytes = image_tensor_to_png_bytes(image)
            token = github_token.strip() or os.environ.get("GITHUB_TOKEN", "").strip()
            public_image_url = upload_png_to_github(
                png_bytes,
                token,
                github_repo,
                github_branch,
                github_path_prefix,
            )

        task_id = submit_kirox(api_key, public_image_url, final_prompt, aspect_ratio, image_size)
        result_url = poll_kirox(api_key, task_id, poll_interval, timeout_seconds)
        result_bytes = download_bytes(result_url)
        return (png_bytes_to_image_tensor(result_bytes), result_url)


NODE_CLASS_MAPPINGS = {
    "KiroxNanoBananaImagePrompt": KiroxNanoBananaImagePrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KiroxNanoBananaImagePrompt": "Kirox Nano Banana Pro (Image + Prompt)",
}
