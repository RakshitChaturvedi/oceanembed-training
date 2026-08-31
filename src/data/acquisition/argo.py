from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

def download_incois_argo(cfg: dict) -> None:
    source = cfg["argo"]

    if not source.get("enabled", True):
        print("INCOIS ARGO download disabled.")
        return
    url = source.get("download_url", "").strip()
    if not url:
        raise RuntimeError(
            "INCOIS ARGO is enabled but no download_url is configured.\n"
            "Use the INCOIS LAS 'Gridded Product based on Variational Analysis "
            "Methodology' product and paste its generated/download URL into "
            "configs/acquisition.yaml.\n"
            "We deliberately do not hard-code an unverified LAS endpoint."
        )

    output = Path(source["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading INCOIS ARGO -> {output}")
    request = Request(url ,headers={"User-Agent": "OceanEmbed/1.0"})

    with urlopen(request) as response, output.open("wb") as f:
        while chunk := response.read(1024*1024):
            f.write(chunk)
    print(f"Saved {output}")