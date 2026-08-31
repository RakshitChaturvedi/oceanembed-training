from __future__ import annotations

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.acquisition.argo import download_incois_argo
from src.data.acquisition.copernicus import acquire_copernicus
#from src.data.acquisition.podaac import acquire_podaac
from src.data.acquisition.oscar import acquire_oscar


def main() -> None:
    config_path = ROOT / "configs" / "acquisition.yaml"

    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    Path(cfg["output_root"]).mkdir(parents=True, exist_ok=True)

    print("=== OceanEmbed Phase 1: Data Acquisition ===")
    print(f"Date range: {cfg['dates']['start']} -> {cfg['dates']['end']}")
    print(
        "Domain: "
        f'{cfg["domain"]["lat_min"]}..{cfg["domain"]["lat_max"]} N, '
        f'{cfg["domain"]["lon_min"]}..{cfg["domain"]["lon_max"]} E'
    )

    print("\n[1/3] Copernicus")
    acquire_copernicus(cfg)

    print("\n[2/3] NASA PO.DAAC")
    #acquire_podaac(cfg)

    print("\n[3/3] OSCAR Surface Currents.")
    acquire_oscar(cfg)
    """
    print("\n[3/3] INCOIS ARGO")
    download_incois_argo(cfg)
    """
    print("\nPhase 1 acquisition finished.")


if __name__ == "__main__":
    main()