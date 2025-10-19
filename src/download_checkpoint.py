import os
import subprocess
from pathlib import Path
import argparse
import traceback

MODEL_LINKS = {
    "pi0_base": "gs://openpi-assets/checkpoints/pi0_base",

    "pi0_fast_base": "gs://openpi-assets/checkpoints/pi0_fast_base",
    "pi0_fast_droid": "gs://openpi-assets/checkpoints/pi0_fast_droid",

    "pi05_base": "gs://openpi-assets/checkpoints/pi05_base",
    "pi05_droid": "gs://openpi-assets/checkpoints/pi05_droid"

}

def ensure_gsutil_download(model_name: str, model_link: str, target_dir: Path):
    local_model_dir = target_dir / model_name
    local_model_dir.mkdir(parents=True, exist_ok=True)

    if any(local_model_dir.iterdir()):
        print(f"[INFO] Model exists in {local_model_dir}, skip downloading.")
        return local_model_dir

    print(f"[INFO] Using gsutil downloading model: {model_name}")
    try:
        subprocess.run(
            [
                "gsutil", "-m", "cp", "-r",
                f"{model_link}",
                str(target_dir),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] gsutil download failed: {e}")
        raise

    print(f"[INFO] Model is downloaded to: {local_model_dir}")
    return local_model_dir


def main():
    parser = argparse.ArgumentParser(description="Download and load OpenPI model using gsutil.")
    parser.add_argument("--model", required=True, choices=MODEL_LINKS.keys(),
                        help="Model name to download and load.")
    args = parser.parse_args()

    current_file = Path(__file__).resolve()
    parent_dir = current_file.parent.parent
    target_dir = parent_dir / "models" / "official_checkpoints"
    os.environ["OPENPI_DATA_HOME"] = str(target_dir)

    model_name = args.model
    model_link = MODEL_LINKS[model_name]

    try:
        checkpoint_dir = ensure_gsutil_download(model_name, model_link, target_dir)

        from openpi.training import config
        from openpi.policies import policy_config

        config = config.get_config(model_name)
        policy = policy_config.create_trained_policy(config, checkpoint_dir)

    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
