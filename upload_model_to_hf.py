"""
Helper script to upload local trained model weights to Hugging Face Model Hub.
Usage:
    python upload_model_to_hf.py
"""
import os
import sys

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("Error: 'huggingface_hub' is not installed. Please run: pip install huggingface-hub")
    sys.exit(1)

LOCAL_MODEL = os.path.join("models", "swin_model_new.pth")
REPO_NAME = "skin-cancer-swin"

def main():
    if not os.path.exists(LOCAL_MODEL):
        print(f"Error: Model file '{LOCAL_MODEL}' not found.")
        sys.exit(1)

    print("=" * 60)
    print("Hugging Face Model Hub Uploader for AI-Medico")
    print("=" * 60)
    print("To upload, you need a Hugging Face Access Token with 'Write' permissions.")
    print("Get one here: https://huggingface.co/settings/tokens (Role: Write)")
    print("-" * 60)

    token = input("Paste your HF Write Token (or press Enter to use existing token): ").strip()
    if not token:
        # Check v.txt as fallback if present
        if os.path.exists("v.txt"):
            with open("v.txt", "r") as f:
                token = f.read().strip()
        else:
            token = None

    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        username = user_info["name"]
        repo_id = f"{username}/{REPO_NAME}"
        print(f"\nAuthenticated as: {username}")
        print(f"Target Repository: https://huggingface.co/{repo_id}")

        print(f"\n1. Ensuring public repository exists: {repo_id} ...")
        create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True, token=token)

        print(f"2. Uploading '{LOCAL_MODEL}' (~110 MB) to Hugging Face...")
        print("   (This may take 1-3 minutes depending on your internet upload speed)...")
        api.upload_file(
            path_or_fileobj=LOCAL_MODEL,
            path_in_repo="swin_model_new.pth",
            repo_id=repo_id,
            repo_type="model",
            token=token
        )

        print("\n" + "=" * 60)
        print(" SUCCESS: Model uploaded to Hugging Face Model Hub!")
        print(f" Model link: https://huggingface.co/{repo_id}")
        print(" Your Streamlit app is now ready to download this model on Streamlit Cloud!")
        print("=" * 60)

    except Exception as e:
        print(f"\n Upload failed: {e}")
        print("\nAlternative (Super Simple manual upload):")
        print("1. Go to https://huggingface.co/new and create a model repo named 'skin-cancer-swin' (Public).")
        print("2. Click 'Add file' -> 'Upload files' and drag 'models/swin_model_new.pth' into it.")
        print("3. Click 'Commit changes to main'. Done!")

if __name__ == "__main__":
    main()
