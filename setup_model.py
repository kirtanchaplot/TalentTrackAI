import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def main():
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Model URL (using a smaller model for faster download)
    model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
    model_path = "models/llama-2-7b-chat.gguf"
    
    print("Starting model download...")
    print("This may take a while depending on your internet connection.")
    print("The model is about 4GB in size.")
    
    try:
        download_file(model_url, model_path)
        print("\nModel downloaded successfully!")
        print(f"Model saved to: {model_path}")
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        print("Please try downloading manually from: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF")
        print("And place the model file in the 'models' directory.")

if __name__ == "__main__":
    main() 