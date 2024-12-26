import torch
from transformers import CLIPProcessor, CLIPModel

# ViT-L/14 모델 URL 및 경로
model_url = "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
model_path = "ViT-L-14.pt"

torch.hub.download_url_to_file(model_url, model_path)

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def get_parameter_count(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params}")
        print(f"Trainable Parameters: {trainable_params}")
    return total_params

args.num_base_model_parameters = int(get_parameter_count(model, verbose=True))

# 임베딩 추출 함수
def get_embeddings(model, processor, text_inputs, image_inputs):
    text_inputs = processor(text=text_inputs, return_tensors="pt", padding=True, truncation=True)
    image_inputs = processor(images=image_inputs, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = model.get_text_features(**text_inputs)
        image_embeddings = model.get_image_features(**image_inputs)
    return text_embeddings, image_embeddings

# 데이터셋 임베딩 추출 함수
def get_dataset_embeddings(model, processor, dataset):
    text_embeddings_list = []
    image_embeddings_list = []
    for text, image in dataset:
        text_embedding, image_embedding = get_embeddings(model, processor, [text], [image])
        text_embeddings_list.append(text_embedding)
        image_embeddings_list.append(image_embedding)
    return torch.cat(text_embeddings_list), torch.cat(image_embeddings_list)

# 제로샷 예측 함수
def get_zeroshot_predictions(model, processor, texts, images):
    text_embeddings, image_embeddings = get_embeddings(model, processor, texts, images)
    similarities = torch.matmul(image_embeddings, text_embeddings.T)
    return similarities

# 예제 텍스트 및 이미지 데이터
texts = ["a photo of a cat", "a photo of a dog"]
images = [torch.randn(3, 224, 224), torch.randn(3, 224, 224)]  # 임의의 이미지 텐서

# 제로샷 예측 실행
similarities = get_zeroshot_predictions(model, processor, texts, images)
print(similarities)
