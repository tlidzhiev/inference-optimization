import torch
from torch.export import Dim
from transformers import AutoModel, AutoTokenizer

from src.config import MODEL_NAME, ONNX_PATH


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    dummy = tokenizer(  # ty:ignore[call-non-callable]
        ['первый текст', 'второй текст'],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt',
    )

    batch = Dim('batch')
    seq = Dim('seq')

    torch.onnx.export(
        model,
        (dummy['input_ids'], dummy['attention_mask']),
        ONNX_PATH,
        input_names=['input_ids', 'attention_mask'],
        output_names=['last_hidden_state'],
        dynamic_shapes=(
            {0: batch, 1: seq},  # input_ids
            {0: batch, 1: seq},  # attention_mask
        ),
    )
    print(f'Model exported to {ONNX_PATH}')


if __name__ == '__main__':
    main()
