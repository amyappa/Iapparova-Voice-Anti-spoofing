import torch


def collate_fn(dataset_items: list[dict]):
    data_object = torch.stack([item["data_object"] for item in dataset_items])

    labels = torch.tensor([item["labels"] for item in dataset_items])

    return {"data_object": data_object, "labels": labels}
