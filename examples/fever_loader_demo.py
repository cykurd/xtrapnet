import os
from xtrapnet.benchmarks import FeverDataset


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fever")
    ds = FeverDataset(data_dir=data_dir, ood_subject_ratio=0.5, val_ratio=0.1)
    ds.load_data()
    split = ds.get_split("train")

    print({
        "train_size": int(len(split.train_data)),
        "test_size": int(len(split.test_data)),
        "val_size": int(split.metadata.get("val_size", 0)),
        "n_subjects_total": int(split.metadata.get("n_subjects_total", 0)),
        "n_subjects_id": int(split.metadata.get("n_subjects_id", 0)),
        "n_subjects_ood": int(split.metadata.get("n_subjects_ood", 0)),
    })


if __name__ == "__main__":
    main()


