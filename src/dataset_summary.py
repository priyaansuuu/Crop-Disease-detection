import os
from prettytable import PrettyTable



TRAIN_DIR = "../data/Train"
VAL_DIR = "../data/Validation"

def count_images_in_folder(folder_path):
    """Counts all image files in a folder."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    return sum(
        1 for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in valid_exts
    )

def summarize_dataset(train_dir, val_dir):
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

    table = PrettyTable()
    table.field_names = ["Class Name", "Train Images", "Validation Images"]

    for cls in classes:
        train_path = os.path.join(train_dir, cls)
        val_path = os.path.join(val_dir, cls)

        train_count = count_images_in_folder(train_path) if os.path.exists(train_path) else 0
        val_count = count_images_in_folder(val_path) if os.path.exists(val_path) else 0

        table.add_row([cls, train_count, val_count])

    print(f"📊 Total Classes: {len(classes)}\n")
    print(table)

if __name__ == "__main__":
    summarize_dataset(TRAIN_DIR, VAL_DIR)
