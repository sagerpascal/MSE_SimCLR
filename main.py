from simclr import SimCLR
from augmented_images_loader import ImagesDataSet


def main():
    batch_size = 256
    dataset = ImagesDataSet(batch_size)
    simclr = SimCLR(dataset, batch_size)
    simclr.train()


if __name__ == "__main__":
    main()
