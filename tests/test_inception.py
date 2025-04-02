from metrics.inception_score import InceptionScoreMetric
from utils import create_dummy_tensor

if __name__ == "__main__":
    inception_metric = InceptionScoreMetric()

    batch_size = 16
    test_tensor = create_dummy_tensor(batch_size)

    inception_metric.update_images(test_tensor)

    inception_score = inception_metric.compute()
    print("Inception score:", inception_score)
