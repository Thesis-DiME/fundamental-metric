from metrics.fid_score import FIDMetric
from utils import create_dummy_tensor

if __name__ == "__main__":
    fid_metric = FIDMetric()

    batch_size = 16
    test_real_tensor = create_dummy_tensor(batch_size)
    test_generated_tensor = create_dummy_tensor(batch_size)

    fid_metric.update_real_images(
        reference_images=test_real_tensor,
    )

    fid_metric.update_generated_images(
        generated_images=test_generated_tensor,
    )

    fid_score = fid_metric.compute()
    print("FID Score:", fid_score)
