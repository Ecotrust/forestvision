import pytest
import torch

from forestvision.transforms import Normalize, Denormalize


class TestNormalize:
    @pytest.fixture
    def data(self, request):
        setup = {
            "float": {
                "values": [0.5, 0.5],
                "expected": [0.5, 0.5],
            },
            "list": {
                "values": [
                    [0.5, 0.6],
                    [0.7, 0.8],
                ],
                "expected": [
                    [0.5, 0.6],
                    [0.7, 0.8],
                ],
            },
            "tuple": {
                "values": [
                    (0.5, 0.6),
                    (0.7, 0.8),
                ],
                "expected": [
                    [0.5, 0.6],
                    [0.7, 0.8],
                ],
            },
            "tensor": {
                "values": [
                    torch.tensor([0.5, 0.6]),
                    torch.tensor([0.7, 0.8]),
                ],
                "expected": [
                    [0.5, 0.6],
                    [0.7, 0.8],
                ],
            },
        }

        return setup[request.param]

    @pytest.mark.parametrize(
        "data", ["float", "list", "tuple", "tensor"], indirect=True
    )
    def test_init(self, data):
        tmean, tstd = data["values"]
        emean, estd = data["expected"]
        normalize = Normalize(tmean, tstd)
        assert torch.all(normalize.mean == torch.tensor([emean]))
        assert torch.all(normalize.std == torch.tensor([estd]))

    def test_normalize_image(self, sample_rgb):
        # Clone sample data
        sample = {k: v.clone() for k, v in sample_rgb.items()}
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        normalize = Normalize(mean, std)
        result = normalize(sample)

        assert result["image"].shape == sample_rgb["image"].shape

        input_value = sample_rgb["image"][0, 0, 0, 0].item()
        expected_value = (input_value - 0.5) / 0.5
        normalized_value = result["image"][0, 0, 0, 0].item()
        assert abs(normalized_value - expected_value) < 1e-5

    def test_normalize_mask(self, sample_nodata):
        sample = {k: v.clone() for k, v in sample_nodata.items()}
        mean = [0.5]
        std = [0.5]
        normalize = Normalize(mean, std, on_key="mask")
        result = normalize(sample)

        assert result["mask"].shape == sample_nodata["mask"].shape

        original_value = float(sample_nodata["mask"][0, 0, 0].item())
        expected_value = (original_value - 0.5) / 0.5
        normalized_value = float(result["mask"][0, 0, 0].item())
        assert abs(normalized_value - expected_value) < 1e-5

    def test_normalize_with_nodata(self, sample_nodata):
        sample = {k: v.clone() for k, v in sample_nodata.items()}
        mean = [0.5, 0.5, 0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5, 0.5, 0.5]
        nodata = -5.0
        normalize = Normalize(mean, std, nodata=nodata)
        result = normalize(sample)

        assert torch.all(result["image"][result["image"] == nodata] == nodata)

        indexes = torch.where(sample["image"] != nodata)
        b, c, h, w = [coord[0].item() for coord in indexes]
        original_value = sample["image"][b, c, h, w].item()
        expected_value = (original_value - 0.5) / 0.5
        normalized_value = result["image"][b, c, h, w].item()
        assert abs(normalized_value - expected_value) < 1e-5


class TestDenormalize:
    @pytest.fixture
    def data(self, request):
        setup = {
            "float": {
                "values": [0.5, 0.5],
                "expected": [0.5, 0.5],
            },
            "list": {
                "values": [
                    [0.5, 0.6],
                    [0.7, 0.8],
                ],
                "expected": [
                    [0.5, 0.6],
                    [0.7, 0.8],
                ],
            },
            "tuple": {
                "values": [
                    (0.5, 0.6),
                    (0.7, 0.8),
                ],
                "expected": [
                    [0.5, 0.6],
                    [0.7, 0.8],
                ],
            },
            "tensor": {
                "values": [
                    torch.tensor([0.5, 0.6]),
                    torch.tensor([0.7, 0.8]),
                ],
                "expected": [
                    [0.5, 0.6],
                    [0.7, 0.8],
                ],
            },
        }

        return setup[request.param]

    @pytest.mark.parametrize(
        "data", ["float", "list", "tuple", "tensor"], indirect=True
    )
    def test_init(self, data):
        tmean, tstd = data["values"]
        emean, estd = data["expected"]
        normalize = Normalize(tmean, tstd)
        assert torch.all(normalize.mean == torch.tensor([emean]))
        assert torch.all(normalize.std == torch.tensor([estd]))

    def test_denormalize(self):
        # Create a normalized tensor
        tensor = torch.tensor(
            [
                [-1.0, 0.0, 1.0],
                [-1.0, 0.0, 1.0],
                [-1.0, 0.0, 1.0],
            ]
        ).unsqueeze(0)
        mean = 0.5
        std = 0.5
        denormalize = Denormalize(mean, std)
        result = denormalize(tensor)
        expected = tensor * 0.5 + 0.5

        assert torch.allclose(result, expected)

    def test_denormalize_multichannel(self):
        tensor = torch.tensor(
            [
                [
                    [-1.0, 0.0, 1.0],
                    [-1.0, 0.0, 1.0],
                    [-1.0, 0.0, 1.0],
                ],
                [
                    [-2.0, 0.0, 2.0],
                    [-2.0, 0.0, 2.0],
                    [-2.0, 0.0, 2.0],
                ],
            ]
        ).unsqueeze(0)

        mean = [0.5, 0.6]
        std = [0.5, 0.3]
        denormalize = Denormalize(mean, std)
        result = denormalize(tensor)
        expected_ch0 = tensor[0, 0] * 0.5 + 0.5
        expected_ch1 = tensor[0, 1] * 0.3 + 0.6

        assert torch.allclose(result[0, 0], expected_ch0)
        assert torch.allclose(result[0, 1], expected_ch1)

    def test_denormalize_with_nodata(self):
        tensor = torch.tensor(
            [
                [-1.0, 0.0, 1.0],
                [-1.0, -999, 1.0],
                [-1.0, 0.0, -999],
            ]
        ).unsqueeze(0)
        mean = 0.5
        std = 0.5
        nodata = -999
        denormalize = Denormalize(mean, std, nodata=nodata)
        result = denormalize(tensor)

        assert torch.all(result[tensor == nodata] == nodata)

        mask = tensor != nodata
        expected = tensor[mask] * 0.5 + 0.5
        assert torch.allclose(result[mask], expected)

    def test_normalize_denormalize_roundtrip(self, sample_rgb):
        mean = [0.5, 0.5, 0.5]
        std = [0.25, 0.25, 0.25]

        normalize = Normalize(mean, std)
        normalized = normalize(sample_rgb.copy())

        denormalize = Denormalize(mean, std)
        denormalized = denormalized = denormalize(normalized["image"])

        assert torch.allclose(denormalized, sample_rgb["image"])
