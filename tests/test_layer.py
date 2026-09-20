"""Tests for KANLayer and KANNetwork."""

import torch
import pytest

from inkan import KANLayer, KANNetwork


class TestKANLayer:

    def test_forward_shape(self):
        layer = KANLayer(32, 16)
        x = torch.randn(8, 32)
        y = layer(x)
        assert y.shape == (8, 16)

    def test_backward(self):
        layer = KANLayer(32, 16)
        x = torch.randn(8, 32)
        y = layer(x)
        y.sum().backward()
        # Gradients flow to the packed weight parameter
        assert layer.weight.grad is not None

    def test_parameter_count_1d(self):
        layer = KANLayer(32, 16, grid_size=5, spline_order=3)
        n_bases = 5 + 3
        expected = 16 * 32 * n_bases + 16 * 32  # spline + base
        actual = sum(p.numel() for p in layer.parameters())
        assert actual == expected

    def test_parameter_count_2d(self):
        layer = KANLayer(2, 4, dim=2, grid_size=5, spline_order=3)
        n_bases = 5 + 3
        expected = 4 * (n_bases * n_bases + 2)  # spline K*K + base 2
        actual = sum(p.numel() for p in layer.parameters())
        assert actual == expected

    @pytest.mark.parametrize("grid_size", [3, 5, 10, 20])
    def test_various_grid_sizes(self, grid_size):
        layer = KANLayer(16, 8, grid_size=grid_size)
        x = torch.randn(4, 16)
        y = layer(x)
        assert y.shape == (4, 8)
        y.sum().backward()

    def test_custom_grid_range(self):
        layer = KANLayer(16, 8, grid_range=(-2.0, 2.0))
        x = torch.randn(4, 16)
        y = layer(x)
        assert y.shape == (4, 8)

    def test_basis_mode_dense(self):
        layer = KANLayer(16, 8, basis_mode="dense")
        x = torch.randn(4, 16)
        y = layer(x)
        assert y.shape == (4, 8)
        y.sum().backward()

    def test_basis_mode_invalid(self):
        with pytest.raises(ValueError):
            KANLayer(16, 8, basis_mode="invalid")

    def test_2d_forward_backward(self):
        layer = KANLayer(2, 4, dim=2)
        x = torch.randn(8, 2)
        y = layer(x)
        assert y.shape == (8, 4)
        y.sum().backward()
        assert layer.weight.grad is not None

    def test_spline_weight_view(self):
        layer = KANLayer(32, 16, grid_size=5)
        assert layer.spline_weight.shape == (16, 32, 8)
        # Writing to view should modify the packed weight
        with torch.no_grad():
            layer.spline_weight.fill_(0.0)
        IK = 32 * 8
        assert (layer.weight[:, :IK] == 0).all()

    def test_repr(self):
        layer = KANLayer(32, 16, grid_size=10)
        s = repr(layer)
        assert "32" in s
        assert "16" in s
        assert "10" in s

    @pytest.mark.skipif(not torch.backends.mps.is_available(),
                        reason="MPS not available")
    def test_mps_device(self):
        layer = KANLayer(32, 16).to("mps")
        x = torch.randn(8, 32, device="mps")
        y = layer(x)
        assert y.device.type == "mps"
        y.sum().backward()

    @pytest.mark.skipif(not torch.cuda.is_available(),
                        reason="CUDA not available")
    def test_cuda_device(self):
        layer = KANLayer(32, 16).to("cuda")
        x = torch.randn(8, 32, device="cuda")
        y = layer(x)
        assert y.device.type == "cuda"
        y.sum().backward()


class TestKANNetwork:

    def test_forward_shape(self):
        net = KANNetwork([784, 64, 10])
        x = torch.randn(32, 784)
        y = net(x)
        assert y.shape == (32, 10)

    def test_three_layers(self):
        net = KANNetwork([128, 64, 32, 10])
        assert len(net.layers) == 3
        x = torch.randn(8, 128)
        y = net(x)
        assert y.shape == (8, 10)
        y.sum().backward()

    def test_minimum_dims(self):
        net = KANNetwork([8, 4])
        x = torch.randn(2, 8)
        y = net(x)
        assert y.shape == (2, 4)

    def test_invalid_dims(self):
        with pytest.raises(ValueError):
            KANNetwork([8])

    def test_basis_mode_passthrough(self):
        net = KANNetwork([8, 4], basis_mode="dense")
        assert net.layers[0].basis_mode == "dense"
