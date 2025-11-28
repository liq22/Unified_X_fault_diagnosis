"""
Unit tests for explainable fault diagnosis models
"""

import unittest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_collection.GradCAM_XFD import GradCAM_XFD
from model_collection.CI_GNN import CI_GNN_XFD
from model_collection.Physics_informed_PDN import PhysicsInformedPDN_XFD


class TestExplainableModels(unittest.TestCase):
    """Test suite for explainable fault diagnosis models"""

    def setUp(self):
        """Set up test data and configurations"""
        self.batch_size = 8
        self.seq_length = 1024
        self.num_classes = 10
        self.device = torch.device('cpu')

        # Generate dummy data
        self.test_data = torch.randn(self.batch_size, 1, self.seq_length)
        self.test_labels = torch.randint(0, self.num_classes, (self.batch_size,))

        # Create dataset and dataloader
        self.test_dataset = TensorDataset(self.test_data, self.test_labels)
        self.test_loader = DataLoader(self.test_dataset, batch_size=4, shuffle=False)

        # Model configurations
        self.config_gradcam = {
            'input_channels': 1,
            'num_classes': self.num_classes,
            'seq_length': self.seq_length,
            'dropout': 0.2
        }

        self.config_cignn = {
            'num_sensors': 4,
            'num_classes': self.num_classes,
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.2
        }

        self.config_physics = {
            'input_dim': self.seq_length,
            'num_classes': self.num_classes,
            'hidden_dim': 64,
            'num_samples': 5,
            'physics_params': {
                'resonance_freq': 100.0,
                'damping_ratio': 0.1,
                'freq_range': [0, 1000]
            }
        }

    def test_grad_cam_initialization(self):
        """Test GradCAM-XFD model initialization"""
        model = GradCAM_XFD(self.config_gradcam)
        self.assertIsNotNone(model.model)
        self.assertEqual(model.model.input_channels, 1)
        self.assertEqual(model.model.num_classes, self.num_classes)

    def test_ci_gnn_initialization(self):
        """Test CI-GNN model initialization"""
        model = CI_GNN_XFD(self.config_cignn)
        self.assertIsNotNone(model.model)
        self.assertEqual(model.model.num_sensors, 4)
        self.assertEqual(model.model.num_classes, self.num_classes)

    def test_physics_pdn_initialization(self):
        """Test Physics-informed PDN model initialization"""
        model = PhysicsInformedPDN_XFD(self.config_physics)
        self.assertIsNotNone(model.model)
        self.assertEqual(model.model.num_classes, self.num_classes)
        self.assertEqual(model.model.num_samples, 5)

    def test_grad_cam_forward(self):
        """Test GradCAM-XFD forward pass"""
        model = GradCAM_XFD(self.config_gradcam)
        model.model.eval()

        with torch.no_grad():
            logits = model.model(self.test_data)
            self.assertEqual(logits.shape, (self.batch_size, self.num_classes))

    def test_ci_gnn_forward(self):
        """Test CI-GNN forward pass"""
        model = CI_GNN_XFD(self.config_cignn)
        model.model.eval()

        with torch.no_grad():
            logits, explanations = model.model(self.test_data)
            self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
            self.assertIn('causal_matrix', explanations)

    def test_physics_pdn_forward(self):
        """Test Physics-informed PDN forward pass"""
        model = PhysicsInformedPDN_XFD(self.config_physics)
        model.model.eval()

        with torch.no_grad():
            logits, uncertainty, explanations = model.model(self.test_data, return_uncertainty=True)
            self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
            self.assertEqual(uncertainty.shape, (self.batch_size,))
            self.assertIn('feature_importance', explanations)

    def test_grad_cam_explanation(self):
        """Test GradCAM-XFD explanation generation"""
        model = GradCAM_XFD(self.config_gradcam)
        model.model.eval()

        # Test single sample explanation
        single_input = self.test_data[:1]
        explanation = model.model.get_explanation(single_input)

        self.assertIn('prediction', explanation)
        self.assertIn('probabilities', explanation)
        self.assertIn('cam', explanation)
        self.assertIn('important_regions', explanation)

    def test_ci_gnn_explanation(self):
        """Test CI-GNN explanation generation"""
        model = CI_GNN_XFD(self.config_cignn)
        model.model.eval()

        # Test single sample explanation
        explanation = model.model.get_explanation(self.test_data[:1])

        self.assertIn('prediction', explanation)
        self.assertIn('confidence', explanation)
        self.assertIn('causal_relationships', explanation)
        self.assertIn('sensor_importance', explanation)

    def test_physics_pdn_explanation(self):
        """Test Physics-informed PDN explanation generation"""
        model = PhysicsInformedPDN_XFD(self.config_physics)
        model.model.eval()

        # Test single sample explanation
        explanation = model.model.get_explanation(self.test_data[:1])

        self.assertIn('prediction', explanation)
        self.assertIn('confidence', explanation)
        self.assertIn('prediction_uncertainty', explanation)
        self.assertIn('reliability_score', explanation)

    def test_grad_cam_training_step(self):
        """Test GradCAM-XFD training step"""
        model = GradCAM_XFD(self.config_gradcam)
        model.model.train()

        # Single training step
        data, target = next(iter(self.test_loader))
        model.optimizer.zero_grad()
        output = model.model(data)
        loss = model.criterion(output, target)
        loss.backward()
        model.optimizer.step()

        self.assertLess(loss.item(), 10.0)  # Reasonable loss value

    def test_ci_gnn_training_step(self):
        """Test CI-GNN training step"""
        model = CI_GNN_XFD(self.config_cignn)
        model.model.train()

        # Single training step
        data, target = next(iter(self.test_loader))
        model.optimizer.zero_grad()
        logits, _ = model.model(data)
        loss = model.criterion(logits, target)
        loss.backward()
        model.optimizer.step()

        self.assertLess(loss.item(), 10.0)

    def test_physics_pdn_training_step(self):
        """Test Physics-informed PDN training step"""
        model = PhysicsInformedPDN_XFD(self.config_physics)
        model.model.train()

        # Single training step
        data, target = next(iter(self.test_loader))
        model.optimizer.zero_grad()
        logits, uncertainty, _ = model.model(data, return_uncertainty=True)
        cls_loss = model.criterion(logits, target)
        unc_loss = torch.mean(uncertainty)
        total_loss = cls_loss + 0.01 * unc_loss
        total_loss.backward()
        model.optimizer.step()

        self.assertLess(total_loss.item(), 10.0)

    def test_model_predictions(self):
        """Test model predictions"""
        # Test GradCAM
        model = GradCAM_XFD(self.config_gradcam)
        pred, prob = model.predict(self.test_data)
        self.assertEqual(pred.shape, (self.batch_size,))
        self.assertEqual(prob.shape, (self.batch_size, self.num_classes))

        # Test CI-GNN
        model = CI_GNN_XFD(self.config_cignn)
        pred, prob = model.predict(self.test_data)
        self.assertEqual(pred.shape, (self.batch_size,))
        self.assertEqual(prob.shape, (self.batch_size, self.num_classes))

        # Test Physics-PDN
        model = PhysicsInformedPDN_XFD(self.config_physics)
        pred, prob, unc = model.predict(self.test_data)
        self.assertEqual(pred.shape, (self.batch_size,))
        self.assertEqual(prob.shape, (self.batch_size, self.num_classes))
        self.assertEqual(unc.shape, (self.batch_size,))

    def test_model_explain(self):
        """Test model explain method"""
        # Test GradCAM
        model = GradCAM_XFD(self.config_gradcam)
        explanations = model.explain(self.test_data[:2])
        self.assertEqual(len(explanations), 2)
        self.assertIn('confidence', explanations[0])

        # Test CI-GNN
        model = CI_GNN_XFD(self.config_cignn)
        explanations = model.explain(self.test_data[:2])
        self.assertEqual(len(explanations), 2)
        self.assertIn('causal_relationships', explanations[0])

        # Test Physics-PDN
        model = PhysicsInformedPDN_XFD(self.config_physics)
        explanations = model.explain(self.test_data[:2])
        self.assertEqual(len(explanations), 2)
        self.assertIn('prediction_uncertainty', explanations[0])

    def test_model_save_load(self):
        """Test model saving and loading"""
        # Test GradCAM
        model = GradCAM_XFD(self.config_gradcam)
        save_path = 'test_grad_cam_model.pth'
        model.save_model(save_path)

        # Create new model and load
        new_model = GradCAM_XFD(self.config_gradcam)
        new_model.load_model(save_path)

        # Check predictions are the same
        pred1, _ = model.predict(self.test_data)
        pred2, _ = new_model.predict(self.test_data)
        np.testing.assert_array_equal(pred1, pred2)

        # Clean up
        os.remove(save_path)

    def test_different_input_sizes(self):
        """Test models with different input sizes"""
        # Test with different sequence lengths
        for seq_len in [512, 2048]:
            data = torch.randn(4, 1, seq_len)

            # GradCAM
            config = self.config_gradcam.copy()
            config['seq_length'] = seq_len
            model = GradCAM_XFD(config)
            pred, _ = model.predict(data)
            self.assertEqual(pred.shape[0], 4)

            # Physics-PDN
            config = self.config_physics.copy()
            config['input_dim'] = seq_len
            model = PhysicsInformedPDN_XFD(config)
            pred, _, _ = model.predict(data)
            self.assertEqual(pred.shape[0], 4)

    def test_edge_cases(self):
        """Test edge cases"""
        # Test with single sample
        single_data = torch.randn(1, 1, self.seq_length)
        model = GradCAM_XFD(self.config_gradcam)
        pred, prob = model.predict(single_data)
        self.assertEqual(pred.shape, (1,))
        self.assertEqual(prob.shape, (1, self.num_classes))

        # Test with extreme values
        extreme_data = torch.randn(2, 1, self.seq_length) * 100
        model = GradCAM_XFD(self.config_gradcam)
        pred, prob = model.predict(extreme_data)
        self.assertTrue(np.all(prob >= 0))
        self.assertTrue(np.all(prob <= 1))


if __name__ == '__main__':
    unittest.main()