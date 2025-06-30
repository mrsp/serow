#!/usr/bin/env python3

import unittest
import numpy as np
import tempfile
import os
import shutil
from utils import Normalizer


class TestNormalizer(unittest.TestCase):
    """Unit tests for the Normalizer class."""
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests in the class are complete."""
        # Final cleanup of any remaining files
        current_dir = os.getcwd()
        for filename in os.listdir(current_dir):
            if filename.endswith('.npy') and ('normalizer' in filename or 
                                            'innovation' in filename or 
                                            'R_history' in filename or 
                                            'R_' in filename):
                try:
                    os.remove(os.path.join(current_dir, filename))
                except (OSError, PermissionError):
                    pass
        
        # Clean up any test directories
        test_dirs = ['test', 'normalizer_test', 'stats_test']
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                try:
                    shutil.rmtree(test_dir)
                except (OSError, PermissionError):
                    pass
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.normalizer = Normalizer()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after each test method."""
        # Remove temporary directory and files
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Clean up any files that might have been saved in the current directory
        # Look for files with patterns that match the normalizer save format
        current_dir = os.getcwd()
        for filename in os.listdir(current_dir):
            if filename.endswith('.npy') and ('normalizer' in filename or 
                                            'innovation' in filename or 
                                            'R_history' in filename or 
                                            'R_' in filename):
                try:
                    os.remove(os.path.join(current_dir, filename))
                except (OSError, PermissionError):
                    pass  # Ignore errors if file is already deleted or locked
        
        # Also clean up any test directories that might have been created
        test_dirs = ['test', 'normalizer_test', 'stats_test']
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                try:
                    shutil.rmtree(test_dir)
                except (OSError, PermissionError):
                    pass  # Ignore errors if directory is already deleted or locked
    
    def test_normalizer_initialization(self):
        """Test that Normalizer initializes correctly."""
        self.assertIsNotNone(self.normalizer.innovation_stats)
        self.assertIsNotNone(self.normalizer.R_history_stats)
        self.assertIsNotNone(self.normalizer.R_stats)
        
        # Check initial dimensions
        self.assertEqual(self.normalizer.innovation_stats.dim, 3)
        self.assertEqual(self.normalizer.R_history_stats.dim, 6)
        self.assertEqual(self.normalizer.R_stats.dim, 6)
    
    def test_innovation_normalization(self):
        """Test innovation vector normalization."""
        # Test with a simple innovation vector
        innovation = np.array([1.0, 2.0, 3.0])
        normalized = self.normalizer.normalize_innovation(innovation)
        
        # Check that normalized output has correct shape
        self.assertEqual(normalized.shape, (3,))
        
        # Check that after normalization, stats are updated
        mean, std = self.normalizer.innovation_stats.get_stats()
        self.assertGreater(self.normalizer.innovation_stats.n, 0)
        
        # Test with a second innovation to ensure normalization is working
        innovation2 = np.array([4.0, 5.0, 6.0])
        normalized2 = self.normalizer.normalize_innovation(innovation2)
        
        # Now the second normalization should definitely be different from the input
        # because the statistics have been updated from the first normalization
        self.assertFalse(np.allclose(innovation2, normalized2))
        
        # Check that the normalized values are finite
        self.assertTrue(np.all(np.isfinite(normalized)))
        self.assertTrue(np.all(np.isfinite(normalized2)))
    
    def test_R_matrix_normalization(self):
        """Test covariance matrix R normalization."""
        # Create a positive definite matrix
        R = np.eye(3) * np.random.rand(3)
        R = R @ R.T
        
        # Test normalize_R method
        R_normalized = self.normalizer.normalize_R(R)
        
        # Check that normalized output has correct shape
        self.assertEqual(R_normalized.shape, (3, 3))
        
        # Check that after normalization, stats are updated
        mean, std = self.normalizer.R_stats.get_stats()
        self.assertGreater(self.normalizer.R_stats.n, 0)
        
        # Test with a second R matrix to ensure normalization is working
        R2 = np.eye(3) * np.random.rand(3)
        R2 = R2 @ R2.T
        R_normalized2 = self.normalizer.normalize_R(R2)
        
        # Now the second normalization should definitely be different from the input
        # because the statistics have been updated from the first normalization
        self.assertFalse(np.allclose(R2, R_normalized2))
        
        # Check that the normalized values are finite
        self.assertTrue(np.all(np.isfinite(R_normalized)))
        self.assertTrue(np.all(np.isfinite(R_normalized2)))
    
    def test_R_history_normalization(self):
        """Test R history matrix normalization."""
        # Create a positive definite matrix
        R_history = np.eye(3) * np.random.rand(3)
        R_history = R_history @ R_history.T
        
        # Test normalize_R_with_action method
        R_history_normalized = self.normalizer.normalize_R_with_action(R_history)
        
        # Check that normalized output has correct shape
        self.assertEqual(R_history_normalized.shape, (3, 3))
        
        # Check that after normalization, stats are updated
        mean, std = self.normalizer.R_history_stats.get_stats()
        self.assertGreater(self.normalizer.R_history_stats.n, 0)
        
        # Test with a second R_history matrix to ensure normalization is working
        R_history2 = np.eye(3) * np.random.rand(3)
        R_history2 = R_history2 @ R_history2.T
        R_history_normalized2 = self.normalizer.normalize_R_with_action(R_history2)
        
        # Now the second normalization should definitely be different from the input
        # because the statistics have been updated from the first normalization
        self.assertFalse(np.allclose(R_history2, R_history_normalized2))
        
        # Check that the normalized values are finite
        self.assertTrue(np.all(np.isfinite(R_history_normalized)))
        self.assertTrue(np.all(np.isfinite(R_history_normalized2)))
    
    def test_save_and_load_stats(self):
        """Test saving and loading of normalization statistics."""
        # Generate some data to build up statistics
        for i in range(50):
            innovation = np.random.rand(3)
            R = np.eye(3) * np.random.rand(3)
            R = R @ R.T
            R_history = np.eye(3) * np.random.rand(3)
            R_history = R_history @ R_history.T
            
            self.normalizer.normalize_innovation(innovation)
            self.normalizer.normalize_R(R)
            self.normalizer.normalize_R_with_action(R_history)
        
        # Get statistics before saving
        prev_innovation_mean, prev_innovation_std = self.normalizer.innovation_stats.get_stats()
        prev_R_history_mean, prev_R_history_std = self.normalizer.R_history_stats.get_stats()
        prev_R_mean, prev_R_std = self.normalizer.R_stats.get_stats()
        
        # Save statistics
        self.normalizer.save_stats(self.temp_dir + '/', 'normalizer')
        
        # Reset statistics
        self.normalizer.reset_stats()
        
        # Verify reset worked
        new_innovation_mean, new_innovation_std = self.normalizer.innovation_stats.get_stats()
        new_R_history_mean, new_R_history_std = self.normalizer.R_history_stats.get_stats()
        new_R_mean, new_R_std = self.normalizer.R_stats.get_stats()
        
        # Check that stats were reset
        self.assertEqual(self.normalizer.innovation_stats.n, 0)
        self.assertEqual(self.normalizer.R_history_stats.n, 0)
        self.assertEqual(self.normalizer.R_stats.n, 0)
        
        # Load statistics back
        self.normalizer.load_stats(self.temp_dir + '/', 'normalizer')
        
        # Get statistics after loading
        loaded_innovation_mean, loaded_innovation_std = self.normalizer.innovation_stats.get_stats()
        loaded_R_history_mean, loaded_R_history_std = self.normalizer.R_history_stats.get_stats()
        loaded_R_mean, loaded_R_std = self.normalizer.R_stats.get_stats()
        
        # Check that loaded statistics match the original ones
        np.testing.assert_array_almost_equal(loaded_innovation_mean, prev_innovation_mean)
        np.testing.assert_array_almost_equal(loaded_innovation_std, prev_innovation_std)
        np.testing.assert_array_almost_equal(loaded_R_history_mean, prev_R_history_mean)
        np.testing.assert_array_almost_equal(loaded_R_history_std, prev_R_history_std)
        np.testing.assert_array_almost_equal(loaded_R_mean, prev_R_mean)
        np.testing.assert_array_almost_equal(loaded_R_std, prev_R_std)
    
    def test_reset_stats(self):
        """Test that reset_stats properly resets all statistics."""
        # Generate some data to build up statistics
        for i in range(10):
            innovation = np.random.rand(3)
            R = np.eye(3) * np.random.rand(3)
            R = R @ R.T
            R_history = np.eye(3) * np.random.rand(3)
            R_history = R_history @ R_history.T
            
            self.normalizer.normalize_innovation(innovation)
            self.normalizer.normalize_R(R)
            self.normalizer.normalize_R_with_action(R_history)
        
        # Verify that stats were accumulated
        self.assertGreater(self.normalizer.innovation_stats.n, 0)
        self.assertGreater(self.normalizer.R_history_stats.n, 0)
        self.assertGreater(self.normalizer.R_stats.n, 0)
        
        # Reset statistics
        self.normalizer.reset_stats()
        
        # Verify that all stats were reset
        self.assertEqual(self.normalizer.innovation_stats.n, 0)
        self.assertEqual(self.normalizer.R_history_stats.n, 0)
        self.assertEqual(self.normalizer.R_stats.n, 0)
        
        # Check that means are zero and stds are one (initial values)
        innovation_mean, innovation_std = self.normalizer.innovation_stats.get_stats()
        R_history_mean, R_history_std = self.normalizer.R_history_stats.get_stats()
        R_mean, R_std = self.normalizer.R_stats.get_stats()
        
        np.testing.assert_array_almost_equal(innovation_mean, np.zeros(3))
        np.testing.assert_array_almost_equal(innovation_std, np.ones(3))
        np.testing.assert_array_almost_equal(R_history_mean, np.zeros(6))
        np.testing.assert_array_almost_equal(R_history_std, np.ones(6))
        np.testing.assert_array_almost_equal(R_mean, np.zeros(6))
        np.testing.assert_array_almost_equal(R_std, np.ones(6))
    
    def test_get_normalization_stats(self):
        """Test the get_normalization_stats method."""
        # Generate some data
        for i in range(5):
            innovation = np.random.rand(3)
            R = np.eye(3) * np.random.rand(3)
            R = R @ R.T
            R_history = np.eye(3) * np.random.rand(3)
            R_history = R_history @ R_history.T
            
            self.normalizer.normalize_innovation(innovation)
            self.normalizer.normalize_R(R)
            self.normalizer.normalize_R_with_action(R_history)
        
        # Get normalization stats
        stats = self.normalizer.get_normalization_stats()
        
        # Check that all expected keys are present
        expected_keys = ['innovation_mean', 'innovation_std', 'R_mean', 'R_std', 
                        'R_history_mean', 'R_history_std', 'n_samples']
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Check that n_samples matches the innovation stats count
        self.assertEqual(stats['n_samples'], self.normalizer.innovation_stats.n)
        
        # Check that the stats values match the individual get_stats() calls
        innovation_mean, innovation_std = self.normalizer.innovation_stats.get_stats()
        R_mean, R_std = self.normalizer.R_stats.get_stats()
        R_history_mean, R_history_std = self.normalizer.R_history_stats.get_stats()
        
        np.testing.assert_array_almost_equal(stats['innovation_mean'], innovation_mean)
        np.testing.assert_array_almost_equal(stats['innovation_std'], innovation_std)
        np.testing.assert_array_almost_equal(stats['R_mean'], R_mean)
        np.testing.assert_array_almost_equal(stats['R_std'], R_std)
        np.testing.assert_array_almost_equal(stats['R_history_mean'], R_history_mean)
        np.testing.assert_array_almost_equal(stats['R_history_std'], R_history_std)
    
    def test_covariance_matrix_positive_definite(self):
        """Test that covariance matrix normalization handles positive definite matrices correctly."""
        # Create a positive definite matrix using Cholesky decomposition
        L = np.random.rand(3, 3)
        L = np.tril(L)  # Make it lower triangular
        R = L @ L.T  # This ensures positive definiteness
        
        # Test normalization
        R_normalized = self.normalizer.normalize_R(R)
        
        # Check that the result is still a valid matrix
        self.assertEqual(R_normalized.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(R_normalized)))
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with very small values
        innovation = np.array([1e-10, 1e-10, 1e-10])
        normalized = self.normalizer.normalize_innovation(innovation)
        self.assertEqual(normalized.shape, (3,))
        self.assertTrue(np.all(np.isfinite(normalized)))
        
        # Test with very large values
        innovation = np.array([1e10, 1e10, 1e10])
        normalized = self.normalizer.normalize_innovation(innovation)
        self.assertEqual(normalized.shape, (3,))
        self.assertTrue(np.all(np.isfinite(normalized)))
        
        # Test with identity matrix
        R = np.eye(3)
        R_normalized = self.normalizer.normalize_R(R)
        self.assertEqual(R_normalized.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(R_normalized)))


if __name__ == '__main__':
    # Run the tests
    try:
        unittest.main(verbosity=2)
    finally:
        # Final cleanup after all tests
        print("\nCleaning up test files...")
        current_dir = os.getcwd()
        cleaned_files = []
        
        # Clean up any remaining .npy files
        for filename in os.listdir(current_dir):
            if filename.endswith('.npy') and ('normalizer' in filename or 
                                            'innovation' in filename or 
                                            'R_history' in filename or 
                                            'R_' in filename):
                try:
                    os.remove(os.path.join(current_dir, filename))
                    cleaned_files.append(filename)
                except (OSError, PermissionError):
                    pass
        
        # Clean up any test directories
        test_dirs = ['test', 'normalizer_test', 'stats_test']
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                try:
                    shutil.rmtree(test_dir)
                    cleaned_files.append(test_dir)
                except (OSError, PermissionError):
                    pass
        
        if cleaned_files:
            print(f"Cleaned up: {', '.join(cleaned_files)}")
        else:
            print("No test files to clean up.")
