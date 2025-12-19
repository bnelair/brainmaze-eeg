"""Test numpy compatibility across versions."""
import pytest
import numpy as np


def test_numpy_version():
    """Verify numpy is installed and version is accessible."""
    assert hasattr(np, '__version__')
    version_parts = np.__version__.split('.')
    major_version = int(version_parts[0])
    assert major_version >= 1, f"NumPy version {np.__version__} is too old"
    print(f"Testing with NumPy version: {np.__version__}")


def test_numpy_concatenate():
    """Test that np.concatenate works correctly across numpy versions."""
    # Test 1D arrays
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    result = np.concatenate([a, b])
    assert result.shape == (6,)
    assert np.array_equal(result, np.array([1, 2, 3, 4, 5, 6]))
    
    # Test 2D arrays with axis=0
    c = np.array([[1, 2], [3, 4]])
    d = np.array([[5, 6]])
    result = np.concatenate([c, d], axis=0)
    assert result.shape == (3, 2)
    
    # Test 2D arrays with axis=1
    e = np.array([[1], [2]])
    f = np.array([[3], [4]])
    result = np.concatenate([e, f], axis=1)
    assert result.shape == (2, 2)
    assert np.array_equal(result, np.array([[1, 3], [2, 4]]))


def test_numpy_basic_operations():
    """Test basic numpy operations work across versions."""
    # Array creation
    arr = np.random.randn(10, 5)
    assert arr.shape == (10, 5)
    
    # Basic operations
    arr_mean = np.mean(arr, axis=0)
    assert arr_mean.shape == (5,)
    
    arr_std = np.std(arr, axis=1)
    assert arr_std.shape == (10,)
    
    # NaN handling
    arr_with_nan = arr.copy()
    arr_with_nan[0, 0] = np.nan
    assert np.isnan(arr_with_nan[0, 0])
    assert not np.isnan(np.nanmean(arr_with_nan))
    
    # Stack operations
    stacked = np.stack([arr, arr], axis=0)
    assert stacked.shape == (2, 10, 5)


def test_numpy_dtypes():
    """Test numpy data types compatibility."""
    # Float types
    arr_float64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    assert arr_float64.dtype == np.float64
    
    arr_float32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert arr_float32.dtype == np.float32
    
    # Integer types
    arr_int64 = np.array([1, 2, 3], dtype=np.int64)
    assert arr_int64.dtype == np.int64
    
    # Boolean
    arr_bool = np.array([True, False, True], dtype=bool)
    assert arr_bool.dtype == bool
