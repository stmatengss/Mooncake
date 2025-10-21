import os
import threading
import ctypes
from importlib import resources
from typing import Dict, Final, Optional

from torch import device as torch_device
from torch.cuda.memory import CUDAPluggableAllocator


class NVLinkAllocator:
    _instances: Dict[torch_device, CUDAPluggableAllocator] = {}
    _lock: Final = threading.Lock()

    @classmethod
    def _get_so_path(cls) -> str:
        """Dynamically locate nvlink_allocator.so in the mooncake package installation"""
        try:
            # Attempt to locate package resource
            with resources.path("mooncake", "nvlink_allocator.so") as so_path:
                if so_path.exists():
                    return str(so_path)
        except (ImportError, FileNotFoundError, TypeError):
            pass

        # Fallback strategy: check in package location via import metadata
        try:
            import mooncake

            base_path = os.path.dirname(os.path.abspath(mooncake.__file__))
            so_path = os.path.join(base_path, "nvlink_allocator.so")
            if os.path.exists(so_path):
                return so_path
        except (ImportError, FileNotFoundError, TypeError):
            raise ImportError(
                "SGLANG_MOONCAKE_CUSTOM_MEM_POOL require mooncake-transfer-engine >= 0.3.3.post2."
            )

    @classmethod
    def get_allocator(cls, device: torch_device) -> CUDAPluggableAllocator:
        with cls._lock:
            if device not in cls._instances:
                so_path = cls._get_so_path()
                cls._instances[device] = CUDAPluggableAllocator(
                    so_path, "mc_nvlink_malloc", "mc_nvlink_free"
                )
            return cls._instances[device]


class BarexAllocator:
    _instances: Dict[torch_device, CUDAPluggableAllocator] = {}
    _lock: Final = threading.Lock()
    _barex_lib = None
    _wrapper_funcs_created = False

    @classmethod
    def _get_so_path(cls) -> str:
        """Dynamically locate libaccl_barex.so for barex memory allocation"""
        # Check common system paths for libaccl_barex.so
        possible_paths = [
            "/usr/lib/libaccl_barex.so",  # Ubuntu [deb]
            "/usr/lib64/libaccl_barex.so",  # AliOS [rpm]
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Try to locate in mooncake package installation
        try:
            # Attempt to locate package resource
            with resources.path("mooncake", "libaccl_barex.so") as so_path:
                if so_path.exists():
                    return str(so_path)
        except (ImportError, FileNotFoundError, TypeError):
            pass

        # Fallback strategy: check in package location via import metadata
        try:
            import mooncake

            base_path = os.path.dirname(os.path.abspath(mooncake.__file__))
            so_path = os.path.join(base_path, "libaccl_barex.so")
            if os.path.exists(so_path):
                return so_path
        except (ImportError, FileNotFoundError, TypeError):
            pass
        
        raise ImportError(
            "BarexAllocator requires libaccl_barex.so to be installed. "
            "Please install the barex library or ensure it's in the system path."
        )

    @classmethod
    def _load_barex_lib(cls) -> ctypes.CDLL:
        """Load the barex library using ctypes"""
        if cls._barex_lib is None:
            so_path = cls._get_so_path()
            cls._barex_lib = ctypes.CDLL(so_path)
            
            # Set up the function signatures
            cls._barex_lib.u2mm_alloc_wrapper.argtypes = [ctypes.c_ssize_t, ctypes.c_int]
            cls._barex_lib.u2mm_alloc_wrapper.restype = ctypes.c_void_p
            
            cls._barex_lib.u2mm_free_wrapper.argtypes = [ctypes.c_void_p]
            cls._barex_lib.u2mm_free_wrapper.restype = None
        
        return cls._barex_lib

    @classmethod
    def _create_wrapper_functions(cls):
        """Create wrapper functions that adapt barex functions to CUDA interface"""
        if cls._wrapper_funcs_created:
            return
            
        barex_lib = cls._load_barex_lib()
        
        # Define wrapper functions that match CUDA pluggable allocator interface
        def barex_malloc(size: int, device: int, stream) -> ctypes.c_void_p:
            """Wrapper function that adapts u2mm_alloc_wrapper to CUDA interface"""
            # The barex allocator doesn't use device or stream parameters,
            # so we ignore them and just pass size and a default device (0)
            return barex_lib.u2mm_alloc_wrapper(size, device)
        
        def barex_free(ptr: ctypes.c_void_p, size: int, device: int, stream):
            """Wrapper function that adapts u2mm_free_wrapper to CUDA interface"""
            # The barex free function only needs the pointer
            # Size, device, and stream parameters are ignored
            barex_lib.u2mm_free_wrapper(ptr)
        
        # Store the wrapper functions in the class
        cls._barex_malloc = barex_malloc
        cls._barex_free = barex_free
        cls._wrapper_funcs_created = True

    @classmethod
    def get_allocator(cls, device: torch_device) -> CUDAPluggableAllocator:
        with cls._lock:
            if device not in cls._instances:
                so_path = cls._get_so_path()
                
                # Create wrapper functions if not already created
                cls._create_wrapper_functions()
                
                # Create the CUDA pluggable allocator with wrapper functions
                cls._instances[device] = CUDAPluggableAllocator(
                    so_path, "barex_malloc", "barex_free"
                )
            return cls._instances[device]
