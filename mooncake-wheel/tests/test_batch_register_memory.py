import unittest
import os
import ctypes
from mooncake.engine import TransferEngine


class TestBatchRegisterMemory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target_server_name = os.getenv("TARGET_SERVER_NAME", "127.0.0.1:12345")
        cls.initiator_server_name = os.getenv("INITIATOR_SERVER_NAME", "127.0.0.1:12347")
        cls.metadata_server = os.getenv("MC_METADATA_SERVER", "127.0.0.1:2379")
        cls.protocol = os.getenv("PROTOCOL", "tcp")
        cls.circle = int(os.getenv("CIRCLE", 100))

        cls.adaptor = TransferEngine()
        ret = cls.adaptor.initialize(
            cls.initiator_server_name,
            cls.metadata_server,
            cls.protocol,
            ""
        )
        if ret != 0:
            raise RuntimeError(f"Initialization failed with code {ret}")

    def test_batch_register_memory_basic(self):
        """Test basic functionality of batch_register_memory with single buffer."""
        adaptor = self.adaptor
        
        # Create buffer using ctypes
        buffer_size = 1024 * 1024  # 1MB
        buffer = ctypes.create_string_buffer(buffer_size)
        buffer_addr = ctypes.addressof(buffer)
        
        # Register memory using batch API
        buffer_addresses = [buffer_addr]
        capacities = [buffer_size]
        
        result = adaptor.batch_register_memory(buffer_addresses, capacities)
        self.assertEqual(result, 0, "batch_register_memory failed")
        
        # Clean up
        result = adaptor.batch_unregister_memory(buffer_addresses)
        self.assertEqual(result, 0, "batch_unregister_memory failed")
        
        print("[✓] Basic batch_register_memory test passed")

    def test_batch_register_memory_multiple_buffers(self):
        """Test batch_register_memory with multiple buffers."""
        adaptor = self.adaptor
        
        # Create multiple buffers using ctypes
        buffer_sizes = [1024 * 1024, 2 * 1024 * 1024, 512 * 1024]  # 1MB, 2MB, 512KB
        buffers = []
        buffer_addresses = []
        
        for size in buffer_sizes:
            buffer = ctypes.create_string_buffer(size)
            buffers.append(buffer)
            buffer_addresses.append(ctypes.addressof(buffer))
        
        # Register all buffers at once
        result = adaptor.batch_register_memory(buffer_addresses, buffer_sizes)
        self.assertEqual(result, 0, "batch_register_memory failed for multiple buffers")
        
        # Verify buffers can be used for transfer
        target_addr = adaptor.get_first_buffer_address(self.target_server_name)
        test_data = b"Hello, batch register memory test!"
        data_len = len(test_data)
        
        # Write test data to first buffer
        result = adaptor.write_bytes_to_buffer(buffer_addresses[0], test_data, data_len)
        self.assertEqual(result, 0, "Failed to write test data")
        
        # Transfer data to target
        result = adaptor.transfer_sync_write(
            self.target_server_name, buffer_addresses[0], target_addr, data_len
        )
        self.assertEqual(result, 0, "Transfer failed after batch register")
        
        # Clean up
        result = adaptor.batch_unregister_memory(buffer_addresses)
        self.assertEqual(result, 0, "batch_unregister_memory failed")
        
        print("[✓] Multiple buffers batch_register_memory test passed")

    def test_batch_register_memory_with_data_verification(self):
        """Test batch_register_memory with data integrity verification."""
        import random
        import string

        def generate_random_data(size):
            chars = string.ascii_letters + string.digits
            return ''.join(random.choices(chars, k=size)).encode('utf-8')

        adaptor = self.adaptor
        num_buffers = 10
        buffer_size = 64 * 1024  # 64KB per buffer
        
        # Create and register buffers
        buffers = []
        buffer_addresses = []
        original_data = []
        
        for i in range(num_buffers):
            buffer = ctypes.create_string_buffer(buffer_size)
            buffers.append(buffer)
            buffer_addr = ctypes.addressof(buffer)
            buffer_addresses.append(buffer_addr)
            
            # Generate and store test data
            data = generate_random_data(buffer_size)
            original_data.append(data)
            
            # Write data to buffer
            result = adaptor.write_bytes_to_buffer(buffer_addr, data, buffer_size)
            self.assertEqual(result, 0, f"Failed to write data to buffer {i}")
        
        # Register all buffers
        capacities = [buffer_size] * num_buffers
        result = adaptor.batch_register_memory(buffer_addresses, capacities)
        self.assertEqual(result, 0, "batch_register_memory failed")
        
        # Get target buffer addresses
        target_base_addr = adaptor.get_first_buffer_address(self.target_server_name)
        target_addresses = [target_base_addr + i * buffer_size for i in range(num_buffers)]
        
        # Perform batch transfer to verify registered memory works
        result = adaptor.batch_transfer_sync_write(
            self.target_server_name, buffer_addresses, target_addresses, capacities
        )
        self.assertEqual(result, 0, "Batch transfer failed")
        
        # Clear local buffers and read back
        for addr in buffer_addresses:
            clear_data = bytes([0] * buffer_size)
            adaptor.write_bytes_to_buffer(addr, clear_data, buffer_size)
        
        result = adaptor.batch_transfer_sync_read(
            self.target_server_name, buffer_addresses, target_addresses, capacities
        )
        self.assertEqual(result, 0, "Batch read failed")
        
        # Verify data integrity
        for i in range(num_buffers):
            read_back = adaptor.read_bytes_from_buffer(buffer_addresses[i], buffer_size)
            self.assertEqual(read_back, original_data[i], f"Data mismatch in buffer {i}")
        
        # Clean up
        result = adaptor.batch_unregister_memory(buffer_addresses)
        self.assertEqual(result, 0, "batch_unregister_memory failed")
        
        print(f"[✓] Data verification test passed for {num_buffers} buffers")

    def test_batch_register_memory_edge_cases(self):
        """Test edge cases for batch_register_memory."""
        adaptor = self.adaptor
        
        # Test empty lists
        result = adaptor.batch_register_memory([], [])
        self.assertEqual(result, 0, "Empty lists should be handled gracefully")
        
        # Test single buffer with zero size
        buffer = ctypes.create_string_buffer(1024)
        buffer_addr = ctypes.addressof(buffer)
        
        result = adaptor.batch_register_memory([buffer_addr], [0])
        # This might fail or succeed depending on implementation
        
        print("[✓] Edge cases test completed")

    def test_batch_register_memory_stress(self):
        """Stress test batch_register_memory with many buffers."""
        adaptor = self.adaptor
        num_buffers = 100
        buffer_size = 4 * 1024  # 4KB per buffer
        
        # Create buffers
        buffers = []
        buffer_addresses = []
        for i in range(num_buffers):
            buffer = ctypes.create_string_buffer(buffer_size)
            buffers.append(buffer)
            buffer_addresses.append(ctypes.addressof(buffer))
        
        # Register all buffers in one batch
        capacities = [buffer_size] * num_buffers
        result = adaptor.batch_register_memory(buffer_addresses, capacities)
        self.assertEqual(result, 0, f"batch_register_memory failed for {num_buffers} buffers")
        
        # Unregister all buffers
        result = adaptor.batch_unregister_memory(buffer_addresses)
        self.assertEqual(result, 0, f"batch_unregister_memory failed for {num_buffers} buffers")
        
        print(f"[✓] Stress test passed for {num_buffers} buffers")

    def test_batch_register_memory_consistency(self):
        """Test consistency between batch and individual register/unregister."""
        adaptor = self.adaptor
        
        # Create buffers
        buffer_size = 1024 * 1024
        buffers = []
        buffer_addresses = []
        
        for i in range(5):
            buffer = ctypes.create_string_buffer(buffer_size)
            buffers.append(buffer)
            buffer_addresses.append(ctypes.addressof(buffer))
        
        # Register using batch API
        capacities = [buffer_size] * 5
        result = adaptor.batch_register_memory(buffer_addresses, capacities)
        self.assertEqual(result, 0, "Batch register failed")
        
        # Unregister individually
        for addr in buffer_addresses:
            result = adaptor.unregister_memory(addr)
            self.assertEqual(result, 0, f"Individual unregister failed for addr {addr}")
        
        # Register individually
        for addr in buffer_addresses:
            result = adaptor.register_memory(addr, buffer_size)
            self.assertEqual(result, 0, f"Individual register failed for addr {addr}")
        
        # Unregister using batch API
        result = adaptor.batch_unregister_memory(buffer_addresses)
        self.assertEqual(result, 0, "Batch unregister failed")
        
        print("[✓] Consistency test passed")

if __name__ == '__main__':
    unittest.main()
