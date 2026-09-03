import os
import time
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

from mooncake.store import MooncakeDistributedStore

DEFAULT_DEFAULT_KV_LEASE_TTL = 5000  # 5000 milliseconds
default_kv_lease_ttl = int(
    os.getenv("DEFAULT_KV_LEASE_TTL", DEFAULT_DEFAULT_KV_LEASE_TTL)
)


def cuda_available():
    return torch is not None and torch.cuda.is_available()


def get_dummy_client(store):
    mem_pool_size = 3200 * 1024 * 1024  # 3200 MB
    local_buffer_size = 512 * 1024 * 1024  # 512 MB
    real_client_address = "127.0.0.1:50052"
    retcode = store.setup_dummy(mem_pool_size, local_buffer_size, real_client_address)
    if retcode:
        raise RuntimeError(
            f"Failed to setup dummy store client. Return code: {retcode}"
        )


class TestPutTensorFrom(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch is None:
            raise unittest.SkipTest("PyTorch is not available")
        cls.store = MooncakeDistributedStore()
        get_dummy_client(cls.store)

    def _cleanup(self, keys):
        time.sleep(default_kv_lease_ttl / 1000)
        for key in keys:
            self.store.remove(key)

    def test_put_tensor_from_uses_tensor_data_ptr(self):
        prefix = f"test_put_tensor_from_obj_{os.getpid()}"
        tensor = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        key = f"{prefix}_single"
        self.assertEqual(self.store.put_tensor_from(key, tensor), 0)
        retrieved = self.store.get_tensor(key)
        self.assertIsNotNone(retrieved)
        self.assertTrue(torch.equal(retrieved, tensor))

        keys = [f"{prefix}_batch_{i}" for i in range(2)]
        tensors = [tensor + 1, tensor.reshape(4, 6).to(torch.int64)]
        self.assertEqual(list(self.store.batch_put_tensor_from(keys, tensors)), [0, 0])
        for stored_key, expected in zip(keys, tensors):
            actual = self.store.get_tensor(stored_key)
            self.assertIsNotNone(actual)
            self.assertTrue(torch.equal(actual, expected))

        self._cleanup([key, *keys])

    @unittest.skipUnless(cuda_available(), "CUDA is not available")
    def test_put_tensor_from_cuda_tensor_data_ptr(self):
        prefix = f"test_put_tensor_from_cuda_{os.getpid()}"
        tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        cuda_tensor = tensor.to("cuda")
        torch.cuda.synchronize()

        key = f"{prefix}_single"
        self.assertEqual(self.store.put_tensor_from(key, cuda_tensor), 0)
        retrieved = self.store.get_tensor(key)
        self.assertIsNotNone(retrieved)
        self.assertTrue(torch.equal(retrieved, tensor))

        batch_keys = [f"{prefix}_batch_{i}" for i in range(2)]
        batch_cpu = [tensor + 1, (tensor + 2).to(torch.int64)]
        batch_cuda = [item.to("cuda") for item in batch_cpu]
        torch.cuda.synchronize()
        self.assertEqual(
            list(self.store.batch_put_tensor_from(batch_keys, batch_cuda)),
            [0, 0],
        )
        for stored_key, expected in zip(batch_keys, batch_cpu):
            actual = self.store.get_tensor(stored_key)
            self.assertIsNotNone(actual)
            self.assertTrue(torch.equal(actual, expected))

        self._cleanup([key, *batch_keys])


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
