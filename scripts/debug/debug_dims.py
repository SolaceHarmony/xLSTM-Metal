
"""Debug dimensions issue"""

# Simulate the config values that cause the dimension mismatch
class MockConfig:
    def __init__(self):
        self.embedding_dim = 128
        self.num_heads = 2
        self.v_dim_factor = 1.0
        self.qk_dim_factor = 0.5

config = MockConfig()

# Calculate dimensions
v_dim = int(config.embedding_dim * config.v_dim_factor)
qk_dim = int(config.embedding_dim * config.qk_dim_factor)

print(f"embedding_dim: {config.embedding_dim}")
print(f"v_dim_factor: {config.v_dim_factor}")
print(f"qk_dim_factor: {config.qk_dim_factor}")
print(f"v_dim: {v_dim}")
print(f"qk_dim: {qk_dim}")
print(f"num_heads: {config.num_heads}")
print(f"v_head_dim: {v_dim // config.num_heads}")
print(f"qk_head_dim: {qk_dim // config.num_heads}")

# The issue: v and qk have different head dimensions!
# v has shape [B, NH, S, v_dim/NH] = [B, 2, S, 64]
# q, k have shape [B, NH, S, qk_dim/NH] = [B, 2, S, 32]