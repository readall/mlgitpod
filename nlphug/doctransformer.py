from transformers import ReformerConfig, PyTorchBenchmark, PyTorchBenchmarkArguments

# config = ReformerConfig.from_pretrained("google/reformer-enwik8", 
#                                     lsh_attn_chunk_length=16386,
#                                     local_attn_chunk_length=16386,
#                                     lsh_num_chunks_before=0,
#                                     local_num_chunks_before=0
#                                     )

config = ReformerConfig.from_pretrained("google/reformer-enwik8")

benchmark_args = PyTorchBenchmarkArguments(sequence_lengths=[2048, 4096, 8192, 16386],
                                batch_sizes=[1],
                                models=["Reformer"]
                                )
benchmark = PyTorchBenchmark(configs=[config], args=benchmark_args)
result = benchmark.run()
print(result)
