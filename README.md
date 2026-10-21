# quantlab
LLM quantization measured on CPU: INT8 keeps fp32 quality (PPL 11.72 vs 11.70) at less than half the size; INT4 GPTQ 13.71 vs naive INT4 16.82. RTN, GPTQ and AWQ implemented from scratch. Qwen2.5-0.5B, no GPU.
