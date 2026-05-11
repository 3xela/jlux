Renames pending before weight load:
- Modulation.linear → lin
- FluxSelfAttention.o → proj
- FluxMLP: restructure to match Sequential indexing
- LayerNorm: keep gamma/beta, remap at load time
- RMSNorm: keep gamma, remap at load time
- eps=1e-5 → 1e-6 everywhere

New modules (build with reference names from day 1):
- MLPEmbedder: in_layer, out_layer
- LastLayer: norm_final, linear, adaLN_modulation
- Flux: img_in, txt_in, time_in, vector_in, guidance_in, double_blocks, single_blocks, final_layer