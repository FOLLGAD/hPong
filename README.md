# hPong (WIP)

![hPong Demo 2](./assets/pong_vae.png)

![hPong Demo](./assets/pong_simulation.png)

## Doing:

- [ ] implement **temporal** position embedding using rope

## Todo:

- [ ] Read up on rotary position embeddings in transformers, look at example implementations
- [ ] implement **spacial** position embedding using rope
- [ ] make vae training output images after each epoch
- [ ] Figure out why KL divergence of the DiT is so high (in the 9 orders of magnitude)
- [ ] guidance for user actions on DiT (STGuidance)

Done:

- [x] Split each frame into its own VAE encoding/decoding.
- [x] in the dataset, the "player" paddle should sometimes play like a good player for data quality

# References

- [RoPE: Rotary Position Embedding](https://arxiv.org/pdf/2104.09864v5)
- [Spatiotemporal Skip Guidance](https://arxiv.org/pdf/2411.18664)
- [Classifier-free Guidance](https://arxiv.org/pdf/2207.12598)
- [Axial Attention](https://arxiv.org/pdf/1912.12180)
