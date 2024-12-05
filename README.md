# hPong (WIP)

![hPong Demo 2](./assets/pong_vae.png)

![hPong Demo](./assets/pong_simulation.png)

## Doing:

VAE:

- [ ] make vae training output images after each epoch
- [x] Split each frame into its own VAE encoding/decoding.

DiT:

- [ ] Read up on rotary position embeddings in transformers, look at example implementations
  - [ ] implement **spacial** position embedding using rope
  - [ ] implement **temporal** position embedding using rope

## Todo:

- [ ] Figure out why KL divergence of the DiT is so high (in the 9 orders of magnitude)
- [ ] guidance for user actions on DiT (STGuidance)

# References

- [RoPE: Rotary Position Embedding](https://arxiv.org/pdf/2104.09864v5)
- [Spatiotemporal Skip Guidance](https://arxiv.org/pdf/2411.18664#page=9&zoom=100,84,296)
- [Axial Attention](https://arxiv.org/pdf/1912.12180)
