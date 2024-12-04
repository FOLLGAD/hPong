# hPong (WIP)

![hPong Demo 2](./assets/pong_vae.png)

![hPong Demo](./assets/pong_simulation.png)

## Doing:

- [ ] Split each frame into its own VAE encoding/decoding.
      Right now, the VAE is encoding/decoding 3 frames at once, which is not flexible.

## Todo:

- [ ] Figure out why KL divergence of the DiT is so high (in the 9 orders of magnitude)
- [ ] rotary position embedding
  - [ ] implement **spacial** position embedding
  - [ ] implement **temporal** position embedding
- [ ] classifier-free guidance for user actions on DiT
