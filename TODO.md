# General TODO list of the project

## VAE Module

### Theoretical aspects
- [ ] Gaussian implementation for more generality. Bernoulli is missing.
- [ ] Time-dependency: check if loading / loss calculation needs to change.
- [ ] Masking of input data (see original implementation).

### Code
- [ ] Refactor to create a VAE Torch class, train_vae should only wrap it with Lightning.
- [ ] Tests
