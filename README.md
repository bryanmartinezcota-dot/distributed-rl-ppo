# Distributed PPO with Ray 🚀

A highly scalable, custom implementation of Proximal Policy Optimization (PPO) designed for distributed compute clusters. This framework leverages [Ray](https://www.ray.io/) to decouple environment rollouts from policy optimization, allowing it to scale seamlessly to 1,024+ concurrent environments.

## Architecture Highlights
- **Asynchronous Rollouts:** Actors collect trajectories in parallel across CPU nodes.
- **Centralized Learner:** GPU-accelerated learner aggregates gradients and updates the global policy.
- **Efficient Memory Management:** Uses Ray's shared memory object store to minimize serialization overhead during trajectory transfer.

## Installation
```bash
git clone [https://github.com/BryanMartinez/distributed-rl-ppo.git](https://github.com/BryanMartinez/distributed-rl-ppo.git)
cd distributed-rl-ppo
pip install -r requirements.txt
