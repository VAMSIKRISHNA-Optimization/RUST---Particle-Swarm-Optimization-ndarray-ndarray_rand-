# RUST---Particle-Swarm-Optimization-ndarray-ndarray_rand-
ndarray_rand)Thinking This GitHub repository provides a high-performance implementation of the Particle Swarm Optimization (PSO) algorithm written in Rust. It leverages the ndarray ecosystem for efficient multidimensional array operations and ndarray-rand for high-speed stochastic sampling.
Features:
Vectorized Computation: Optimized for speed using ndarray to handle particle positions and velocities.
Customizable Fitness Functions: Easily define objective functions using Rust closures or traits.
Configurable Hyperparameters: Full control over inertia weight (\(w\)), cognitive constant (\(c_{1}\)), and social constant (\(c_{2}\)).
Stochastic Initialization: Utilizes ndarray-rand for uniform distribution of particles across the search space.

Tech Stack:
Rust (Core Logic)
ndarray (Linear Algebra & Array Manipulation)
ndarray-rand (Random Initialization)

Quick StartTo use this in your project, add the following to your Cargo.toml:
toml
[dependencies]
ndarray = "0.15"
ndarray-rand = "0.14"

# Add this repo as a dependency
pso_rust = { git = "github.com" }


Rust Particle Swarm Optimization (PSO)
A high-performance, memory-safe implementation of the Particle Swarm Optimization algorithm using the ndarray ecosystem. This project demonstrates how to leverage Rust's zero-cost abstractions and linear algebra libraries to solve multi-dimensional continuous optimization problems.

🚀 Key Implementation Features
Vectorized Computation: Uses ndarray for bulk operations on particle positions and velocities, ensuring high cache locality and performance.
Velocity Clamping: Implements a strict VMAX / VMIN range factor (customizable) to prevent particle divergence and "explosion."
Boundary Enforcement: Particles are strictly clamped within search space bounds (LB, UB) using azip! for efficient, element-wise position updates.
Reproducible Research: Uses StdRng with a fixed seed (45141) to ensure deterministic results across runs, which is critical for debugging and benchmarking.
Modular Fitness Functions: Supports easy swapping of objective functions (includes a pre-implemented Sphere Function and a template for the Rastrigin Function).

🛠 Tuning Parameters
The implementation exposes standard PSO hyperparameters for fine-tuning swarm behavior:
Parameter	Value	Description
W	0.4	Inertia Weight: Controls the impact of the previous velocity.
C1	1.0	Cognitive Coefficient: Pulls the particle toward its personal best.
C2	2.0	Social Coefficient: Pulls the particle toward the swarm's global best.
VRANGE	0.05	Velocity Range Factor: Dynamically calculates velocity limits based on search space scale.
