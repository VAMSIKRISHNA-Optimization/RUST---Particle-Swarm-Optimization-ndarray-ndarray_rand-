
use ndarray::iter::Iter;
/// 1. LIBRARIES and DEPENDENCIES
//1.1 Libraries for Matrix Operations
use ndarray::{Array1, Array2, ArrayView1};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::Rng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::rand::{SeedableRng, rngs::StdRng};
use ndarray::azip;
//1.2 Libraries for Random Numbers
// use rand::{Rng, SeedableRng};
// use rand::rngs::StdRng;





// // Initialize with a fixed seed
const SEED: u64 = 45141;

// // Basic Intialization parameters
const N: usize = 500; // Np: Population-Size
const T: u32   = 5000; // T : Number of iterations

// // Problem Specific paramters
const DIM: usize = 15;   // Number of dimensions
const LB: f64    = -5.12;  // Lower Bound of the Search Space
const UB: f64    = 5.12;   // Upper Bound of the Search Space

// // Tuning Parameters
const W : f64 = 0.4; // Inertia Weight
const C1: f64 = 1.0; // Cognitive Coefficient
const C2: f64 = 2.0; // Social Coefficient
const VRANGE: f64 = 0.05; // Velocity Range Factor ( 10% to 40% of the range of the position dimensions (Xmax - Xmin) )

const VMAX: f64 = VRANGE * (UB - LB);
const VMIN: f64 = -VMAX;




fn main() 
{
    // All Testing here
    
    // // Seeding
    let mut rng = StdRng::seed_from_u64(SEED);

    // // Population Initialization
    let mut population = population_initialization(N, DIM, LB, UB, &mut rng);

    // // Velocity Initialization
    let mut velocities = velocity_initialization(N, DIM, &mut rng);

    // // Fitness Evaluation of the Initial Population
    let mut fitness_values = fitness_evaluation_of_initial_population(N, &population, objective_function);

    // //  Personal best positions : Initalization
    let mut pbest_positions = population.clone();

    // // Personal best fitness : Initalization
    let mut pbest_fitness =  fitness_values.clone();

    // // Global best fitness : Initalization
    let (min_index, min_value) = fitness_values
                                .indexed_iter()
                                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                .unwrap();

    let mut gbest_fitness = *min_value;

    // // Global best position : Initalization
    let mut gbest_position = population.row(min_index.0).to_owned();

    ////  PSO Loop ////
    let mut Iteration: u32 = 0;

    'PSO_LOOP: loop
    {
        // Iterate through all the members of the population
        for pop_mem_ind in 0..population.nrows()
        //for pop_mem_ind in 0..2
        {
            // a. Update the velocity
            let r1: f64 = rng.r#gen();
            let r2: f64 = rng.r#gen();

            // println!("Iter: {:?}, r1 = {:?}, r2 = {:?}", Iteration, r1, r2);
            let new_velocity = ( W * &velocities.row(pop_mem_ind) ) 
                             + ( C1 * r1 * (&pbest_positions.row(pop_mem_ind) - &population.row(pop_mem_ind)) ) 
                             + ( C2 * r2 * (&gbest_position - &population.row(pop_mem_ind)) );
            
            velocities.row_mut(pop_mem_ind).assign(&new_velocity);

            // b. Apply Velocity Clamping
            velocities.row_mut(pop_mem_ind).mapv_inplace(|x| x.max(VMIN).min(VMAX));


            // c. Update the position
            let mut p_row = population.row_mut(pop_mem_ind);
            let v_row = velocities.row(pop_mem_ind);

            // d. Boundary Check
            azip!((p in &mut p_row, &v in &v_row) 
            {
                *p += v;
                *p = p.clamp(LB, UB); 
            });

            // e. Evaluate new fitness
            fitness_values[[pop_mem_ind,0]] = fitness_evaluation_of_individual_member(population.row(pop_mem_ind), objective_function);

            // f. Update personal best
            if fitness_values[[pop_mem_ind,0]] < pbest_fitness[[pop_mem_ind,0]] 
            {
                pbest_positions.row_mut(pop_mem_ind).assign(&population.row(pop_mem_ind));
                pbest_fitness[[pop_mem_ind,0]] = fitness_values[[pop_mem_ind,0]];
            }

            // g. Update global best
            if fitness_values[[pop_mem_ind,0]] < gbest_fitness
            {
                gbest_position = population.row(pop_mem_ind).to_owned();
                gbest_fitness  = fitness_values[[pop_mem_ind,0]];    
            }

        }


        println!("Iter#:{:?} Gbest: {:?}", Iteration+1, gbest_fitness);
        Iteration += 1;
        if Iteration >= T
        {
            break 'PSO_LOOP;
        }
        
    }

    println!("Final Gbest Fitness: {:?}", gbest_fitness);
    println!("Final Gbest Position: {:?}", gbest_position);

    

    // println!("{}, {}", x,y);
}


fn population_initialization(n: usize, d: usize, x_min: f64, x_max: f64, rng: &mut StdRng) -> Array2<f64>
{
    // Generate an N x D array with values in [x_min, x_max)
    let population: Array2<f64> = Array2::random_using((n , d), Uniform::new(x_min, x_max), rng);
    return population;
}

fn velocity_initialization(n: usize, d: usize, rng: &mut StdRng) -> Array2<f64>
{
    // let Vmax: f64 = VRANGE * (UB - LB);
    // let Vmin: f64 = -Vmax;

    // Generate an N x D array with values in [x_min, x_max)
    let velocities: Array2<f64> = Array2::random_using((n , d), Uniform::new(VMIN, VMAX), rng);
    return velocities;
}

fn fitness_evaluation_of_initial_population(n: usize, pop: &Array2<f64>, f: fn(ndarray::ArrayView1<f64>)->f64) -> Array2<f64>
{
    let mut vec_evaluated_fitness: Array2<f64> = Array2::zeros((n, 1));
    
    for (ind, row) in pop.rows().into_iter().enumerate() 
    {
        vec_evaluated_fitness[[ind, 0]] = f(row);
    }

    return vec_evaluated_fitness;

}

fn fitness_evaluation_of_individual_member(pop: ndarray::ArrayView1<f64>, f: fn(ndarray::ArrayView1<f64>)->f64) -> f64
{
   return f(pop);
}

fn objective_function(x: ArrayView1<f64>) -> f64
{
    // Rastrigin Function Implementation
    // let a: f64 = 10.0;
    // let d: usize = x.shape()[1];

    // let sum_of_squares = x.mapv(|xi| xi.powi(2) - a * (2.0 * std::f64::consts::PI * xi).cos());
    // let result = a * (d as f64) + sum_of_squares;

    // Sphere Function Implementation
    let result = x.mapv(|xi| xi.powi(2)).sum();

    return result;
}