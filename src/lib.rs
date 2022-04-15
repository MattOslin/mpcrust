use nalgebra as na;
use optimization_engine::core::SolverStatus;
use optimization_engine::{
    constraints, panoc, FunctionCallResult, Optimizer, Problem, SolverError,
};
use std::fmt::Debug;

pub mod cost;
pub mod dynamics;
pub use crate::cost::Cost;
pub use crate::dynamics::Dynamics;

#[derive(Debug)]
pub struct MPC<
    const NUM_STATES: usize,
    const NUM_CONROLS: usize,
    const NUM_STAGES: usize,
    const NUM_STAGES_1: usize,
    DynamicsType,
    CostType,
> where
    DynamicsType: Dynamics<NUM_STATES, NUM_CONROLS>,
    CostType: Cost<NUM_STATES, NUM_CONROLS>,
{
    pub initial_state: na::SVector<f64, NUM_STATES>,
    pub dynamics: DynamicsType,
    pub cost: CostType,
}

// pub fn add_to<T>(a: &mut [T], b: &[T])
// where
//     T: Copy + AddAssign,
// {
//     assert!(a.len() == b.len());
//     a.iter_mut().zip(b.iter()).map(|(x, y)| (*x) += *y);
// }

// // c = A^T*b
// pub fn inv_matrix_multiply(A: &[f64], b: &[f64], c: &mut [f64]) {
//     assert!(b.len() * c.len() == A.len());
//     A.chunks(c.len())
//         .map(|col| col.iter().zip(b).map(|(x, y)| (*x) * (*y)).sum())
//         .collect_slice(c);
// }

impl<
        const NUM_STATES: usize,
        const NUM_CONROLS: usize,
        const NUM_STAGES: usize,
        const NUM_STAGES_1: usize,
        DynamicsType,
        CostType,
    > MPC<NUM_STATES, NUM_CONROLS, NUM_STAGES, NUM_STAGES_1, DynamicsType, CostType>
where
    DynamicsType: Dynamics<NUM_STATES, NUM_CONROLS>,
    CostType: Cost<NUM_STATES, NUM_CONROLS>,
{
    fn cost(&self, u: &[f64], c: &mut f64) -> FunctionCallResult {
        *c = 0.0;
        // TODO: preallocate?
        let mut x = self.initial_state.clone();
        let mut x_next = na::SVector::<f64, NUM_STATES>::zeros();
        for i in 0..NUM_STAGES {
            let u_i = i * NUM_CONROLS;
            let u = na::SVector::<f64, NUM_CONROLS>::from_column_slice(&u[u_i..u_i + NUM_CONROLS]);
            *c += self.cost.stage_cost(&x, &u);
            self.dynamics.step(&x, &u, &mut x_next);
            std::mem::swap(&mut x, &mut x_next);
        }
        *c += self.cost.terminal_cost(&x);
        Ok(())
    }

    fn grad(&self, u: &[f64], grad: &mut [f64]) -> FunctionCallResult {
        // TODO: Cache x allocation?
        let mut x = [na::SVector::<f64, NUM_STATES>::zeros(); NUM_STAGES_1];
        x[0] = self.initial_state;
        for i in 0..NUM_STAGES {
            let u_i = i * NUM_CONROLS;
            let u = na::SVector::<f64, NUM_CONROLS>::from_column_slice(&u[u_i..u_i + NUM_CONROLS]);
            let (x_0, x_1) = x[i..i + 2].split_at_mut(1);
            self.dynamics.step(&x_0[0], &u, &mut x_1[0]);
        }
        let x = x;

        let mut p = na::SVector::<f64, NUM_STATES>::zeros();
        let mut p_next = na::SVector::<f64, NUM_STATES>::zeros();

        self.cost.terminal_grad(&x[NUM_STAGES], &mut p_next);

        let mut du = na::SVector::<f64, NUM_CONROLS>::zeros();
        let mut jac_x = na::SMatrix::<f64, NUM_STATES, NUM_STATES>::zeros();
        let mut jac_u = na::SMatrix::<f64, NUM_STATES, NUM_CONROLS>::zeros();
        let mut grad_x = na::SVector::<f64, NUM_STATES>::zeros();
        let mut grad_u = na::SVector::<f64, NUM_CONROLS>::zeros();

        for i in (0..NUM_STAGES).rev() {
            let u_i = i * NUM_CONROLS;
            let u = na::SVector::<f64, NUM_CONROLS>::from_column_slice(&u[u_i..u_i + NUM_CONROLS]);

            self.dynamics
                .jac(&x[i], &u, &x[i + 1], &mut jac_x, &mut jac_u);
            self.cost.stage_grad(&x[i], &u, &mut grad_x, &mut grad_u);

            p = jac_x.transpose() * p_next + grad_x;
            du = jac_u.transpose() * p + grad_u;
            grad[u_i..u_i + NUM_CONROLS].copy_from_slice(du.as_slice());
            std::mem::swap(&mut p_next, &mut p);
        }
        Ok(())
    }

    pub fn solve(
        &self,
        u: &mut na::SMatrix<f64, NUM_CONROLS, NUM_STAGES>,
    ) -> Result<SolverStatus, SolverError> {
        let tolerance = 1e-14;
        let lbfgs_memory = 10;
        let max_iters = 80;

        let problem = Problem::new(
            &constraints::NoConstraints {},
            |u: &[f64], grad: &mut [f64]| self.grad(u, grad),
            |u: &[f64], c: &mut f64| self.cost(u, c),
        );
        let mut panoc_cache =
            panoc::PANOCCache::new(NUM_CONROLS * NUM_STAGES, tolerance, lbfgs_memory);
        let mut panoc =
            panoc::PANOCOptimizer::new(problem, &mut panoc_cache).with_max_iter(max_iters);

        // Invoke the solver
        panoc.solve(&mut u.as_mut_slice())
    }
}
