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
        let mut x = na::SMatrix::<f64, NUM_STATES, 2>::zeros();
        let u = na::SMatrixSlice::<f64, NUM_CONROLS, NUM_STAGES>::from_slice(&u);
        x.set_column(0, &self.initial_state);
        for i in 0..NUM_STAGES {
            *c += self.cost.stage_cost(&x.column(0), &u.column(i));
            let (x0, mut x1) = x.columns_range_pair_mut(0, 1);
            self.dynamics.step(
                &na::SVectorSlice::<f64, NUM_STATES>::from(x0),
                &u.column(i),
                &mut x1,
            );
            x.swap_columns(0, 1);
        }
        *c += self.cost.terminal_cost(&x.column(0));
        Ok(())
    }

    fn grad(&self, u: &[f64], grad: &mut [f64]) -> FunctionCallResult {
        let u = na::SMatrixSlice::<f64, NUM_CONROLS, NUM_STAGES>::from_slice(&u);

        let x = self.rollout(u);

        let mut p = na::SVector::<f64, NUM_STATES>::zeros();
        let mut p_next = na::SVector::<f64, NUM_STATES>::zeros();

        self.cost
            .terminal_grad(&x.column(NUM_STAGES), &mut p_next.column_mut(0));

        let mut du = na::SVector::<f64, NUM_CONROLS>::zeros();
        let mut jac_x = na::SMatrix::<f64, NUM_STATES, NUM_STATES>::zeros();
        let mut jac_u = na::SMatrix::<f64, NUM_STATES, NUM_CONROLS>::zeros();
        let mut grad_x = na::SVector::<f64, NUM_STATES>::zeros();
        let mut grad_u = na::SVector::<f64, NUM_CONROLS>::zeros();

        for i in (0..NUM_STAGES).rev() {
            self.dynamics.jac(
                &x.column(i),
                &u.column(i),
                &x.column(i + 1),
                &mut jac_x.fixed_columns_mut(0),
                &mut jac_u.fixed_columns_mut(0),
            );
            self.cost.stage_grad(
                &x.column(i),
                &u.column(i),
                &mut grad_x.column_mut(0),
                &mut grad_u.column_mut(0),
            );

            p = jac_x.transpose() * p_next + grad_x;
            du = jac_u.transpose() * p + grad_u;
            let u_i = i * NUM_CONROLS;
            grad[u_i..u_i + NUM_CONROLS].copy_from_slice(du.as_slice());
            std::mem::swap(&mut p_next, &mut p);
        }
        Ok(())
    }

    pub fn rollout(
        &self,
        u: na::SMatrixSlice<f64, NUM_CONROLS, NUM_STAGES>,
    ) -> na::SMatrix<f64, NUM_STATES, NUM_STAGES_1> {
        let mut x = na::SMatrix::<f64, NUM_STATES, NUM_STAGES_1>::zeros();
        x.set_column(0, &self.initial_state);
        for i in 0..NUM_STAGES {
            let (x0, mut x1) = x.columns_range_pair_mut(i, i + 1);
            self.dynamics.step(
                &na::SVectorSlice::<f64, NUM_STATES>::from(x0),
                &u.column(i),
                &mut x1,
            );
        }
        x
    }

    pub fn solve(
        &self,
        u: &mut na::SMatrix<f64, NUM_CONROLS, NUM_STAGES>,
    ) -> Result<SolverStatus, SolverError> {
        let tolerance = 1e-14;
        let lbfgs_memory = 10;
        let max_iters = 1000;

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
