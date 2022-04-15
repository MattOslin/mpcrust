#[macro_use(assert_ulps_eq)]
extern crate approx;
use nalgebra as na;
use optimization_engine::{constraints, panoc, FunctionCallResult, Optimizer, Problem};
use std::fmt::Debug;

mod cost;
mod dynamics;
use crate::cost::Cost;
use crate::dynamics::Dynamics;

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
    initial_state: na::SVector<f64, NUM_STATES>,
    dynamics: DynamicsType,
    cost: CostType,
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
            let (x_0, x_1) = x[i..i + 1].split_at_mut(1);
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

    pub fn solve(&self) -> na::SMatrix<f64, NUM_CONROLS, NUM_STAGES> {
        let tolerance = 1e-14;
        let lbfgs_memory = 10;
        let max_iters = 80;
        let mut u = na::SMatrix::<f64, NUM_CONROLS, NUM_STAGES>::zeros();

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
        let status = panoc.solve(&mut u.as_mut_slice());
        dbg!(status);
        u
    }
}

#[cfg(test)]
mod tests {

    // x = [x, y, theta]
    // u = [v, w]
    // x1 = [x + v * cos(theta), y + v * sin(theta), theta + w]
    // cost = 0.5*a*y^2 + 0.5*b*theta^2 + 0.5*c*v^2 + 0.5*d*w^2
    // terminal_cost = 0.5*a*y^2 + 0.5*b*theta^2 + 0.5*e*(x-x^star)^2

    use super::*;
    struct RoombaCost {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        e: f64,
        x_star: f64,
    }

    impl RoombaCost {
        pub fn new(x_star: f64) -> Self {
            RoombaCost {
                a: 1.,
                b: 1.,
                c: 1.,
                d: 1.,
                e: 1.,
                x_star,
            }
        }
    }

    const NUM_STATES: usize = 3;
    const NUM_CONROLS: usize = 2;
    type State = na::SVector<f64, NUM_STATES>;
    type Control = na::SVector<f64, NUM_CONROLS>;

    impl Cost<NUM_STATES, NUM_CONROLS> for RoombaCost {
        fn stage_cost(&self, x: &State, u: &Control) -> f64 {
            0.5 * (self.a * x[1] * x[1]
                + self.b * x[2] * x[2]
                + self.c * u[0] * u[0]
                + self.d * u[1] * u[1])
        }
        fn stage_grad(&self, x: &State, u: &Control, grad_x: &mut State, grad_u: &mut Control) {
            grad_x[1] = self.a * x[1];
            grad_x[2] = self.b * x[2];

            grad_u[0] = self.c * u[0];
            grad_u[1] = self.d * u[1];
        }
        fn terminal_cost(&self, x: &State) -> f64 {
            0.5 * (self.a * x[1] * x[1]
                + self.b * x[2] * x[2]
                + self.e * (x[0] - self.x_star) * (x[0] - self.x_star))
        }
        fn terminal_grad(&self, x: &State, grad: &mut State) {
            grad[0] = self.e * (x[0] - self.x_star);
            grad[1] = self.a * x[1];
            grad[2] = self.b * x[2];
        }
    }

    struct RoombaDynamics {}

    impl Dynamics<NUM_STATES, NUM_CONROLS> for RoombaDynamics {
        fn step(&self, x: &State, u: &Control, x_next: &mut State) {
            x_next[0] = x[0] + u[0] * x[2].cos();
            x_next[1] = x[1] + u[0] * x[2].sin();
            x_next[2] = x[2] + u[1];
        }
        fn jac(
            &self,
            x: &State,
            u: &Control,
            _x_next: &State,
            jac_x: &mut na::SMatrix<f64, NUM_STATES, NUM_STATES>,
            jac_u: &mut na::SMatrix<f64, NUM_STATES, NUM_CONROLS>,
        ) {
            let s = x[2].sin();
            let c = x[2].cos();

            jac_x[(0, 0)] = 1.;
            jac_x[(0, 1)] = 0.;
            jac_x[(0, 2)] = -u[0] * s;
            jac_x[(1, 0)] = 0.;
            jac_x[(1, 1)] = 1.;
            jac_x[(1, 2)] = u[0] * c;
            jac_x[(2, 0)] = 0.;
            jac_x[(2, 1)] = 0.;
            jac_x[(2, 2)] = 1.;

            jac_u[(0, 0)] = c;
            jac_u[(0, 1)] = 0.;
            jac_u[(1, 0)] = s;
            jac_u[(1, 1)] = 0.;
            jac_u[(2, 0)] = 0.;
            jac_u[(2, 1)] = 1.;
        }
    }

    fn make_mpc() -> MPC<NUM_STATES, NUM_CONROLS, 100, 101, RoombaDynamics, RoombaCost> {
        MPC {
            initial_state: State::zeros(),
            dynamics: RoombaDynamics {},
            cost: RoombaCost::new(10.),
        }
    }

    #[test]
    fn starting_cost() {
        let problem = make_mpc();
        let mut cost = 0.;
        assert_eq!(problem.cost(&[0.; NUM_CONROLS * 100], &mut cost), Ok(()));
        assert_ulps_eq!(cost, 50.);
    }
}
