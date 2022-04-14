use optimization_engine::{constraints, panoc, FunctionCallResult, Optimizer, Problem};
use std::{fmt::Debug, ops::AddAssign};

pub trait Cost {
    fn stage_cost(&self, x: &[f64], u: &[f64]) -> f64;
    // TODO: Combine?
    // grad = grad + dcost/dx
    fn stage_grad_x(&self, x: &[f64], u: &[f64], grad: &mut [f64]);
    // grad = grad + dcost/du
    fn stage_grad_u(&self, x: &[f64], u: &[f64], grad: &mut [f64]);
    fn terminal_cost(&self, x: &[f64]) -> f64;
    fn terminal_grad(&self, x: &[f64], grad: &mut [f64]);
}

pub trait Dynamics {
    fn step(&self, x: &[f64], u: &[f64], x_next: &mut [f64]);
    // Apply the operation output = dStep/dx * input.
    fn apply_x_grad(&self, x: &[f64], u: &[f64], input: &[f64], output: &mut [f64]);
    // Apply the operation output = dStep/du * input.
    fn apply_u_grad(&self, x: &[f64], u: &[f64], input: &[f64], output: &mut [f64]);
}

#[derive(Debug)]
pub struct MPC<DynamicsType, CostType>
where
    DynamicsType: Dynamics,
    CostType: Cost,
{
    initial_state: Vec<f64>,
    num_controls: usize,
    num_stages: usize,
    dynamics: DynamicsType,
    cost: CostType,
}

pub fn add_to<T>(a: &mut [T], b: &[T])
where
    T: Copy + AddAssign,
{
    assert!(a.len() == b.len());
    a.iter_mut().zip(b.iter()).map(|(x, y)| (*x) += *y);
}

impl<DynamicsType, CostType> MPC<DynamicsType, CostType>
where
    DynamicsType: Dynamics,
    CostType: Cost,
{
    fn cost(&self, u: &[f64], c: &mut f64) -> FunctionCallResult {
        *c = 0.0;
        let mut x = self.initial_state.clone();
        let mut x_next = Vec::with_capacity(self.initial_state.len());
        for i in 0..self.num_stages {
            let u_i = i * self.num_controls;
            let u = &u[u_i..u_i + self.num_controls];
            *c += self.cost.stage_cost(&x, u);
            self.dynamics.step(&x, u, &mut x_next);
            x.swap_with_slice(&mut x_next);
        }
        *c += self.cost.terminal_cost(&x);
        Ok(())
    }

    fn grad(&self, u: &[f64], grad: &mut [f64]) -> FunctionCallResult {
        // TODO: Cache x allocation?
        let n = self.initial_state.len();
        let mut x = Vec::with_capacity(n * (self.num_stages + 1));
        x[0..n].clone_from_slice(&self.initial_state);
        for i in 0..self.num_stages {
            let x_i = i * n;
            let u_i = i * self.num_controls;
            let (x0, x1) = x[x_i..x_i + 2 * n].split_at_mut(x_i + n);
            self.dynamics
                .step(&x0, &u[u_i..u_i + self.num_controls], x1);
        }
        let x = x;

        let mut p_next = Vec::with_capacity(n);
        let mut p = Vec::with_capacity(n);
        self.cost
            .terminal_grad(&x[n * self.num_stages..n * (self.num_stages + 1)], &mut p);

        for i in (0..self.num_stages).rev() {
            let x_i = i * n;
            let u_i = i * self.num_controls;
            let x = &x[x_i..x_i + n];
            let u = &u[u_i..u_i + self.num_controls];
            let du = &mut grad[u_i..u_i + self.num_controls];

            p_next.swap_with_slice(&mut p);
            self.dynamics.apply_x_grad(x, u, &p_next, &mut p);
            self.cost.stage_grad_x(x, u, &mut p);
            self.dynamics.apply_u_grad(x, u, &p, du);
            self.cost.stage_grad_x(x, u, du);
        }
        Ok(())
    }

    pub fn solve(&self, u: &mut [f64]) {
        let tolerance = 1e-14;
        let lbfgs_memory = 10;
        let max_iters = 80;
        let mut u = [-1.5, 0.9];

        let problem = Problem::new(
            &constraints::NoConstraints {},
            |u: &[f64], grad: &mut [f64]| self.grad(u, grad),
            |u: &[f64], c: &mut f64| self.cost(u, c),
        );
        let mut panoc_cache =
            panoc::PANOCCache::new(self.num_controls * self.num_stages, tolerance, lbfgs_memory);
        let mut panoc =
            panoc::PANOCOptimizer::new(problem, &mut panoc_cache).with_max_iter(max_iters);

        // Invoke the solver
        let status = panoc.solve(&mut u);
        dbg!(status);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
