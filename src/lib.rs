use optimization_engine::{constraints, panoc, FunctionCallResult, Optimizer, Problem};

#[derive(Debug)]
pub struct Cost<GradientType, CostType>
where
    GradientType: Fn(&[f64], &mut [f64]),
    CostType: Fn(&[f64], &mut f64),
{
    stage_cost: CostType,
    stage_cost_gradient: GradientType,
    terminal_cost: CostType,
    terminal_cost_gradient: GradientType,
}

#[derive(Debug)]
pub struct MPC<CostType, DynamicsType>
where
    DynamicsType: Fn(&[f64], &mut [f64]),
{
    num_states: usize,
    num_controls: usize,
    num_stages: usize,
    dynamics: DynamicsType,
    cost: CostType,
}

impl<CostType, DynamicsType> MPC<CostType, DynamicsType>
where
    DynamicsType: Fn(&[f64], &mut [f64]),
{
    fn cost(&self, u: &[f64], c: &mut f64) -> FunctionCallResult {
        // TODO: single shot.
        Ok(())
    }

    fn grad(&self, u: &[f64], grad: &mut [f64]) -> FunctionCallResult {
        // TODO: rollout followed by backprop.
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
