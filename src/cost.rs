use nalgebra as na;

pub trait Cost<const NUM_STATES: usize, const NUM_CONROLS: usize> {
    fn stage_cost(
        &self,
        x: &na::SVector<f64, NUM_STATES>,
        u: &na::SVector<f64, NUM_CONROLS>,
    ) -> f64;
    fn stage_grad(
        &self,
        x: &na::SVector<f64, NUM_STATES>,
        u: &na::SVector<f64, NUM_CONROLS>,
        grad_x: &mut na::SVector<f64, NUM_STATES>,
        grad_u: &mut na::SVector<f64, NUM_CONROLS>,
    );
    fn terminal_cost(&self, x: &na::SVector<f64, NUM_STATES>) -> f64;
    fn terminal_grad(
        &self,
        x: &na::SVector<f64, NUM_STATES>,
        grad: &mut na::SVector<f64, NUM_STATES>,
    );
}
