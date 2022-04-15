use nalgebra as na;

pub trait Dynamics<const NUM_STATES: usize, const NUM_CONROLS: usize> {
    fn step(
        &self,
        x: &na::SVector<f64, NUM_STATES>,
        u: &na::SVector<f64, NUM_CONROLS>,
        x_next: &mut na::SVector<f64, NUM_STATES>,
    );
    fn jac(
        &self,
        x: &na::SVector<f64, NUM_STATES>,
        u: &na::SVector<f64, NUM_CONROLS>,
        x_next: &na::SVector<f64, NUM_STATES>,
        jac_x: &mut na::SMatrix<f64, NUM_STATES, NUM_STATES>,
        jac_u: &mut na::SMatrix<f64, NUM_STATES, NUM_CONROLS>,
    );
}
