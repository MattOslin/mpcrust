use nalgebra as na;

pub trait Dynamics<const NUM_STATES: usize, const NUM_CONROLS: usize> {
    fn step(
        &self,
        x: &na::SVectorSlice<f64, NUM_STATES>,
        u: &na::SVectorSlice<f64, NUM_CONROLS>,
        x_next: &mut na::SVectorSliceMut<f64, NUM_STATES>,
    );
    fn jac(
        &self,
        x: &na::SVectorSlice<f64, NUM_STATES>,
        u: &na::SVectorSlice<f64, NUM_CONROLS>,
        x_next: &na::SVectorSlice<f64, NUM_STATES>,
        jac_x: &mut na::SMatrixSliceMut<f64, NUM_STATES, NUM_STATES>,
        jac_u: &mut na::SMatrixSliceMut<f64, NUM_STATES, NUM_CONROLS>,
    );
}
