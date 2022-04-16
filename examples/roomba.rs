use mpcrust::{Cost, Dynamics, MPC};
use nalgebra as na;
use plotly::{Plot, Scatter};

// x = [x, y, theta]
// u = [v, w]
// x1 = [x + v * cos(theta), y + v * sin(theta), theta + w]
// cost = 0.5*a*y^2 + 0.5*b*theta^2 + 0.5*c*v^2 + 0.5*d*w^2
// terminal_cost = 0.5*a*y^2 + 0.5*b*theta^2 + 0.5*e*(x-x^star)^2

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
const NUM_CONTROLS: usize = 2;
type State<'a> = na::SVectorSlice<'a, f64, NUM_STATES>;
type Control<'a> = na::SVectorSlice<'a, f64, NUM_CONTROLS>;
type StateMut<'a> = na::SVectorSliceMut<'a, f64, NUM_STATES>;
type ControlMut<'a> = na::SVectorSliceMut<'a, f64, NUM_CONTROLS>;

impl Cost<NUM_STATES, NUM_CONTROLS> for RoombaCost {
    fn stage_cost(&self, x: &State, u: &Control) -> f64 {
        0.5 * (self.a * x[1] * x[1]
            + self.b * x[2] * x[2]
            + self.c * u[0] * u[0]
            + self.d * u[1] * u[1])
    }
    fn stage_grad(&self, x: &State, u: &Control, grad_x: &mut StateMut, grad_u: &mut ControlMut) {
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
    fn terminal_grad(&self, x: &State, grad: &mut StateMut) {
        grad[0] = self.e * (x[0] - self.x_star);
        grad[1] = self.a * x[1];
        grad[2] = self.b * x[2];
    }
}

struct RoombaDynamics {}

impl Dynamics<NUM_STATES, NUM_CONTROLS> for RoombaDynamics {
    fn step(&self, x: &State, u: &Control, x_next: &mut StateMut) {
        x_next[0] = x[0] + u[0] * x[2].cos();
        x_next[1] = x[1] + u[0] * x[2].sin();
        x_next[2] = x[2] + u[1];
    }
    fn jac(
        &self,
        x: &State,
        u: &Control,
        _x_next: &State,
        jac_x: &mut na::SMatrixSliceMut<f64, NUM_STATES, NUM_STATES>,
        jac_u: &mut na::SMatrixSliceMut<f64, NUM_STATES, NUM_CONTROLS>,
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

fn main() {
    let problem = MPC::<NUM_STATES, NUM_CONTROLS, 100, 101, RoombaDynamics, RoombaCost> {
        initial_state: na::SVector::<f64, NUM_STATES>::new(0., 1., 0.),
        dynamics: RoombaDynamics {},
        cost: RoombaCost::new(10.),
    };
    let mut u = na::SMatrix::<f64, NUM_CONTROLS, 100>::zeros();
    dbg!(problem.solve(&mut u));
    dbg!(u);

    let x = problem.rollout(u.fixed_columns(0));

    let mut xs = Vec::with_capacity(101);
    for x in &x.row(0) {
        xs.push(*x);
    }

    let mut ys = Vec::with_capacity(101);
    for y in &x.row(1) {
        ys.push(*y);
    }

    let trace = Scatter::new(xs, ys).name("path");

    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.to_html("/home/moslin/plots/plot.html");
}
