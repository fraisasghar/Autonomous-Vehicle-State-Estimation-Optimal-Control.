<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,100:0d1117&height=210&section=header&text=Autonomous%20Vehicle%20Control&fontSize=42&fontAlign=50&fontAlignY=36&animation=fadeIn&fontColor=e6edf3&desc=State%20Estimation%20%7C%20Sensor%20Fusion%20%7C%20Optimal%20Control&descAlign=50&descAlignY=60&descSize=17&descColor=8b95a8&shadow=true" alt="Header" />
</div>

<h3 align="center">A from-scratch MATLAB framework for autonomous vehicle state estimation and path tracking</h3>
<p align="center"><i>Nonlinear vehicle dynamics &nbsp;&middot;&nbsp; Extended Kalman Filter sensor fusion &nbsp;&middot;&nbsp; LQR optimal control &nbsp;&middot;&nbsp; built entirely in base MATLAB, no toolboxes</i></p>

<div align="center">

![MATLAB](https://img.shields.io/badge/Language-MATLAB%20(Base%20Only)-ff7a59?style=for-the-badge&labelColor=161b22)
![Estimator](https://img.shields.io/badge/Estimator-Extended%20Kalman%20Filter-e063ff?style=for-the-badge&labelColor=161b22)
![Control](https://img.shields.io/badge/Control-LQR%20%2B%20Feedforward-ffc93d?style=for-the-badge&labelColor=161b22)
![Validation](https://img.shields.io/badge/Validated-Monte%20Carlo%20%2B%20Baseline-5ce68c?style=for-the-badge&labelColor=161b22)
![Toolboxes](https://img.shields.io/badge/Toolboxes-Zero%20Required-8b95a8?style=for-the-badge&labelColor=161b22)

</div>

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0d1117,100:0d1117&height=2">

---

## Introduction

This project simulates the core pipeline used in real autonomous-driving stacks: a vehicle drives a closed circuit while estimating its own position from noisy, low-rate GPS and higher-rate IMU measurements, then tracks the planned path using an optimal LQR controller. Everything is built from first principles in **base MATLAB** -- no Control System, Optimization, or Robotics toolboxes anywhere in the code. The nonlinear dynamics, the Kalman filter's Jacobians, and the discrete-time LQR solve are all hand-written, and every run automatically produces a matching video, a multi-page PDF report, a PowerPoint showcase, and a machine-readable JSON results file.


<img src="assets/architecture.svg" alt="System architecture: sensors to EKF to LQR controller to vehicle dynamics, closing the loop" width="100%">

The estimate feeds back into the filter every step and the loop runs closed, in real time -- the controller only ever acts on what the EKF currently believes, exactly like a real system would.

## Core Components

Three pieces do all the work, each validated on its own before being wired into the closed loop. The bicycle model supplies the ground-truth physics the other two have to contend with. The Extended Kalman Filter turns noisy, multi-rate sensor readings into one trustworthy state estimate. The LQR controller turns that estimate into a steering command, never touching the ground truth directly. Each gets its own diagram below, in the order data actually flows through them.

<img src="assets/bicycle_model.svg" alt="Dynamic bicycle model: inputs, RK4 integration, state output" width="100%">

<img src="assets/ekf_flow.svg" alt="Extended Kalman Filter: predict, update from IMU, update from GPS" width="100%">

<img src="assets/lqr_flow.svg" alt="LQR path tracking: tracking error, cached Riccati gain, curvature feedforward, steering command" width="100%">



## Why Fusion Wins

The two traces below come from the same maneuver, plotted on the same axis. One is what raw GPS reports on its own; the other is what the EKF believes after combining it with the IMU. Same ground truth, two very different pictures of it.

<img src="assets/signal_comparison.svg" alt="Signal comparison: raw noisy GPS trace versus the smooth fused EKF estimate over the same maneuver" width="100%">

The gap isn't really about sensor quality, it's about update rate. GPS alone only knows where the car was up to 100 ms ago; the EKF fills that gap with IMU propagation in between and corrects it the moment a new fix lands, so the controller is never steering off a stale, blind guess.

<img src="assets/sensor_timing.svg" alt="Sensor timing: GPS fixes at 10Hz interleaved with IMU updates at 50Hz" width="100%">

## Testing & Verification:
A single good-looking run doesn't prove the estimator actually helps. It could be one lucky draw of sensor noise. Four independent checks close that gap: a raw-GPS baseline to measure what fusion actually buys, many randomized trials to confirm it isn't luck, a sweep across controller tuning, and a disturbance that's never turned off. None of them can fail silently and take the others down with it.

<img src="assets/validation_pipeline.svg" alt="Validation pipeline: no-EKF baseline, Monte Carlo trials, LQR sensitivity sweep, and the disturbance model feed into a validation report" width="100%">

- **No-EKF baseline:** Same controller, re-run headlessly on raw GPS alone, to measure exactly what the filter buys.
- **Monte Carlo trials:** Same comparison across many random noise seeds, so the result is a distribution, not one lucky run.
- **LQR sensitivity sweep:** Steering-effort weight swept across a wide range, tracing the classic accuracy-vs-effort curve.
- **Realistic disturbance:** A physical disturbance on the true vehicle every step, so validation reflects real imperfection.


> Every run's PDF, PPTX, and JSON export these exact numbers computed from *that* run -- they are never a static template. Because sensor noise and the physical disturbance are randomized every run, expect these to vary within a consistent range rather than match exactly.

## Getting Started

No install steps, no dependency list, no toolbox to license -- clone the repository, open av_simulation.m in MATLAB, and press Run. Everything below happens automatically: the live dashboard animates in real time, and once the run ends, the video, PDF report, PPTX showcase, and JSON results all export on their own.

<img src="assets/report_pipeline.svg" alt="Report generation pipeline: simulation end fans out to video, PDF, PPTX, and JSON exports" width="100%">





