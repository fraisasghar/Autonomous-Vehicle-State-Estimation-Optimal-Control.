%% =======================================================================
%  AUTONOMOUS VEHICLE STATE ESTIMATION & OPTIMAL CONTROL SIMULATION
% =========================================================================
%
%  A complete, from-scratch simulation of the core pipeline used in real
%  autonomous-driving stacks, built using only base MATLAB (no toolboxes):
%
%    1. VEHICLE DYNAMICS   - nonlinear dynamic bicycle model with linear
%                             tire forces (Cf, Cr cornering stiffness),
%                             integrated with 4th-order Runge-Kutta.
%
%    2. SENSORS            - multi-rate noisy measurements:
%                               GPS position  @ 10 Hz  (noisy, slow)
%                               IMU yaw-rate + accel @ 50 Hz (noisy, fast)
%
%    3. STATE ESTIMATION   - an Extended Kalman Filter (EKF) with
%                             numerically-computed Jacobians fuses the
%                             noisy, multi-rate sensors into a smooth,
%                             accurate 6-state estimate.
%
%    4. PATH REPRESENTATION- the track is expressed in Frenet-like error
%                             coordinates (cross-track error e1, its rate
%                             e2, heading error e3, its rate e4) relative
%                             to the nearest point on the reference path.
%
%    5. OPTIMAL CONTROL    - a Linear-Quadratic Regulator (LQR) computed
%                             via a hand-written discrete-time Riccati
%                             iteration (no dlqr / Control Toolbox call)
%                             drives the error state to zero, combined
%                             with a steady-state curvature feedforward
%                             term from classical vehicle dynamics theory.
%
%    6. SPEED PLANNING     - target speed is limited by a maximum lateral
%                             acceleration constraint given local curvature.
%
%    7. PREDICTIVE ROLLOUT - at every frame, the SAME nonlinear model and
%                             controller are rolled forward from the
%                             CURRENT ESTIMATE (not the true state, since
%                             a real controller never has the truth) to
%                             show the predicted short-horizon trajectory.
%
%    8. DASHBOARD           - a live multi-panel animation: track view,
%                             speed profile, steering decomposition, and
%                             estimation-error tracking.
%
%  Run this file directly in MATLAB (F5). Close the figure to stop early.
%
%  REQUIREMENTS: base MATLAB only. No Control System, Optimization, or
%  Robotics toolboxes are used anywhere in this file -- the LQR solver,
%  Jacobians, and estimator are all implemented from first principles.
%
%  WHAT THIS SCRIPT PRODUCES (four figures + console output):
%    1. Live animated dashboard   - track view, speed, steering, error
%    2. Static diagnostics figure - tire slip angles, lateral accel,
%                                   EKF covariance convergence
%    3. Monte Carlo bar chart     - EKF vs. raw-GPS baseline, N trials
%    4. LQR sensitivity curve     - tracking error vs. steering-effort
%                                   weight R (classic tuning trade-off)
%
%  ALREADY BUILT ON TOP OF THE CORE PIPELINE ABOVE (toggle in Section 1):
%    - Pacejka "magic formula" tire model as an alternative to the linear
%      tire law, for studying behavior near the friction limit (set
%      tireModelType = 'pacejka').
%    - A parallel Unscented Kalman Filter that runs alongside the EKF
%      purely for an end-of-run accuracy comparison (estimatorCompare).
%    - A second ("ghost") vehicle driving the same track with a live
%      predicted-collision check, reusing the same predictive-rollout
%      mechanism used for the primary car's on-screen preview.
%    - A finite-horizon linear MPC alternative to the analytic LQR +
%      feedforward controller, solved by projected gradient descent with
%      hard steering limits (controllerType = 'mpc').
%    - A no-EKF baseline comparison and Monte Carlo trials across
%      randomized sensor-noise seeds, both driven by the shared, headless
%      runTrial() used nowhere else but these validation sections.
%    - Auto-generated PDF and PowerPoint reports, built with plain file
%      I/O (no Report Generator toolbox, no COM automation), reflecting
%      each run's own numbers rather than a static template.
%
%  IDEAS FOR FURTHER EXTENDING THIS SCRIPT:
%    - Feed the estimated state into a road-friction or heat-generation
%      model at each tire to connect this back to a thermal/mechanical
%      analysis (tying together vehicle dynamics and heat transfer).
%    - Replace the fixed Q/R LQR weights with a gain-scheduled or
%      learned policy and compare tracking performance against the
%      classical controller built here.
%    - Extend the Monte Carlo study to sweep sensor noise MAGNITUDE
%      (not just the random seed) to map out a performance-vs-noise
%      curve rather than a single-noise-level distribution.
% =========================================================================

clear; clc; close all;
rng(7);   % reproducible sensor noise

% Exported files (video/report) are saved in the SAME folder as this
% script file, regardless of whatever MATLAB's "Current Folder" happens
% to be set to -- much more predictable than relying on pwd. Falls back
% to pwd only if this is run in an odd context where mfilename is empty
% (e.g. pasted directly into the Command Window rather than run as a file).
scriptOutDir = fileparts(mfilename('fullpath'));
if isempty(scriptOutDir)
    scriptOutDir = pwd;
end

%% ======================= SECTION 1: PARAMETERS =========================

% --- Vehicle physical parameters (typical mid-size sedan) ---------------
veh.m   = 1600;        % mass                       [kg]
veh.Iz  = 2500;        % yaw moment of inertia       [kg m^2]
veh.lf  = 1.2;         % CG to front axle distance   [m]
veh.lr  = 1.6;         % CG to rear axle distance    [m]
veh.Cf  = 90000;       % TOTAL front axle cornering stiffness   [N/rad]
veh.Cr  = 110000;      % TOTAL rear axle cornering stiffness    [N/rad]
veh.L   = veh.lf + veh.lr;

% Understeer gradient (classical steady-state cornering formula)
veh.Kus = veh.m*veh.lr/(veh.Cf*veh.L) - veh.m*veh.lf/(veh.Cr*veh.L);

% Tire model selection (see vehicleDynamicsCT for details) and friction
% scale (1.0 = full grip; scaled down inside low-grip zones if enabled)
veh.tireModel = 'linear';   % overwritten below if tireModelType='pacejka'
veh.muScale   = 1.0;        % default full grip; overridden per-step for zones

% Pacejka "Magic Formula" coefficients, chosen so the small-slip-angle
% behavior matches the existing linear Cf/Cr (consistent handling near
% zero slip; the nonlinearity only becomes visible as slip grows):
%   Fy = D*sin(C*atan(B*alpha - E*(B*alpha - atan(B*alpha))))
% with slope at alpha=0 equal to B*C*D, so D = Cf/(B*C) matches Cf exactly.
veh.pacB = 10; veh.pacC = 1.9; veh.pacE = 0.97;
veh.pacDf = veh.Cf/(veh.pacB*veh.pacC);
veh.pacDr = veh.Cr/(veh.pacB*veh.pacC);

% --- Actuator / planning limits ------------------------------------------
lims.maxSteer = deg2rad(32);
lims.maxSteerRate = deg2rad(200);  % [rad/s] limits how fast steering can change --
                                    % catches any transient spikes (e.g. from a
                                    % sudden curvature change) before they turn
                                    % into a jerky, unrealistic command
lims.maxAccel = 2.5;          % [m/s^2]
lims.maxDecel = 5.0;          % [m/s^2]
lims.vmin     = 5.0;          % [m/s]
lims.vmax     = 14.0;         % [m/s]
lims.ayMax    = 3.6;          % lateral accel limit for speed planning [m/s^2]

% --- Simulation timing ----------------------------------------------------
dt        = 0.02;      % main integration / control step  [s]  (50 Hz)
tEnd      = 70;         % target simulated time             [s] -- the sim
                        % will run a little PAST this if needed to finish
                        % whichever lap is already in progress at that
                        % point (see maxLapFinishExtension below), rather
                        % than cutting off mid-lap.
maxLapFinishExtension = 30;   % [s] hard safety cap on how much PAST tEnd
                        % the sim is allowed to run while finishing the
                        % current lap -- generous relative to a typical
                        % ~15s lap, but bounded so a stuck car (or an
                        % unusually long lap from a slow-driving disturbance)
                        % can't run away indefinitely.
nSteps    = round((tEnd + maxLapFinishExtension)/dt);   % preallocation
                        % size only (with safety margin) -- the actual
                        % loop below stops well before this in the normal
                        % case; unused preallocated slots stay NaN and are
                        % already filtered out by the plotting/report code.
nStepsNominal = round(tEnd/dt);   % the ORIGINAL, un-padded step count --
                        % used anywhere we want "progress toward the
                        % intended 70s", like console progress printouts,
                        % as opposed to nSteps' generous safety margin.

gpsDecim  = 5;          % GPS update every 5 steps  -> 10 Hz
imuDecim  = 1;          % IMU update every step     -> 50 Hz

% --- Real-time playback pacing ---------------------------------------------
% The physics still integrate at fixed dt (accuracy is unaffected), but
% RENDERING is now paced against the wall clock instead of a fixed step
% count. This is what actually fixes the laggy/uneven animation: previously
% every single physics step also re-solved a Riccati equation AND ran the
% full predictive rollout (itself another 25 Riccati solves), whether or
% not that step was ever drawn. That work is now done only on frames that
% actually render, and rendering itself is throttled to a real-time frame
% rate rather than "as fast as the CPU can churn through steps."
playbackSpeed   = 1.5;    % 1.0 = real-time, 2.0 = 2x speed, 0.5 = slow-mo
                          % Bumped from 1.0 -- the real-time throttle only
                          % ever pauses when the sim is running AHEAD of
                          % this target, so on any machine with headroom
                          % this directly makes the live view noticeably
                          % snappier; it does nothing extra on a machine
                          % that's already compute/render-bound (there's
                          % no pause to remove in that case regardless).
liveTargetFPS   = 2;      % on-screen rendering rate while WATCHING live
                          % -- lowered from 4 as a gentler STARTING guess,
                          % before the adaptive tuning below has measured
                          % this machine's real render cost; it
                          % self-corrects within a few frames either way
                          % (see liveStepsPerFrame). This is now the ONLY
                          % render-cadence setting in the script -- see
                          % TIMING FIX #10 near the main loop for why
                          % exportVideo no longer needs (or has) its own
                          % separate, slower cadence: video frames are
                          % captured as a byproduct of this same adaptive
                          % live rendering, at no extra cost, and encoded
                          % to disk only after the run finishes.
lqrGainCacheTol = 0.25;   % [m/s] re-solve LQR gain only if speed drifts this much

% --- Predictive rollout (preview) settings --------------------------------
Npred   = 25;
dtPred  = 0.10;

% --- Sensor noise standard deviations -------------------------------------
noise.gpsPos   = 0.6;      % [m]           GPS position noise
noise.imuYaw   = 0.02;     % [rad/s]       gyro noise
noise.imuAccel = 0.15;     % [m/s^2]       accelerometer noise

% --- TRUE-VEHICLE process disturbance --------------------------------------
% Small, physically-motivated noise applied to what the actuator ACTUALLY
% delivers to the TRUE vehicle each step -- standing in for real-world
% effects no controller can perfectly cancel (crosswind gusts, road-surface
% irregularities, tiny actuator imperfections). This is the fix for laps
% (and every plotted trace -- speed, steering, cross-track error) coming
% out bit-for-bit identical every time around: previously the TRUE plant
% had ZERO disturbance of its own, only the SENSORS were noisy, so a
% deterministic LQR controller tracking a perfectly deterministic
% reference around a perfectly repeating track produced an essentially
% identical trajectory on every single lap -- there was nothing left to
% vary. Real driving, even fully automated, is never that clean. These
% are deliberately small (won't destabilize tracking or look chaotic),
% but a real, physical process disturbance the estimator has to contend
% with -- not just noisier sensors reporting a still-perfect vehicle.
procNoise.steer = 0.006;   % [rad]    steering actuator disturbance (~0.34 deg)
procNoise.accel = 0.08;    % [m/s^2]  longitudinal disturbance (crosswind, surface)

% --- EKF process / measurement covariances --------------------------------
Qekf = diag([0.002 0.002 0.0006 0.06 0.06 0.02]);    % process noise (6x6)
Rgps = diag([noise.gpsPos^2, noise.gpsPos^2]);        % GPS meas noise (2x2)
Rimu = diag([noise.imuYaw^2, noise.imuAccel^2]);      % IMU meas noise (2x2)

% --- LQR weighting (error state [e1 e2 e3 e4], input = steering) ---------
Qlqr = diag([6, 1, 6, 1]);
Rlqr = 700;
lqrIters = 25;    % 4x4 DARE converges well before 25 iterations; the old
                  % value of 60 was pure wasted computation every step.

% --- HUD / camera / lap-timing feature settings ----------------------------
cameraMode  = 'full';     % 'full' = fixed whole-track view (default);
                          % 'follow' = camera pans and zooms with the car
followSpan  = 22;         % [m] half-width of the follow-cam viewing window
trailMaxLen = 200;        % number of recent points kept in the bright fading highlight
                          % SMOOTHNESS FIX: was 500. This trail is drawn as an
                          % alpha-blended patch (the most expensive rendering
                          % primitive used anywhere in this dashboard), and with
                          % TWO trails (true + EKF estimate) at 500 points each,
                          % up to 1000 blended faces were being redrawn on every
                          % single render. Cutting this is a direct reduction in
                          % real per-render cost -- which is the only lever left
                          % that can let the adaptive pacing system afford MORE
                          % frequent, smaller updates within the SAME real-time
                          % budget it's already correctly hitting, without
                          % touching the pacing logic/target itself at all. The
                          % visible fading highlight is simply a bit shorter now.
skidThreshold = 4.0;      % [deg] tire slip angle above which skid marks are laid
maxSkidPoints = 250;      % cap on stored skid-mark points (oldest are dropped)
                          % -- also reduced from 500, same reasoning as above.
gMeterScale = 30.0;       % [m/s^2] full-scale range of the G-force meter
                          % Raised from 6.0 (was only ~0.61g at the outer
                          % ring -- observed peaks during hard cornering
                          % have been up to ~2.9g, wildly exceeding it).
                          % Now ~3.06g at the outer ring, same circles,
                          % comfortable headroom above anything observed.

% --- NEW FEATURE TOGGLES ----------------------------------------------------
% These are now set to a SHOWCASE configuration so you can just press
% Run and see the new additions working, with no editing required. The
% core physics/control (linear tires + LQR) are left on their original,
% most-tested settings -- everything layered on top of them is the new,
% visually-demonstrable stuff.
tireModelType    = 'linear';   % kept on the proven default (not 'pacejka')
veh.tireModel = tireModelType;

controllerType   = 'lqr';      % kept on the proven default (not 'mpc')
mpcHorizon = 15;                % steps of preview the MPC optimizes over
                                 % (only relevant if you later try 'mpc' above)

useLowGripZones = true;         % ON: two icy/wet patches now appear on the
                                 % track (light-blue tint) to stress-test grip
lowGripMu = 0.45;               % grip multiplier inside a low-grip zone

estimatorCompare = false;       % ON: a parallel Unscented Kalman Filter runs
                                 % alongside the EKF and prints a comparison
                                 % at the end (does not affect the driving)
                                 % DEFAULT CHANGED: was true. This runs a
                                 % full second filter (predict+update) on
                                 % EVERY physics step purely for an
                                 % end-of-run comparison metric -- real,
                                 % avoidable compute cost with no visual
                                 % benefit while watching live. Turn back
                                 % on only when you specifically want the
                                 % EKF-vs-UKF accuracy comparison.

showConfidenceEllipse = true;   % ON: EKF's 2-sigma position-uncertainty
                                 % ellipse drawn live on the track

numVehicles = 2;                % ON: a second ("ghost") car now drives the
                                 % track too, with a live collision-risk check
collisionThreshold = 4.0;       % [m] predicted-path proximity that triggers
                                 % a collision warning between the two cars

showMinimap = true;             % ON: small always-visible full-track overview

exportVideo  = true;             % ON: saves av_simulation.mp4 next to this
                                 % script when the run finishes
                                 %
                                 % DEFAULT RESTORED to true. Earlier in
                                 % this project, exportVideo=true meant
                                 % taking a DIFFERENT, slower code path
                                 % with its own fixed render cadence --
                                 % that's why it was switched off by
                                 % default for a while. That's no longer
                                 % how this works: video capture is now a
                                 % byproduct of the SAME fast, adaptive
                                 % live-rendering loop (see TIMING FIX #10
                                 % near the main loop), buffered in memory
                                 % and encoded to disk only after the run
                                 % finishes -- so it no longer costs any
                                 % extra render time during the live run,
                                 % and the .mp4 file just comes along for
                                 % free every time you press Run.
exportReport = true;            % ON: saves av_simulation_report.pdf (6 pages:
                                 % title, summary, dashboard, diagnostics,
                                 % Monte Carlo, LQR sweep)
                                 % next to this script when the run finishes

fprintf('=== Autonomous Vehicle Estimation + Control Simulation ===\n');
fprintf('Vehicle: m=%.0f kg, wheelbase=%.2f m, Cf=%.0f N/rad, Cr=%.0f N/rad\n', ...
        veh.m, veh.L, veh.Cf, veh.Cr);
fprintf('Understeer gradient Kus = %.6f rad*s^2/m\n', veh.Kus);
fprintf('Simulating %.0f s at %.0f Hz (%d steps, extending to finish the final lap if needed)...\n\n', tEnd, 1/dt, nStepsNominal);

%% ======================= SECTION 1.5: SELF-TESTS / VALIDATION ============
% Before trusting any of the downstream results, verify two things that
% would otherwise fail silently and produce plausible-looking-but-wrong
% output: (a) the numerical Jacobian used throughout the EKF is actually
% accurate, and (b) the LQR gain really does stabilize the closed-loop
% error dynamics at the operating speeds we care about. This kind of
% sanity check is standard practice before relying on any control code.

fprintf('--- Running self-tests ---\n');

% (a) Verify numJacobian against a function with a KNOWN analytic Jacobian.
testFun = @(x) [x(1)^2 + x(2)*x(3); sin(x(1)) + x(3)^2; x(2)*x(3)];
x0test  = [0.7; -1.3; 2.1];
Janalytic = [2*x0test(1), x0test(3), x0test(2);
             cos(x0test(1)), 0, 2*x0test(3);
             0, x0test(3), x0test(2)];
Jnumeric = numJacobian(testFun, x0test);
jacError = max(abs(Janalytic(:) - Jnumeric(:)));
fprintf('  numJacobian max error vs analytic Jacobian: %.3e  (%s)\n', ...
        jacError, ternaryStr(jacError < 1e-4, 'PASS', 'FAIL'));

% (b) Verify closed-loop stability of the LQR-controlled error dynamics
%     across the operating speed range: the discrete closed-loop matrix
%     (Ad - Bd*K) must have all eigenvalues strictly inside the unit
%     circle for the linearized error dynamics to be stable.
testSpeeds = linspace(lims.vmin, lims.vmax, 6);
fprintf('  Closed-loop stability check (|eig| < 1 required):\n');
allStable = true;
for vTest = testSpeeds
    [Atest, Btest] = lateralErrorModel(veh, vTest);
    Ktest = computeLQRGainDARE(Atest, Btest, Qlqr, Rlqr, dt, lqrIters);
    Ad_test = eye(4) + Atest*dt;
    Bd_test = Btest*dt;
    closedLoopEig = eig(Ad_test - Bd_test*Ktest);
    maxMag = max(abs(closedLoopEig));
    stableHere = maxMag < 1;
    allStable = allStable && stableHere;
    fprintf('    v = %5.1f m/s  ->  max |eig(closed-loop)| = %.4f  (%s)\n', ...
            vTest, maxMag, ternaryStr(stableHere, 'stable', 'UNSTABLE'));
end
if allStable
    fprintf('  All tested operating speeds are closed-loop stable. Proceeding.\n\n');
else
    warning('LQR closed-loop instability detected at one or more speeds -- retune Q/R.');
end


%% ======================= SECTION 2: TRACK GENERATION ====================

nTrackPts = 500;
[trackX, trackY, trackKappa, trackPsi, trackS] = generateTrack(nTrackPts, 18);

halfWidth = 3.2;
tnx = -sin(trackPsi); tny = cos(trackPsi);   % unit normals along track
outerX = trackX + halfWidth*tnx;  outerY = trackY + halfWidth*tny;
innerX = trackX - halfWidth*tnx;  innerY = trackY - halfWidth*tny;

% Total lap length (closed loop), used purely for lap-completion detection:
% every time the vehicle's cumulative distance traveled passes another
% multiple of this length, a lap has been completed.
trackTotalLength = sum(hypot(diff([trackX, trackX(1)]), diff([trackY, trackY(1)])));

% Low-grip zones: defined as fractions of the total lap length so they
% scale automatically with track size. Only the TRUE vehicle experiences
% reduced grip here (the controller's internal model does not know about
% them) -- this is what makes it a genuine robustness stress-test rather
% than something the controller can simply plan around.
if useLowGripZones
    lowGripZones = [0.28 0.40; 0.62 0.72] * trackTotalLength;
else
    lowGripZones = zeros(0,2);
end

%% ======================= SECTION 3: STATE INITIALIZATION ================

% True state:      [X; Y; psi; vx; vy; r]
X0 = trackX(1); Y0 = trackY(1); psi0 = trackPsi(1);
trueState = [X0; Y0; psi0; lims.vmin; 0; 0];

% EKF estimate starts with a deliberately imperfect guess to show convergence
estState = trueState + [0.8; -0.8; 0.05; 0; 0; 0];
P = diag([1.0 1.0 0.05 1.0 1.0 0.05]);

% Parallel UKF (comparison only -- never used for control) starts from the
% identical imperfect guess so the two filters are compared on equal footing
estStateUKF = estState;
Pukf = P;

% Controller integrator states
speedIntegral = 0;

% Logging pre-allocation
logT        = nan(1, nSteps);
logVxTrue   = nan(1, nSteps);
logVxEst    = nan(1, nSteps);
logVTarget  = nan(1, nSteps);
logDeltaFF  = nan(1, nSteps);
logDeltaFB  = nan(1, nSteps);
logDeltaCmd = nan(1, nSteps);
logE1       = nan(1, nSteps);
logPosErrEst= nan(1, nSteps);
logPosErrGps= nan(1, nSteps);
logPosErrUKF= nan(1, nSteps);   % UKF position error (comparison only)
logAlphaF   = nan(1, nSteps);   % front tire slip angle          [deg]
logAlphaR   = nan(1, nSteps);   % rear tire slip angle           [deg]
logAyTrue   = nan(1, nSteps);   % lateral acceleration            [m/s^2]
logAxTrue   = nan(1, nSteps);   % longitudinal acceleration        [m/s^2]
logPtrace   = nan(1, nSteps);   % EKF covariance trace (uncertainty)

% --- Lap timer state ---------------------------------------------------
lapCount        = 0;
distanceTraveled = 0;
lastLapStartTime = 0;
bestLapTime      = Inf;
prevTrueXY       = [X0; Y0];
lapCountAtDeadline = -1;   % sentinel: tEnd not reached yet (see the
                           % lap-finish stopping logic in the main loop)
stopAfterThisStep  = false;

% --- Skid-mark buffer (rear-axle contact points logged during high slip) --
skidX = []; skidY = [];

trailTrueX = X0; trailTrueY = Y0;
trailEstX  = X0; trailEstY  = Y0;
gpsHistX   = [];  gpsHistY = [];
maxGpsHist = 60;

% --- Second vehicle ("ghost car") for collision-prediction demo ----------
% Uses its OWN true state directly for control (no EKF -- the point here
% is demonstrating two independently-controlled vehicles predicting a
% potential collision, not a second full estimation stack). Started a
% quarter-lap ahead of vehicle 1.
if numVehicles >= 2
    idx2Start = find(trackS >= 0.25*trackTotalLength, 1, 'first');
    if isempty(idx2Start); idx2Start = 1; end
    trueState2 = [trackX(idx2Start); trackY(idx2Start); trackPsi(idx2Start); lims.vmin; 0; 0];
    mainIdx2 = [];
    cachedK2 = []; cachedVxForK2 = NaN; cachedA_lin2 = []; cachedB_lin2 = [];
    mpcUwarm2 = []; prevDeltaCmd2 = 0; speedIntegral2 = 0;
    trailTrue2X = trueState2(1); trailTrue2Y = trueState2(2);
    collisionWarning = false;
    deltaCmd2 = 0;   % populated once the loop starts; harmless placeholder
end

%% ======================= SECTION 4: DASHBOARD SETUP ======================

fig = figure('Color','k','Position',[60 40 1300 850], 'Name','Autonomous Vehicle Estimation & Control');

% ---- Main track panel (large, left column) ----
axMain = subplot(3,2,[1 3 5]); hold(axMain,'on'); axis(axMain,'equal'); axis(axMain,'off');
set(fig,'Color','k');

% Generous margins on every side so the boundary lines are never clipped
% by the axes box, and so there is dedicated blank space above the track
% (for the readout text) that never overlaps the road surface.
marginSide = 6; marginBottom = 22; marginTop = 11;
xlimMain = [min(outerX)-marginSide, max(outerX)+marginSide];
ylimMain = [min(outerY)-marginBottom, max(outerY)+marginTop];
xlim(axMain, xlimMain); ylim(axMain, ylimMain);

% Asphalt road surface: a single filled polygon between the outer and
% inner boundaries (outer forward, inner reversed) reads as a proper
% paved track instead of two disconnected outline curves.
surfaceX = [outerX, fliplr(innerX)];
surfaceY = [outerY, fliplr(innerY)];
patch(axMain, surfaceX, surfaceY, [0.13 0.13 0.16], 'EdgeColor','none');

% Low-grip zone overlay (icy/wet patches): a tinted section of road
% surface PLUS a dashed warning-colored border and a text label, so it
% reads clearly as "caution" rather than just a subtle color shift.
if useLowGripZones
    for zi = 1:size(lowGripZones,1)
        zoneIdx = find(trackS >= lowGripZones(zi,1) & trackS <= lowGripZones(zi,2));
        if numel(zoneIdx) >= 2
            zx = [outerX(zoneIdx), fliplr(innerX(zoneIdx))];
            zy = [outerY(zoneIdx), fliplr(innerY(zoneIdx))];
            patch(axMain, zx, zy, [0.55 0.80 1.0], 'EdgeColor','none', 'FaceAlpha',0.40);

            % dashed warning borders along the zone's outer and inner edges
            plot(axMain, outerX(zoneIdx), outerY(zoneIdx), '--', 'Color',[0.85 0.95 1.0], 'LineWidth',1.6);
            plot(axMain, innerX(zoneIdx), innerY(zoneIdx), '--', 'Color',[0.85 0.95 1.0], 'LineWidth',1.6);

            % text label at the zone's midpoint, offset outward past the
            % outer boundary (so it's never hidden under the car/trail),
            % rotated to roughly follow the local track direction
            midIdx = zoneIdx(round(numel(zoneIdx)/2));
            labelOffset = 2.2;
            labelX = outerX(midIdx) + labelOffset*tnx(midIdx);
            labelY = outerY(midIdx) + labelOffset*tny(midIdx);
            labelAngleDeg = rad2deg(trackPsi(midIdx));
            if labelAngleDeg > 90 || labelAngleDeg < -90
                labelAngleDeg = labelAngleDeg + 180;   % keep text upright/readable
            end
            text(axMain, labelX, labelY, '\bf\fontsize{9}\color[rgb]{0.75,0.9,1}SLIPPERY', ...
                 'Interpreter','tex', 'HorizontalAlignment','center', ...
                 'Rotation', labelAngleDeg);
        end
    end
end


plot(axMain, outerX([1:end 1]), outerY([1:end 1]), '-', 'Color',[0.75 0.75 0.78], 'LineWidth',1.4);
plot(axMain, innerX([1:end 1]), innerY([1:end 1]), '-', 'Color',[0.75 0.75 0.78], 'LineWidth',1.4);

% Dashed centerline, classic race-track styling
centerDash = plot(axMain, trackX([1:end 1]), trackY([1:end 1]), '--', ...
                   'Color',[0.5 0.5 0.5], 'LineWidth',0.9); %#ok<NASGU>

% Start/finish line: a short marker perpendicular to the track at pt 1
sfX = [trackX(1)+halfWidth*tnx(1), trackX(1)-halfWidth*tnx(1)];
sfY = [trackY(1)+halfWidth*tny(1), trackY(1)-halfWidth*tny(1)];
plot(axMain, sfX, sfY, '-', 'Color',[1 1 1], 'LineWidth',3);

% ---- Motion trails: TWO layers per vehicle path --
%      (1) a persistent, dim, full-history line so the whole path driven
%          so far is always visible at a glance (this is what was missing
%          before -- the fading-only version made the trail nearly
%          invisible except right behind the car), and
%      (2) a brighter, short "highlight" trail over just the last
%          `trailMaxLen` points with a fade-in effect for a premium,
%          motion-blur-like accent on top of the always-visible base line.
%      Separate, never-updated dummy lines are used purely so the legend
%      still shows clean, simple solid-line swatches. ----
hLegTrue = plot(axMain, nan, nan, '-', 'Color',[0 0.85 0.9], 'LineWidth',1.8);
hLegEst  = plot(axMain, nan, nan, '-', 'Color',[0.9 0.3 1.0], 'LineWidth',1.2);

hTrueTrailBase = plot(axMain, nan, nan, '-', 'Color',[0 0.85 0.9 0.30], 'LineWidth',1.3);
hEstTrailBase  = plot(axMain, nan, nan, '-', 'Color',[0.9 0.3 1.0 0.30], 'LineWidth',1.0);

hTrueTrail = patch(axMain, 'Faces', [1 2], 'Vertices', nan(2,2), 'FaceColor','none', ...
                    'EdgeColor',[0 0.85 0.9], 'LineWidth',2.2, 'FaceVertexAlphaData', [1;1], ...
                    'EdgeAlpha','interp', 'AlphaDataMapping','none');
hEstTrail  = patch(axMain, 'Faces', [1 2], 'Vertices', nan(2,2), 'FaceColor','none', ...
                    'EdgeColor',[0.9 0.3 1.0], 'LineWidth',1.6, 'FaceVertexAlphaData', [1;1], ...
                    'EdgeAlpha','interp', 'AlphaDataMapping','none');

hPred      = plot(axMain, nan, nan, '--', 'Color',[1 0.85 0.1], 'LineWidth',2.2);
hGps       = scatter(axMain, nan, nan, 14, [1 1 1]*0.7, 'filled', 'MarkerFaceAlpha',0.5);

% ---- Tire skid marks: small dark dots laid down at the rear axle when
%      slip angle exceeds skidThreshold -- persistent (not fading), like
%      real skid marks on asphalt. ----
hSkid = scatter(axMain, nan, nan, 9, [0.04 0.04 0.04], 'filled', 'MarkerFaceAlpha',0.4);

% ---- EKF position-uncertainty confidence ellipse (2-sigma), drawn live
%      from the covariance submatrix -- shrinks visibly as the filter
%      gains confidence, especially right after each GPS fix. ----
hEllipse = patch(axMain, nan, nan, [0.9 0.3 1.0], 'EdgeColor',[0.9 0.3 1.0], ...
                  'FaceAlpha',0.12, 'LineWidth',1.0);

% ---- Car icon: a proper top-down car silhouette instead of a bare
%      triangle "cursor" -- a rounded-rectangle body (smooth filleted
%      corners, not sharp 90-degree corners, for a more realistic
%      silhouette), a windshield stripe, and 4 visible rectangular tires
%      at the corners, all rotated together with heading.
carLen = 2.8; carWid = 1.5;

% --- Body: rounded rectangle with filleted corners (see roundedRectPoints) ---
bodyCornerRadius = 0.38;
carBodyLocal = roundedRectPoints(carLen/2, carWid/2, bodyCornerRadius, 8)';

% --- Windshield stripe: a smaller rounded rectangle set toward the front ---
cabinLenFrac = 0.34; cabinWidFrac = 0.72; cabinCenterFrac = 0.18;  % offset toward the nose
cabinLh = carLen/2*cabinLenFrac; cabinWh = carWid/2*cabinWidFrac;
cabinPts = roundedRectPoints(cabinLh, cabinWh, min(0.18, 0.4*min(cabinLh,cabinWh)), 6);
carCabinLocal = [cabinPts(:,1)' + carLen/2*cabinCenterFrac; cabinPts(:,2)'];

% --- 4 tires: small rectangles at the corners, aligned with body length
%     axis, sitting slightly proud of the body edges like real wheels ---
tireLen = 0.62; tireWid = 0.30;
tireOffsetX = carLen/2*0.62;
tireOffsetY = carWid/2 + tireWid*0.15;
wheelCorner = [ tireLen/2 tireWid/2; tireLen/2 -tireWid/2; -tireLen/2 -tireWid/2; -tireLen/2 tireWid/2];
wheelCenters = [ tireOffsetX  tireOffsetY;  tireOffsetX -tireOffsetY; ...
                 -tireOffsetX  tireOffsetY; -tireOffsetX -tireOffsetY];
carWheelsLocalVertices = zeros(16,2);
for w = 1:4
    carWheelsLocalVertices((w-1)*4+1 : w*4, :) = wheelCorner + wheelCenters(w,:);
end
carWheelsFaces = reshape(1:16, 4, 4)';

hCarBody   = patch(axMain, nan, nan, [1 0.2 0.2], 'EdgeColor',[0.05 0.05 0.05], 'LineWidth',1.0);
hCarCabin  = patch(axMain, nan, nan, [0.55 0.75 0.85], 'EdgeColor','none', 'FaceAlpha',0.85);
hCarWheels = patch(axMain, 'Faces', carWheelsFaces, 'Vertices', nan(16,2), ...
                    'FaceColor', [0.05 0.05 0.05], 'EdgeColor', 'none');

car1Color = [0.85 0.15 0.10];   % fixed solid red -- was previously shifting
                                 % continuously with speed via a colormap,
                                 % which read as confusing/flickery once a
                                 % second car was added. Now stays constant
                                 % and stays clearly distinct from car 2's blue.

% ---- Second vehicle ("ghost car") graphics: reuses the exact same body/
%      cabin/wheel shapes as the primary car, color-themed blue, and
%      pulses smoothly between semi-transparent and solid (see the
%      car2Alpha computation in the render loop) -- an intuitive
%      "ghost" visual cue that doesn't require reading a floating label. ----
if numVehicles >= 2
    car2Color = [0.25 0.5 1.0];
    blinkPeriod = 1.2;   % [s] duration of one smooth pulse cycle
    hTrue2TrailBase = plot(axMain, nan, nan, '-', 'Color',[0.3 0.55 1.0 0.35], 'LineWidth',1.3);
    hPred2 = plot(axMain, nan, nan, '--', 'Color',[0.5 0.75 1.0], 'LineWidth',1.6);
    hCar2Body   = patch(axMain, nan, nan, car2Color, 'EdgeColor',[0.05 0.05 0.05], 'LineWidth',1.0);
    hCar2Cabin  = patch(axMain, nan, nan, [0.75 0.85 0.95], 'EdgeColor','none', 'FaceAlpha',0.85);
    hCar2Wheels = patch(axMain, 'Faces', carWheelsFaces, 'Vertices', nan(16,2), ...
                        'FaceColor', [0.05 0.05 0.05], 'EdgeColor', 'none');
    txtCollision = text(axMain, 0.5, 0.5, '\bf COLLISION RISK', 'Color',[1 0.15 0.15], ...
                         'FontSize',15, 'Units','normalized', 'HorizontalAlignment','center', ...
                         'Interpreter','tex', 'Visible','off');
end


% ---- Speed panel (top-right) ----
axSpeed = subplot(3,2,2); hold(axSpeed,'on'); set(axSpeed,'Color','k','XColor','w','YColor','w');
title(axSpeed,'Speed tracking','Color','w','FontSize',10);
ylabel(axSpeed,'v [m/s]','Color','w'); grid(axSpeed,'on'); axSpeed.GridColor=[0.3 0.3 0.3];
hVxTrue  = plot(axSpeed, nan, nan, 'Color',[0 0.85 0.9], 'LineWidth',1.3);
hVxEst   = plot(axSpeed, nan, nan, 'Color',[0.9 0.3 1.0], 'LineWidth',1.0);
hVxTgt   = plot(axSpeed, nan, nan, ':', 'Color',[1 0.85 0.1], 'LineWidth',1.3);
% RENDER-SPEED FIX: 'Location','best' (all three legends below) forces
% MATLAB to re-examine the plotted data's extent EVERY SINGLE FRAME to
% find a spot that doesn't overlap it -- a well-documented, expensive
% per-frame cost, and the likely dominant reason each render was taking
% over a second (confirmed by direct frame-by-frame analysis of the
% exported video: ~62 actually-distinct frames across 73s, everything
% else was duplicated/frozen). Pinned to a fixed corner instead --
% visually the same spot 'best' was already landing on in every
% screenshot, but computed once at setup instead of every frame.
legend(axSpeed,[hVxTrue hVxEst hVxTgt],{'true','estimated','target'}, ...
       'TextColor','w','Color','none','EdgeColor',[0.4 0.4 0.4],'Location','northeast');
xlim(axSpeed,[0 tEnd]); ylim(axSpeed,[0 lims.vmax+2]);

% ---- Steering panel (middle-right) ----
axSteer = subplot(3,2,4); hold(axSteer,'on'); set(axSteer,'Color','k','XColor','w','YColor','w');
title(axSteer,'Steering command decomposition','Color','w','FontSize',10);
ylabel(axSteer,'\delta [deg]','Color','w'); grid(axSteer,'on'); axSteer.GridColor=[0.3 0.3 0.3];
hDeltaFF  = plot(axSteer, nan, nan, 'Color',[0.4 1 0.4], 'LineWidth',1.1);
hDeltaFB  = plot(axSteer, nan, nan, 'Color',[0.95 0.15 0.15], 'LineWidth',1.1);
hDeltaCmd = plot(axSteer, nan, nan, 'Color','w', 'LineWidth',1.4);
legend(axSteer,[hDeltaFF hDeltaFB hDeltaCmd],{'feedforward','LQR feedback','total (clamped)'}, ...
       'TextColor','w','Color','none','EdgeColor',[0.4 0.4 0.4],'Location','northeast');
xlim(axSteer,[0 tEnd]);
% RENDER-SPEED FIX: no ylim was set here at all previously, meaning the
% Y-axis was on MATLAB's auto-rescale mode -- every frame, new data could
% trigger a full recompute of axis limits/ticks/gridlines (and, combined
% with the 'best' legend above, its position too). Fixed to a range wide
% enough for the feedforward/feedback/clamped-total steering signals.
ylim(axSteer,[-15 32]);

% ---- Error / estimation panel (bottom-right) ----
axErr = subplot(3,2,6); hold(axErr,'on'); set(axErr,'Color','k','XColor','w','YColor','w');
title(axErr,'Tracking & estimation error','Color','w','FontSize',10);
xlabel(axErr,'time [s]','Color','w'); ylabel(axErr,'error [m]','Color','w');
grid(axErr,'on'); axErr.GridColor=[0.3 0.3 0.3];
hE1     = plot(axErr, nan, nan, 'Color',[1 0.85 0.1], 'LineWidth',1.3);
hPosEst = plot(axErr, nan, nan, 'Color',[0.9 0.3 1.0], 'LineWidth',1.1);
hPosGps = plot(axErr, nan, nan, ':', 'Color',[0.7 0.7 0.7], 'LineWidth',1.0);
legend(axErr,[hE1 hPosEst hPosGps],{'cross-track error e_1','EKF position error','raw GPS error'}, ...
       'TextColor','w','Color','none','EdgeColor',[0.4 0.4 0.4],'Location','northeast');
xlim(axErr,[0 tEnd]);
% RENDER-SPEED FIX: same as axSteer above -- no ylim previously, now fixed
% to a range wide enough for the cross-track/EKF/raw-GPS error signals.
ylim(axErr,[-2.5 2.5]);

% ---- Overall dashboard header ----
sgtitle(fig, {'AUTONOMOUS VEHICLE STATE ESTIMATION & OPTIMAL CONTROL', ...
    '\fontsize{9}\color[rgb]{0.7,0.7,0.7}Dynamic Bicycle Model  \cdot  Extended Kalman Filter  \cdot  LQR Path Tracking  \cdot  Live Simulation'}, ...
    'Color','w', 'FontSize',16, 'FontWeight','bold', 'Interpreter','tex');

% ---- Legend: placed INSIDE the empty infield of the track loop instead
%      of floating on top of the road, using the axes' final (post-title)
%      Position to convert a target DATA point (the track's center, which
%      is always empty black space for this track family) into the
%      normalized figure coordinates the legend needs. ----
drawnow;   % finalize axMain.Position after sgtitle has reserved its space
legMain = legend(axMain, [hLegTrue hLegEst hPred hGps], ...
    {'true path','EKF estimate','predicted (from estimate)','raw GPS'}, ...
    'TextColor','w','Color',[0 0 0], 'EdgeColor',[0.4 0.4 0.4], 'Location','none');
legW = 0.17; legH = 0.14;
targetDataX = mean(xlimMain);                       % horizontal center of track
targetDataY = mean(outerY);                          % vertical center of the infield
axFracX = (targetDataX - xlimMain(1)) / diff(xlimMain);
axFracY = (targetDataY - ylimMain(1)) / diff(ylimMain);
figX = axMain.Position(1) + axFracX*axMain.Position(3) - legW/2;
figY = axMain.Position(2) + axFracY*axMain.Position(4) - legH/2;
legMain.Units = 'normalized';
legMain.Position = [figX, figY, legW, legH];

% ---- Live readout text: placed using NORMALIZED axes units (not data
%      coordinates) so it stays fixed in the same on-screen spot above
%      the road regardless of margin size or camera mode (fixed/follow). ----
txtMain = text(axMain, 0.5, 0.985, '', 'Color','w', 'FontSize',11.5, ...
               'Interpreter','latex', 'HorizontalAlignment','center', ...
               'VerticalAlignment','top', 'Units','normalized');

% ---- HUD gauges: speedometer, steering dial, G-force meter -- three
%      small inset axes positioned along the bottom edge of the main
%      track panel (inside its bottom margin, which is intentionally
%      blank asphalt-free space). Positions are computed from axMain's
%      own normalized Position so they scale correctly with figure size. ----
axPos   = axMain.Position;
gaugeGap = axPos(3)*0.018;
gaugeW   = (axPos(3) - 2*gaugeGap)/3;
gaugeH   = axPos(4)*0.22;
gaugeY   = axPos(2) + axPos(4)*0.004;

speedoPos   = [axPos(1),                       gaugeY, gaugeW, gaugeH];
steerPos    = [axPos(1)+1*(gaugeW+gaugeGap),    gaugeY, gaugeW, gaugeH];
gmeterPos   = [axPos(1)+2*(gaugeW+gaugeGap),    gaugeY, gaugeW, gaugeH];


% --- Speedometer: semicircular dial, needle sweeps from 0 (left) to
%     lims.vmax*1.15 (right). Tick marks are placed at "nice" round
%     intervals (e.g. 0,5,10,15... rather than arbitrary numbers),
%     spaced automatically according to the gauge's actual range. ---
speedoColor = [1 0.35 0.25];
speedMaxDisplay = lims.vmax*1.15;
axSpeedo = axes('Parent',fig, 'Position',speedoPos); hold(axSpeedo,'on');
axis(axSpeedo,'equal'); axis(axSpeedo,'off');
xlim(axSpeedo,[-1.3 1.3]); ylim(axSpeedo,[-0.25 1.48]);
thetaArc = linspace(pi, 0, 60);
plot(axSpeedo, cos(thetaArc), sin(thetaArc), 'Color',[0.75 0.75 0.75], 'LineWidth',2.4);
text(axSpeedo, 0, 1.36, 'SPEED (m/s)', 'Color',[0.9 0.9 0.9], 'FontSize',8.5, 'FontWeight','bold', ...
     'HorizontalAlignment','center');
speedTicks = niceTickValues(speedMaxDisplay, 5);
for tv = speedTicks
    angT = pi*(1 - tv/speedMaxDisplay);
    plot(axSpeedo, [0.90 1.0]*cos(angT), [0.90 1.0]*sin(angT), 'Color',[0.8 0.8 0.8], 'LineWidth',1.3);
    text(axSpeedo, 1.14*cos(angT), 1.14*sin(angT), sprintf('%g',tv), 'Color',[0.85 0.85 0.85], ...
         'FontSize',7.5, 'FontWeight','bold', 'HorizontalAlignment','center', 'VerticalAlignment','middle');
end
hSpeedoNeedle = plot(axSpeedo, [0 0], [0 0.85], 'Color',speedoColor, 'LineWidth',3.2);
scatter(axSpeedo, 0, 0, 36, [0.9 0.9 0.9], 'filled');
txtSpeedoVal = text(axSpeedo, 0, -0.15, '', 'Color',speedoColor, 'FontSize',13, ...
                     'FontWeight','bold', 'HorizontalAlignment','center', 'VerticalAlignment','top');

% --- Steering dial: semicircular dial, needle sweeps from -maxSteer
%     (left) through 0 (straight up) to +maxSteer (right), with
%     symmetric nice-interval tick marks (e.g. -30,-15,0,15,30). ---
steerColor = [0.35 1 0.55];
steerMaxDeg = rad2deg(lims.maxSteer);
axSteerG = axes('Parent',fig, 'Position',steerPos); hold(axSteerG,'on');
axis(axSteerG,'equal'); axis(axSteerG,'off');
xlim(axSteerG,[-1.3 1.3]); ylim(axSteerG,[-0.25 1.48]);
plot(axSteerG, cos(thetaArc), sin(thetaArc), 'Color',[0.75 0.75 0.75], 'LineWidth',2.4);
text(axSteerG, 0, 1.36, 'STEERING (deg)', 'Color',[0.9 0.9 0.9], 'FontSize',8.5, 'FontWeight','bold', ...
     'HorizontalAlignment','center');
posSteerTicks = niceTickValues(steerMaxDeg, 3);
steerTicks = unique([-fliplr(posSteerTicks), posSteerTicks]);
steerTicks(abs(steerTicks) < 1e-9) = 0;   % avoid IEEE "-0" printing as "-0" instead of "0"
for tv = steerTicks
    angT = pi/2 - (tv/steerMaxDeg)*(pi/2);
    plot(axSteerG, [0.90 1.0]*cos(angT), [0.90 1.0]*sin(angT), 'Color',[0.8 0.8 0.8], 'LineWidth',1.3);
    text(axSteerG, 1.14*cos(angT), 1.14*sin(angT), sprintf('%g',tv), 'Color',[0.85 0.85 0.85], ...
         'FontSize',7, 'FontWeight','bold', 'HorizontalAlignment','center', 'VerticalAlignment','middle');
end
hSteerNeedle = plot(axSteerG, [0 0], [0 0.85], 'Color',steerColor, 'LineWidth',3.2);
scatter(axSteerG, 0, 0, 36, [0.9 0.9 0.9], 'filled');
txtSteerVal = text(axSteerG, 0, -0.15, '', 'Color',steerColor, 'FontSize',13, ...
                    'FontWeight','bold', 'HorizontalAlignment','center', 'VerticalAlignment','top');

% --- G-force meter: classic racing "g-meter" -- concentric rings plus a
%     dot showing combined lateral/longitudinal acceleration, with the
%     rings labeled in g like the tick marks on the other two dials. ---
gColor = [1 0.85 0.15];
axGmeter = axes('Parent',fig, 'Position',gmeterPos); hold(axGmeter,'on');
axis(axGmeter,'equal'); axis(axGmeter,'off');
xlim(axGmeter,[-1.3 1.3]); ylim(axGmeter,[-1.35 1.4]);
thetaFull = linspace(0,2*pi,60);
plot(axGmeter, 0.5*cos(thetaFull), 0.5*sin(thetaFull), 'Color',[0.5 0.5 0.5], 'LineWidth',1);
plot(axGmeter, 1.0*cos(thetaFull), 1.0*sin(thetaFull), 'Color',[0.65 0.65 0.65], 'LineWidth',1.2);
plot(axGmeter, [-1.05 1.05], [0 0], 'Color',[0.4 0.4 0.4], 'LineWidth',0.8);
plot(axGmeter, [0 0], [-1.05 1.05], 'Color',[0.4 0.4 0.4], 'LineWidth',0.8);
text(axGmeter, 0.5, 0.08, sprintf('%.1f',0.5*gMeterScale/9.81), 'Color',[0.75 0.75 0.75], 'FontSize',6.5, 'HorizontalAlignment','center');
text(axGmeter, 1.0, 0.08, sprintf('%.1f',1.0*gMeterScale/9.81), 'Color',[0.75 0.75 0.75], 'FontSize',6.5, 'HorizontalAlignment','center');
text(axGmeter, 0, 1.22, 'G-FORCE', 'Color',[0.9 0.9 0.9], 'FontSize',8.5, 'FontWeight','bold', ...
     'HorizontalAlignment','center');
% G-force position dot: switched from scatter() to plot()-based marker --
% this is the exact same fix that resolved the minimap car dot earlier,
% which had the identical issue (scatter() objects not reliably showing
% up / updating in this dashboard, while every plot()-based marker works
% fine). Consistent going forward: no scatter() markers left in the live
% loop anywhere in this file.
hGdot = plot(axGmeter, 0, 0, 'o', 'MarkerSize',9, 'MarkerFaceColor',gColor, 'MarkerEdgeColor','none');
txtGVal = text(axGmeter, 0, -1.24, '', 'Color',gColor, 'FontSize',11.5, 'FontWeight','bold', ...
               'HorizontalAlignment','center');

% ---- Minimap: positioned OUTSIDE axMain, in the gap to its right (NOT
%      an inset overlapping axMain's own area). LAYOUT FIX: the previous
%      "picture-in-picture" version sat inside axMain's own top-right
%      corner, which is exactly where the v=/e1=/pos.err. readout text
%      lives (see txtMain above, anchored at the top of axMain) -- the
%      two were directly colliding. Positioning it OUTSIDE axMain
%      entirely means it geometrically cannot overlap anything drawn
%      inside axMain, by construction, while still being clamped so it
%      can't collapse to nothing or intrude into axSpeed's area either. ----
if showMinimap
    axSpeedPos = axSpeed.Position;
    figAspect  = fig.Position(3)/fig.Position(4);
    dataAspect = diff(xlimMain)/diff(ylimMain);

    gapAvailable = axSpeedPos(1) - (axPos(1) + axPos(3));   % horizontal room between the two panels
    % Genuinely small now (a GTA-style corner minimap, not a second track
    % view) -- the previous enlargement made it look like an oversized
    % empty box instead of a compact overview.
    minimapW = min(gapAvailable - 0.010, axPos(3)*0.075);
    minimapW = max(minimapW, 0.040);
    minimapH = minimapW * figAspect / dataAspect;

    minimapLeft   = axPos(1) + axPos(3) + 0.006;                       % just outside axMain's right edge
    minimapLeft   = min(minimapLeft, axSpeedPos(1) - minimapW - 0.006); % hard clamp: never reaches into axSpeed
    minimapBottom = axPos(2) + axPos(4)*0.62;   % upper-middle of the track panel's height -- clear of the
                                                 % readout text row at the very top, and clear of the legend box

    minimapPos = [minimapLeft, minimapBottom, minimapW, minimapH];
    axMinimap = axes('Parent',fig, 'Position',minimapPos); hold(axMinimap,'on');
    axis(axMinimap,'equal'); axis(axMinimap,'off');
    set(axMinimap,'Color',[0.08 0.08 0.10]);
    xlim(axMinimap, xlimMain); ylim(axMinimap, ylimMain);
    plot(axMinimap, outerX([1:end 1]), outerY([1:end 1]), 'Color',[0.85 0.88 0.92], 'LineWidth',1.3);
    plot(axMinimap, innerX([1:end 1]), innerY([1:end 1]), 'Color',[0.85 0.88 0.92], 'LineWidth',1.3);
    % Car marker: switched from scatter() to plot()-based markers.
    % Every other moving element in this dashboard (trails, predicted
    % path, GPS points, everything) is a plot/line object and works
    % reliably; scatter() has been the one thing that never showed up
    % despite several rounds of fixes (position, size, initial value),
    % so it's removed as a variable entirely rather than patched again.
    % No alpha transparency either -- one less thing that can silently
    % fail to render depending on the graphics renderer in use.
    hMinimapCarGlow = plot(axMinimap, X0, Y0, 'o', 'MarkerSize',9, ...
        'MarkerFaceColor',[0.6 0.1 0.1], 'MarkerEdgeColor','none');
    hMinimapCar = plot(axMinimap, X0, Y0, 'o', 'MarkerSize',3.5, ...
        'MarkerFaceColor',[1 0.25 0.25], 'MarkerEdgeColor','w', 'LineWidth',0.8);
    fprintf('Minimap enabled: Position = [%.4f %.4f %.4f %.4f] (figure-normalized)\n', minimapPos);
end

% --- LAP panel: moved INTO the main gauge row, immediately to the left
%     of the speedometer, using the EXACT SAME size (gaugeW/gaugeH) as
%     SPEED/STEERING/G-FORCE -- those three are locked and untouched;
%     LAP simply extends the row one panel further left. ---
lapColor = [0.4 0.95 1.0];
lapPanelPos = [speedoPos(1)-gaugeGap-gaugeW, gaugeY, gaugeW, gaugeH];
axLapPanel = axes('Parent',fig, 'Position',lapPanelPos); hold(axLapPanel,'on');
axis(axLapPanel,'equal'); axis(axLapPanel,'off');
xlim(axLapPanel,[-1.3 1.3]); ylim(axLapPanel,[-0.25 1.48]);

% Outer bezel -- shorter than before (bottom at y=0.00, not -0.15) so the
% continuously-updating current-lap-time readout can live BELOW the box,
% as its own big colorful number, exactly like how SPEED/STEERING/G-FORCE
% show their live value below their gauge rather than as small text
% crammed inside a border. This also fixes the "best" line overlapping
% the bottom edge -- the box interior now holds only the two things that
% change once per lap (LAP N, best time), not the value that changes
% every single frame.
rectangle(axLapPanel, 'Position',[-1.15 0.00 2.3 1.20], 'Curvature',0.10, ...
          'EdgeColor',[0.5 0.5 0.5], 'LineWidth',1.6, 'FaceColor',[0.05 0.06 0.08]);

% Title floats ABOVE the box (y=1.36) -- same proven-safe placement as the
% SPEED/STEERING/G-FORCE gauge titles right next to it.
text(axLapPanel, 0, 1.36, 'LAP', 'Color',[0.95 0.98 1], 'FontSize',8.5, 'FontWeight','bold', ...
     'HorizontalAlignment','center');

% Recessed "screen" area behind the digits, for a bezel/inset look
rectangle(axLapPanel, 'Position',[-1.08 0.06 2.16 1.06], ...
          'EdgeColor',[0.22 0.24 0.27], 'LineWidth',0.8, 'FaceColor',[0.025 0.03 0.04]);

txtLapNum = text(axLapPanel, 0, 0.88, '', 'Color',lapColor, 'FontSize',15, 'FontWeight','bold', ...
                  'HorizontalAlignment','center', 'FontName','Consolas');
plot(axLapPanel, [-0.85 0.85], [0.62 0.62], 'Color',[0.3 0.32 0.35], 'LineWidth',0.8);
txtLapBest = text(axLapPanel, 0, 0.40, '', 'Color',[1 0.85 0.2], 'FontSize',10, 'FontWeight','bold', ...
                   'HorizontalAlignment','center', 'FontName','Consolas');

% Live progress bar: fills left-to-right as the current lap is completed
rectangle(axLapPanel, 'Position',[-0.95 0.13 1.9 0.10], 'Curvature',0.5, ...
          'EdgeColor',[0.25 0.27 0.3], 'LineWidth',0.6, 'FaceColor',[0.06 0.07 0.09]);
hLapProgress = rectangle(axLapPanel, 'Position',[-0.95 0.13 0 0.10], 'Curvature',0.5, ...
                          'EdgeColor','none', 'FaceColor',lapColor);

% Current lap time: OUT of the box, as a big colorful live readout -- same
% position/size/style/alignment as txtSpeedoVal and txtSteerVal below
% their gauges (matched exactly, including 'VerticalAlignment','top' and
% the default font -- those two were the actual cause of it visually
% sitting lower than the other three: it had neither set, silently
% falling back to MATLAB's default 'middle' alignment plus a different
% font, both of which shift where the glyph body renders relative to the
% same y-coordinate even though the anchor point was numerically identical).
txtLapCur = text(axLapPanel, 0, -0.15, '', 'Color',lapColor, 'FontSize',13, ...
                  'FontWeight','bold', 'HorizontalAlignment','center', 'VerticalAlignment','top');

drawHudCorners(axLapPanel, -1.15, 0.00, 2.3, 1.20, 0.20, lapColor);

% --- CLOCK panel: moved to the figure's top-left corner (per your
%     green-box marking), positioned in absolute figure coordinates
%     rather than relative to axMain, since it now lives entirely
%     outside the track panel's own area. ---
clockColor = [1 0.55 0.85];
recColor   = [1 0.3 0.3];
clockPanelPos = [0.015, 0.855, 0.12, 0.125];
axClock = axes('Parent',fig, 'Position',clockPanelPos); hold(axClock,'on');
axis(axClock,'equal'); axis(axClock,'off');
xlim(axClock,[-1.3 1.3]); ylim(axClock,[-0.25 1.48]);

% Outer bezel
rectangle(axClock, 'Position',[-1.15 -0.15 2.3 1.35], 'Curvature',0.08, ...
          'EdgeColor',[0.5 0.5 0.5], 'LineWidth',1.6, 'FaceColor',[0.05 0.06 0.08]);

% Title floats ABOVE the box (y=1.36), same proven-safe placement as the
% LAP panel and the SPEED/STEERING/G-FORCE gauge titles, plus a pulsing
% status dot next to it (pink "LIVE" while playing back on-screen, red
% "REC" while exporting video -- its blink period is tied to SIMULATED
% time t, not the wall clock, so it pulses at a consistent visual rate in
% the video too).
indicatorColor = clockColor; if exportVideo, indicatorColor = recColor; end
text(axClock, 0.12, 1.36, 'ELAPSED TIME', 'Color',[0.95 0.98 1], 'FontSize',8, 'FontWeight','bold', ...
     'HorizontalAlignment','center');
hClockDot = scatter(axClock, -0.98, 1.36, 34, indicatorColor, 'filled');

% Recessed "screen" area behind the digits
rectangle(axClock, 'Position',[-1.08 -0.08 2.16 1.08], ...
          'EdgeColor',[0.22 0.24 0.27], 'LineWidth',0.8, 'FaceColor',[0.025 0.03 0.04]);

txtClock = text(axClock, 0, 0.80, '00:00', 'Color',clockColor, 'FontSize',15, 'FontWeight','bold', ...
                'HorizontalAlignment','center', 'FontName','Consolas');
plot(axClock, [-0.85 0.85], [0.48 0.48], 'Color',[0.3 0.32 0.35], 'LineWidth',0.8);
txtClockSub = text(axClock, 0, 0.30, sprintf('real-time \\cdot %.1fx', playbackSpeed), ...
                    'Color',[0.75 0.75 0.75], 'FontSize',6.5, ...
                    'HorizontalAlignment','center', 'Interpreter','tex');

% Overall race-progress bar: fraction of tEnd elapsed so far
rectangle(axClock, 'Position',[-0.95 -0.02 1.9 0.09], 'Curvature',0.6, ...
          'EdgeColor',[0.25 0.27 0.3], 'LineWidth',0.6, 'FaceColor',[0.06 0.07 0.09]);
hClockProgress = rectangle(axClock, 'Position',[-0.95 -0.02 0 0.09], 'Curvature',0.6, ...
                            'EdgeColor','none', 'FaceColor',clockColor);

drawHudCorners(axClock, -1.15, -0.15, 2.3, 1.35, 0.22, clockColor);

drawnow;

%% ======================= SECTION 5: MAIN SIMULATION LOOP =================

lastGpsMeas = [X0; Y0];

% --- LQR gain cache: the 4x4 Riccati solve is re-run only when the design
%     speed has drifted by more than lqrGainCacheTol since the last solve,
%     instead of on every single physics step. Speed changes gradually
%     (accel/decel limits are a few m/s^2), so the gain barely changes
%     step-to-step -- recomputing it 50 times per second was pure waste.
cachedK = [];
cachedVxForK = NaN;
cachedA_lin = []; cachedB_lin = [];
mpcUwarm = [];      % MPC warm-start: previous solution, shifted each step
prevDeltaCmd = 0;

% --- Real-time playback pacing: TIMING FIX #4 (redesign, replacing the
%     accumulator/catch-up pattern entirely). The previous design tried to
%     force physics to match real elapsed wall-clock time via an
%     accumulator, with a cap on how much backlog it could "catch up" at
%     once. That cap could be tuned (25 steps, then 500 steps) but the
%     real problem was structural: if a single physics step (control +
%     EKF + parallel UKF + logging) takes longer to COMPUTE than dt=20ms
%     of real time -- entirely plausible with estimatorCompare=true
%     running a full second filter every step -- then no cap, however
%     generous, can make the sim hit real-time, because the bottleneck
%     isn't rendering, it's the physics/estimation math itself. Trying to
%     force a fixed wall-clock ratio in that situation just means the sim
%     perpetually chases a target it structurally cannot reach, which
%     LOOKS like broken slow-motion even though nothing is actually stuck.
%
%     New approach: run one physics step per outer-loop iteration, as
%     fast as the machine can, with NO wall-clock-based throttling on the
%     physics itself. Rendering happens on a fixed, deterministic NUMBER
%     OF PHYSICS STEPS (the same approach already used for video export),
%     not on a wall-clock timer -- so a slow render can never cause
%     frames (or sim-time progress) to be silently dropped. The ONLY
%     wall-clock interaction left is a small optional pause AFTER a
%     render, and only if the sim is currently running AHEAD of real
%     time -- this keeps the "real-time feel" on a machine that can
%     comfortably keep up, while a slower machine just runs at whatever
%     pace it can sustain, smoothly and honestly (see the CLOCK panel's
%     live "Xx realtime" readout), instead of stalling or drifting. ---
liveStepsPerFrame = max(1, round((1/liveTargetFPS)/dt));
renderDurationHistN   = 5;    % rolling median window size for pacing (see TIMING FIX #13)
renderDurationHistory = zeros(1, renderDurationHistN);
renderDurationHistIdx = 0;
renderEventCount = 0;   % counts render events -- used to react FAST for the
                        % first few renders (getting the cadence into a
                        % reasonable ballpark quickly, since this is
                        % exactly the window most screenshots/videos
                        % capture), then switch to the heavily-smoothed,
                        % stable behavior once it's had a chance to find
                        % roughly the right value (see TIMING FIX #14).

% --- Video export setup (optional) ------------------------------------
% TIMING FIX #10 (unified pacing): video capture no longer runs on its
% own separate cadence/branch. Previously, turning exportVideo on meant
% the WHOLE loop switched to a fixed videoTargetFPS render rate instead
% of the fast, adaptive live pacing -- so getting a video always meant
% giving up the fast live-viewing experience. Now the loop ALWAYS uses
% the single adaptive live-pacing cadence (liveStepsPerFrame), and if
% exportVideo is on, each of those SAME render events is also captured
% into an in-memory buffer (getframe is cheap compared to the render
% itself, which already happened for the screen either way) -- there is
% no extra rendering cost for exporting a video anymore. The actual
% VideoWriter can't be opened yet: it needs a fixed FrameRate up front,
% but we don't know the total frame count or final simulated duration
% until the adaptive run finishes, so that happens in a short
% post-processing step after the main loop (Section 6) -- writing
% already-rendered frames to disk is fast; it's the RENDERING that was
% ever expensive, and that cost is now paid exactly once, not twice.
if exportVideo
    videoMaxBufferedFrames = 4000;   % safety cap (~a few GB at typical
                                     % figure resolution) -- if a run
                                     % somehow produces more frames than
                                     % this, further frames are simply
                                     % skipped rather than risking memory
                                     % exhaustion; a normal ~70-100s run
                                     % at this pacing produces nowhere
                                     % near this many.
    % TIMING FIX #12 (patchy live pacing caused by video capture):
    % preallocated up front instead of growing with {end+1} each frame.
    % Growing a cell array of large image structs one element at a time
    % causes occasional, unpredictable reallocation/copy spikes as MATLAB
    % re-sizes the underlying array -- and that overhead was landing
    % INSIDE the render-timing measurement the adaptive pacing depends on
    % (see the renderDuration fix further down), which is exactly what
    % was making video capture visibly corrupt the LIVE simulation's own
    % smoothness, not just the exported file.
    videoFrameBuffer = cell(1, videoMaxBufferedFrames);
    videoFrameTimes  = zeros(1, videoMaxBufferedFrames);
    videoFrameCount  = 0;
    fprintf('Recording video frames as the live simulation runs (encoded to av_simulation.mp4 after it finishes) ...\n');
end
fprintf(['MODE: paced to real time (playbackSpeed=%.1fx) and self-tunes its render ' ...
         'rate to this machine; watch the CLOCK panel''s "Xx realtime" readout to see ' ...
         'how well it''s keeping up.\n'], playbackSpeed);

simTimer  = tic;
lastRenderWallTime = -Inf;   % forces the very first frame to render

% --- Sticky path-search index: seeded with a one-time GLOBAL search, then
%     kept "stuck" to the correct region of the track via the windowed
%     search inside projectToPath (see that function's docstring). ---
mainIdx = [];

step = 0;
while step < nSteps

    step = step + 1;
    t = step*dt;

    % ---------------------------------------------------------------
    % 5.1  CONTROL DESIGN using the CURRENT ESTIMATE (never the truth)
    % ---------------------------------------------------------------
    vxForDesign = max(estState(4), 2.0);   % avoid singular design at v=0

    [mainIdx, psiPath, kappaPath, e1] = projectToPath( ...
        estState(1), estState(2), trackX, trackY, trackKappa, trackPsi, mainIdx);

    e3 = wrapAnglePi(estState(3) - psiPath);
    e2 = estState(4)*sin(e3) + estState(5)*cos(e3);
    psiDotDesired = vxForDesign*kappaPath;
    e4 = estState(6) - psiDotDesired;
    errState = [e1; e2; e3; e4];

    if isempty(cachedK) || abs(vxForDesign - cachedVxForK) > lqrGainCacheTol
        [A_lin, B_lin] = lateralErrorModel(veh, vxForDesign);
        cachedK = computeLQRGainDARE(A_lin, B_lin, Qlqr, Rlqr, dt, lqrIters);
        cachedVxForK = vxForDesign;
        cachedA_lin = A_lin; cachedB_lin = B_lin;
    end

    deltaFF = feedforwardSteering(kappaPath, vxForDesign, veh.L, veh.Kus);

    if strcmp(controllerType, 'mpc')
        % Linear MPC: optimize a full horizon of steering feedback moves
        % subject to a real box constraint |u|<=maxSteer enforced DURING
        % the optimization (not clamped after the fact like the LQR path).
        % Solved via projected gradient descent on the (cheap, linear)
        % quadratic cost -- see solveLinearMPC for details.
        Ad_mpc = eye(4) + cachedA_lin*dt;
        Bd_mpc = cachedB_lin*dt;
        mpcUwarm = solveLinearMPC(errState, Ad_mpc, Bd_mpc, Qlqr, Rlqr, mpcHorizon, lims.maxSteer, mpcUwarm);
        deltaFB = mpcUwarm(1);
    else
        deltaFB = -cachedK*errState;
    end

    deltaCmd = deltaFF + deltaFB;
    deltaCmd = max(min(deltaCmd, lims.maxSteer), -lims.maxSteer);
    deltaCmd = rateLimitStep(deltaCmd, prevDeltaCmd, lims.maxSteerRate*dt);
    prevDeltaCmd = deltaCmd;

    % ---------------------------------------------------------------
    % 5.2  SPEED PLANNING + simple PI control (using the estimate)
    % ---------------------------------------------------------------
    vTarget = speedPlanner(kappaPath, lims.vmin, lims.vmax, lims.ayMax);
    vErr = vTarget - estState(4);
    speedIntegral = speedIntegral + vErr*dt;
    speedIntegral = max(min(speedIntegral, 5), -5);   % anti-windup clamp
    Kp_v = 1.2; Ki_v = 0.3;
    axCmd = Kp_v*vErr + Ki_v*speedIntegral;
    axCmd = max(min(axCmd, lims.maxAccel), -lims.maxDecel);

    inputCmd = [deltaCmd; axCmd];

    % ---------------------------------------------------------------
    % 5.3  ADVANCE THE TRUE VEHICLE (ground truth, unknown to controller)
    % ---------------------------------------------------------------
    if useLowGripZones
        sNowForGrip = mod(distanceTraveled, trackTotalLength);
        vehTrue = veh;
        vehTrue.muScale = frictionScaleAt(sNowForGrip, lowGripZones, lowGripMu);
    else
        vehTrue = veh;
    end
    % Real disturbance applied to what the TRUE vehicle actually receives
    % (see procNoise in Section 4) -- the controller commanded inputCmd,
    % but the physical actuator/environment never delivers that exactly.
    % This is what makes each lap genuinely different instead of a
    % carbon copy of the last.
    trueInputCmd = inputCmd + [procNoise.steer*randn; procNoise.accel*randn];
    trueInputCmd(1) = max(min(trueInputCmd(1), lims.maxSteer), -lims.maxSteer);
    trueState = rk4Step(trueState, trueInputCmd, dt, vehTrue);
    trailTrueX(end+1) = trueState(1); %#ok<AGROW>
    trailTrueY(end+1) = trueState(2); %#ok<AGROW>

    % ---------------------------------------------------------------
    % 5.3b  SECOND VEHICLE ("ghost car"): identical control approach,
    %       driven directly off its own true state (no estimation stack).
    % ---------------------------------------------------------------
    if numVehicles >= 2
        vxForDesign2 = max(trueState2(4), 2.0);
        [mainIdx2, psiPath2, kappaPath2, e1_2] = projectToPath( ...
            trueState2(1), trueState2(2), trackX, trackY, trackKappa, trackPsi, mainIdx2);
        e3_2 = wrapAnglePi(trueState2(3) - psiPath2);
        e2_2 = trueState2(4)*sin(e3_2) + trueState2(5)*cos(e3_2);
        e4_2 = trueState2(6) - vxForDesign2*kappaPath2;
        errState2 = [e1_2; e2_2; e3_2; e4_2];

        if isempty(cachedK2) || abs(vxForDesign2 - cachedVxForK2) > lqrGainCacheTol
            [A_lin2, B_lin2] = lateralErrorModel(veh, vxForDesign2);
            cachedK2 = computeLQRGainDARE(A_lin2, B_lin2, Qlqr, Rlqr, dt, lqrIters);
            cachedVxForK2 = vxForDesign2;
            cachedA_lin2 = A_lin2; cachedB_lin2 = B_lin2;
        end

        deltaFF2 = feedforwardSteering(kappaPath2, vxForDesign2, veh.L, veh.Kus);
        if strcmp(controllerType, 'mpc')
            Ad_mpc2 = eye(4) + cachedA_lin2*dt;
            Bd_mpc2 = cachedB_lin2*dt;
            mpcUwarm2 = solveLinearMPC(errState2, Ad_mpc2, Bd_mpc2, Qlqr, Rlqr, mpcHorizon, lims.maxSteer, mpcUwarm2);
            deltaFB2 = mpcUwarm2(1);
        else
            deltaFB2 = -cachedK2*errState2;
        end
        deltaCmd2 = deltaFF2 + deltaFB2;
        deltaCmd2 = max(min(deltaCmd2, lims.maxSteer), -lims.maxSteer);
        deltaCmd2 = rateLimitStep(deltaCmd2, prevDeltaCmd2, lims.maxSteerRate*dt);
        prevDeltaCmd2 = deltaCmd2;

        vTarget2 = speedPlanner(kappaPath2, lims.vmin, lims.vmax, lims.ayMax);
        vErr2 = vTarget2 - trueState2(4);
        speedIntegral2 = max(min(speedIntegral2 + vErr2*dt, 5), -5);
        axCmd2 = Kp_v*vErr2 + Ki_v*speedIntegral2;
        axCmd2 = max(min(axCmd2, lims.maxAccel), -lims.maxDecel);

        inputCmd2 = [deltaCmd2; axCmd2] + [procNoise.steer*randn; procNoise.accel*randn];
        inputCmd2(1) = max(min(inputCmd2(1), lims.maxSteer), -lims.maxSteer);
        trueState2 = rk4Step(trueState2, inputCmd2, dt, veh);
        trailTrue2X(end+1) = trueState2(1); %#ok<AGROW>
        trailTrue2Y(end+1) = trueState2(2); %#ok<AGROW>
    end

    % --- lap timer: accumulate distance traveled and detect lap completion ---
    stepDist = hypot(trueState(1)-prevTrueXY(1), trueState(2)-prevTrueXY(2));
    distanceTraveled = distanceTraveled + stepDist;
    prevTrueXY = trueState(1:2);
    newLapCount = floor(distanceTraveled/trackTotalLength);
    if newLapCount > lapCount
        lapCount = newLapCount;
        completedLapTime = t - lastLapStartTime;
        if completedLapTime < bestLapTime
            bestLapTime = completedLapTime;
        end
        lastLapStartTime = t;
    end
    currentLapTime = t - lastLapStartTime;

    % Finish the current lap instead of cutting off mid-lap: once t
    % reaches tEnd, remember how many laps were done AT that instant, then
    % keep going (the outer preallocation already has room, see
    % maxLapFinishExtension) until lapCount increases beyond that --
    % i.e. until the lap that was in progress at the tEnd mark completes.
    % This is what was producing "the clock ran out mid-lap" before.
    % A FLAG, not an immediate break: this step still needs to finish
    % (sensors, estimation, rendering) so the frame where the car actually
    % crosses the finish line is the one that gets shown/recorded, not
    % skipped.
    if t >= tEnd
        if lapCountAtDeadline < 0
            lapCountAtDeadline = lapCount;
        end
        if lapCount > lapCountAtDeadline
            stopAfterThisStep = true;
        end
    end

    % ---------------------------------------------------------------
    % 5.4  SIMULATE NOISY SENSORS FROM THE TRUE STATE
    % ---------------------------------------------------------------
    imuMeas = simulateIMU(trueState, inputCmd, veh, noise);

    haveGps = (mod(step, gpsDecim) == 0);
    if haveGps
        gpsMeas = trueState(1:2) + noise.gpsPos*randn(2,1);
        lastGpsMeas = gpsMeas;
        gpsHistX(end+1) = gpsMeas(1); %#ok<AGROW>
        gpsHistY(end+1) = gpsMeas(2); %#ok<AGROW>
        if numel(gpsHistX) > maxGpsHist
            gpsHistX(1) = []; gpsHistY(1) = [];
        end
    end

    % ---------------------------------------------------------------
    % 5.5  EXTENDED KALMAN FILTER: PREDICT then UPDATE
    % ---------------------------------------------------------------
    [estState, P] = ekfPredict(estState, P, inputCmd, dt, veh, Qekf);

    if mod(step, imuDecim) == 0
        [estState, P] = ekfUpdateIMU(estState, P, imuMeas, inputCmd, veh, Rimu);
    end
    if haveGps
        [estState, P] = ekfUpdateGPS(estState, P, gpsMeas, Rgps);
    end

    if estimatorCompare
        % Parallel UKF, fed the IDENTICAL sensor measurements as the EKF
        % above, purely for a side-by-side accuracy comparison -- its
        % output is never used for control.
        [estStateUKF, Pukf] = ukfPredict(estStateUKF, Pukf, inputCmd, dt, veh, Qekf);
        if mod(step, imuDecim) == 0
            [estStateUKF, Pukf] = ukfUpdateIMU(estStateUKF, Pukf, imuMeas, inputCmd, veh, Rimu);
        end
        if haveGps
            [estStateUKF, Pukf] = ukfUpdateGPS(estStateUKF, Pukf, gpsMeas, Rgps);
        end
        logPosErrUKF(step) = hypot(trueState(1)-estStateUKF(1), trueState(2)-estStateUKF(2));
    end

    trailEstX(end+1) = estState(1); %#ok<AGROW>
    trailEstY(end+1) = estState(2); %#ok<AGROW>

    % ---------------------------------------------------------------
    % 5.6  LOGGING (cheap, done every step regardless of rendering)
    % ---------------------------------------------------------------
    logT(step)         = t;
    logVxTrue(step)    = trueState(4);
    logVxEst(step)     = estState(4);
    logVTarget(step)   = vTarget;
    logDeltaFF(step)   = rad2deg(deltaFF);
    logDeltaFB(step)   = rad2deg(deltaFB);
    logDeltaCmd(step)  = rad2deg(deltaCmd);
    logE1(step)        = e1;
    logPosErrEst(step) = hypot(trueState(1)-estState(1), trueState(2)-estState(2));
    if haveGps
        logPosErrGps(step) = hypot(trueState(1)-lastGpsMeas(1), trueState(2)-lastGpsMeas(2));
    elseif step > 1
        logPosErrGps(step) = logPosErrGps(step-1);
    end

    [alphaF_deg, alphaR_deg] = tireSlipAngles(trueState, inputCmd, veh);
    logAlphaF(step) = alphaF_deg;
    logAlphaR(step) = alphaR_deg;
    trueDotNow = vehicleDynamicsCT(trueState, inputCmd, veh);
    logAyTrue(step) = trueDotNow(5) + trueState(4)*trueState(6);   % vy_dot + vx*r
    logAxTrue(step) = trueDotNow(4);                               % longitudinal accel
    logPtrace(step) = trace(P);

    % --- skid marks: drop a mark at the rear axle whenever slip exceeds
    %     skidThreshold, capped to the most recent maxSkidPoints ---
    if abs(alphaF_deg) > skidThreshold || abs(alphaR_deg) > skidThreshold
        RrotSkid = [cos(trueState(3)) -sin(trueState(3)); sin(trueState(3)) cos(trueState(3))];
        rearWorld = RrotSkid*[-veh.lr; 0] + trueState(1:2);
        skidX(end+1) = rearWorld(1); %#ok<AGROW>
        skidY(end+1) = rearWorld(2); %#ok<AGROW>
        if numel(skidX) > maxSkidPoints
            skidX(1) = []; skidY(1) = [];
        end
    end

    % ---------------------------------------------------------------
    % Render every liveStepsPerFrame physics steps -- a single adaptive
    % cadence used regardless of exportVideo now (see TIMING FIX #10
    % above). Video frames, when enabled, are captured on these exact
    % same events, at zero extra rendering cost.
    % ---------------------------------------------------------------
    readyToRender = (mod(step, liveStepsPerFrame) == 0) || step == 1;

    if readyToRender
        % -------------------------------------------------------
        % 5.8  PREDICTIVE ROLLOUT from the CURRENT ESTIMATE.
        %      Now computed ONLY on frames that actually render --
        %      this alone removes the majority of the wasted work
        %      that was causing the lag, since it used to run on
        %      every one of the ~3500 physics steps regardless of
        %      whether that step was ever drawn.
        % -------------------------------------------------------
        predX = zeros(1,Npred); predY = zeros(1,Npred);
        rollState = estState;
        rollIntegral = speedIntegral;
        rollCachedK = [];
        rollCachedVx = NaN;
        rollIdx = mainIdx;         % start the rollout's search right where
                                    % the real controller currently is
        rollPrevDelta = deltaCmd;  % continue from the actual last steering command
        for k = 1:Npred
            vxr = max(rollState(4), 2.0);
            [rollIdx, psiP, kappaP, e1r] = projectToPath(rollState(1), rollState(2), ...
                                                trackX, trackY, trackKappa, trackPsi, rollIdx);
            e3r = wrapAnglePi(rollState(3) - psiP);
            e2r = rollState(4)*sin(e3r) + rollState(5)*cos(e3r);
            e4r = rollState(6) - vxr*kappaP;

            if isempty(rollCachedK) || abs(vxr - rollCachedVx) > lqrGainCacheTol
                [Ar, Br] = lateralErrorModel(veh, vxr);
                rollCachedK = computeLQRGainDARE(Ar, Br, Qlqr, Rlqr, dtPred, 15);
                rollCachedVx = vxr;
            end

            dFF = feedforwardSteering(kappaP, vxr, veh.L, veh.Kus);
            dFB = -rollCachedK*[e1r; e2r; e3r; e4r];
            dCmd = max(min(dFF+dFB, lims.maxSteer), -lims.maxSteer);
            dCmd = rateLimitStep(dCmd, rollPrevDelta, lims.maxSteerRate*dtPred);
            rollPrevDelta = dCmd;

            vTgtR = speedPlanner(kappaP, lims.vmin, lims.vmax, lims.ayMax);
            rollIntegral = rollIntegral + (vTgtR - rollState(4))*dtPred;
            aCmdR = max(min(Kp_v*(vTgtR-rollState(4)) + Ki_v*rollIntegral, lims.maxAccel), -lims.maxDecel);

            rollState = rk4Step(rollState, [dCmd; aCmdR], dtPred, veh);
            predX(k) = rollState(1); predY(k) = rollState(2);
        end

        % -------------------------------------------------------
        % 5.8b  Vehicle 2's own predictive rollout + collision check
        %       against vehicle 1's rollout computed just above.
        % -------------------------------------------------------
        if numVehicles >= 2
            predX2 = zeros(1,Npred); predY2 = zeros(1,Npred);
            rollState2 = trueState2;   % no EKF for the ghost car -- true state directly
            rollIntegral2 = speedIntegral2;
            rollCachedK2 = []; rollCachedVx2 = NaN;
            rollIdx2 = mainIdx2;
            rollPrevDelta2 = deltaCmd2;
            for k = 1:Npred
                vxr2 = max(rollState2(4), 2.0);
                [rollIdx2, psiP2, kappaP2, e1r2] = projectToPath(rollState2(1), rollState2(2), ...
                                                    trackX, trackY, trackKappa, trackPsi, rollIdx2);
                e3r2 = wrapAnglePi(rollState2(3) - psiP2);
                e2r2 = rollState2(4)*sin(e3r2) + rollState2(5)*cos(e3r2);
                e4r2 = rollState2(6) - vxr2*kappaP2;

                if isempty(rollCachedK2) || abs(vxr2 - rollCachedVx2) > lqrGainCacheTol
                    [Ar2, Br2] = lateralErrorModel(veh, vxr2);
                    rollCachedK2 = computeLQRGainDARE(Ar2, Br2, Qlqr, Rlqr, dtPred, 15);
                    rollCachedVx2 = vxr2;
                end

                dFF2 = feedforwardSteering(kappaP2, vxr2, veh.L, veh.Kus);
                dFB2 = -rollCachedK2*[e1r2; e2r2; e3r2; e4r2];
                dCmd2 = max(min(dFF2+dFB2, lims.maxSteer), -lims.maxSteer);
                dCmd2 = rateLimitStep(dCmd2, rollPrevDelta2, lims.maxSteerRate*dtPred);
                rollPrevDelta2 = dCmd2;

                vTgtR2 = speedPlanner(kappaP2, lims.vmin, lims.vmax, lims.ayMax);
                rollIntegral2 = rollIntegral2 + (vTgtR2 - rollState2(4))*dtPred;
                aCmdR2 = max(min(Kp_v*(vTgtR2-rollState2(4)) + Ki_v*rollIntegral2, lims.maxAccel), -lims.maxDecel);

                rollState2 = rk4Step(rollState2, [dCmd2; aCmdR2], dtPred, veh);
                predX2(k) = rollState2(1); predY2(k) = rollState2(2);
            end

            % pairwise minimum distance between the two predicted horizons
            minDist = inf;
            for ii = 1:Npred
                for jj = 1:Npred
                    dd = hypot(predX(ii)-predX2(jj), predY(ii)-predY2(jj));
                    if dd < minDist; minDist = dd; end
                end
            end
            collisionWarning = minDist < collisionThreshold;
        end

        % -------------------------------------------------------
        % 5.9  UPDATE DASHBOARD GRAPHICS
        % -------------------------------------------------------

        % --- persistent dim base trail: the FULL path driven so far,
        %     always visible (this is what fixes "lines not visible") ---
        set(hTrueTrailBase, 'XData', trailTrueX, 'YData', trailTrueY);
        set(hEstTrailBase,  'XData', trailEstX,  'YData', trailEstY);

        % --- bright fading highlight: only the most recent trailMaxLen
        %     points, with an alpha ramp from dim (oldest) to fully solid
        %     (newest) -- floor raised to 0.35 so even the tail of the
        %     highlight stays clearly visible rather than vanishing.
        %     IMPORTANT: Faces is built as (N-1) separate 2-vertex
        %     segments [1 2; 2 3; 3 4; ...], NOT a single N-vertex face.
        %     A single multi-vertex patch face is a POLYGON and always
        %     auto-closes with an edge from the last vertex back to the
        %     first -- that stray "closing" edge is exactly what caused
        %     the long diagonal line from the oldest trail point straight
        %     to the car. A 2-vertex face has no meaningful "closing"
        %     edge (it's just the same segment traced twice), so chaining
        %     N-1 of them draws a clean open polyline with no jump. ---
        nTrailT = min(numel(trailTrueX), trailMaxLen);
        if nTrailT >= 2
            xs = trailTrueX(end-nTrailT+1:end); ys = trailTrueY(end-nTrailT+1:end);
            segFaces = [(1:nTrailT-1)', (2:nTrailT)'];
            set(hTrueTrail, 'Faces', segFaces, 'Vertices', [xs(:) ys(:)], ...
                'FaceVertexAlphaData', linspace(0.35,1,nTrailT)');
        end
        nTrailE = min(numel(trailEstX), trailMaxLen);
        if nTrailE >= 2
            xsE = trailEstX(end-nTrailE+1:end); ysE = trailEstY(end-nTrailE+1:end);
            segFacesE = [(1:nTrailE-1)', (2:nTrailE)'];
            set(hEstTrail, 'Faces', segFacesE, 'Vertices', [xsE(:) ysE(:)], ...
                'FaceVertexAlphaData', linspace(0.35,1,nTrailE)');
        end

        set(hPred,  'XData', [estState(1) predX], 'YData', [estState(2) predY]);
        set(hGps,   'XData', gpsHistX, 'YData', gpsHistY);
        set(hSkid,  'XData', skidX, 'YData', skidY);

        if showConfidenceEllipse
            [ellX, ellY] = confidenceEllipsePoints(estState(1), estState(2), P(1:2,1:2), 2, 40);
            set(hEllipse, 'XData', ellX, 'YData', ellY);
        end

        if showMinimap
            set(hMinimapCar, 'XData', trueState(1), 'YData', trueState(2));
            set(hMinimapCarGlow, 'XData', trueState(1), 'YData', trueState(2));
        end

        Rrot = [cos(trueState(3)) -sin(trueState(3)); sin(trueState(3)) cos(trueState(3))];
        worldBody   = Rrot*carBodyLocal  + trueState(1:2);
        worldCabin  = Rrot*carCabinLocal + trueState(1:2);
        worldWheels = (Rrot*carWheelsLocalVertices')' + trueState(1:2)';
        set(hCarBody,  'XData', worldBody(1,:),  'YData', worldBody(2,:),  'FaceColor', car1Color);
        set(hCarCabin, 'XData', worldCabin(1,:), 'YData', worldCabin(2,:));
        set(hCarWheels,'Vertices', worldWheels);

        if numVehicles >= 2
            set(hTrue2TrailBase, 'XData', trailTrue2X, 'YData', trailTrue2Y);
            set(hPred2, 'XData', [trueState2(1) predX2], 'YData', [trueState2(2) predY2]);

            Rrot2 = [cos(trueState2(3)) -sin(trueState2(3)); sin(trueState2(3)) cos(trueState2(3))];
            worldBody2   = Rrot2*carBodyLocal  + trueState2(1:2);
            worldCabin2  = Rrot2*carCabinLocal + trueState2(1:2);
            worldWheels2 = (Rrot2*carWheelsLocalVertices')' + trueState2(1:2)';
            set(hCar2Cabin, 'XData', worldCabin2(1,:), 'YData', worldCabin2(2,:));
            set(hCar2Wheels,'Vertices', worldWheels2);

            % smooth sinusoidal "ghost" blink -- pulses opacity between
            % ~0.30 and ~0.95 continuously, an intuitive visual cue that
            % this is the ghost/predicted car without needing a label
            car2Alpha = 0.30 + 0.65*(0.5 + 0.5*sin(2*pi*t/blinkPeriod));
            set(hCar2Cabin, 'FaceAlpha', min(0.85, car2Alpha+0.1));

            if collisionWarning
                set(hCar2Body, 'XData', worldBody2(1,:), 'YData', worldBody2(2,:), ...
                    'FaceColor',[1 0.1 0.1], 'FaceAlpha', car2Alpha);
                set(txtCollision, 'Visible','on');
            else
                set(hCar2Body, 'XData', worldBody2(1,:), 'YData', worldBody2(2,:), ...
                    'FaceColor', car2Color, 'FaceAlpha', car2Alpha);
                set(txtCollision, 'Visible','off');
            end
        end

        % --- camera mode: pan/zoom the main view onto the car if enabled ---
        if strcmp(cameraMode, 'follow')
            xlim(axMain, [trueState(1)-followSpan, trueState(1)+followSpan]);
            ylim(axMain, [trueState(2)-followSpan, trueState(2)+followSpan]);
        end

        % --- HUD gauges ---
        speedFrac = max(min(trueState(4)/(lims.vmax*1.15), 1), 0);
        angSpeed = pi*(1 - speedFrac);
        set(hSpeedoNeedle, 'XData',[0 0.85*cos(angSpeed)], 'YData',[0 0.85*sin(angSpeed)]);
        set(txtSpeedoVal, 'String', sprintf('%.1f m/s', trueState(4)));

        steerFrac = max(min(deltaCmd/lims.maxSteer, 1), -1);
        angSteer = pi/2 - steerFrac*(pi/2);
        set(hSteerNeedle, 'XData',[0 0.85*cos(angSteer)], 'YData',[0 0.85*sin(angSteer)]);
        set(txtSteerVal, 'String', sprintf('%.1f^{\\circ}', rad2deg(deltaCmd)));

        % RADIAL clamp, not independent per-axis clamp: the previous version
        % clamped gx and gy separately to [-1,1], but a point like (1,1) has
        % radius sqrt(2) -- outside the outer circle -- even though each
        % individual coordinate was technically "in range". This is what let
        % the dot render outside the gauge during hard combined
        % braking+cornering. Clamping the combined vector's RADIUS instead
        % geometrically guarantees the dot can never leave the circle.
        gx = logAxTrue(step)/gMeterScale;
        gy = logAyTrue(step)/gMeterScale;
        gRadius = hypot(gx, gy);
        if gRadius > 1
            gx = gx/gRadius;
            gy = gy/gRadius;
        end
        set(hGdot, 'XData', gx, 'YData', gy);
        set(txtGVal, 'String', sprintf('%.2f g', hypot(logAxTrue(step),logAyTrue(step))/9.81));

        % --- readout text: t / Lap / Best now live in the dedicated LAP
        %     and CLOCK digital panels below, so they're no longer
        %     duplicated up here -- one source of truth for each value. ---
        set(txtMain, 'String', sprintf('$v=%.1f$ m/s $\\quad e_1=%.2f$ m $\\quad$ pos.\\ err.$=%.2f$ m', ...
            trueState(4), e1, logPosErrEst(step)));

        % --- LAP digital panel ---
        set(txtLapNum, 'String', sprintf('LAP %d', lapCount+1));
        if isinf(bestLapTime)
            set(txtLapBest, 'String', 'best   --');
        else
            set(txtLapBest, 'String', sprintf('best %5.1fs', bestLapTime));
        end
        set(txtLapCur, 'String', sprintf('%.1f s', currentLapTime));
        lapFrac = mod(distanceTraveled, trackTotalLength)/trackTotalLength;
        set(hLapProgress, 'Position',[-0.95 0.13 1.9*lapFrac 0.10]);

        % --- CLOCK digital panel: real elapsed wall-clock time, which at
        %     playbackSpeed=1.0 always exactly equals the simulated time t ---
        clockMin = floor(t/60);
        clockSec = mod(t,60);
        set(txtClock, 'String', sprintf('%02d:%04.1f', clockMin, clockSec));
        raceFrac = min(t/tEnd, 1);
        set(hClockProgress, 'Position',[-0.95 -0.02 1.9*raceFrac 0.09]);
        dotPulse = 0.35 + 0.65*abs(sin(pi*t/1.2));   % ~1.2s pulse period, tied to sim time
        set(hClockDot, 'MarkerFaceAlpha', dotPulse);

        set(hVxTrue, 'XData', logT(1:step), 'YData', logVxTrue(1:step));
        set(hVxEst,  'XData', logT(1:step), 'YData', logVxEst(1:step));
        set(hVxTgt,  'XData', logT(1:step), 'YData', logVTarget(1:step));

        set(hDeltaFF,  'XData', logT(1:step), 'YData', logDeltaFF(1:step));
        set(hDeltaFB,  'XData', logT(1:step), 'YData', logDeltaFB(1:step));
        set(hDeltaCmd, 'XData', logT(1:step), 'YData', logDeltaCmd(1:step));

        set(hE1,     'XData', logT(1:step), 'YData', logE1(1:step));
        set(hPosEst, 'XData', logT(1:step), 'YData', logPosErrEst(1:step));
        set(hPosGps, 'XData', logT(1:step), 'YData', logPosErrGps(1:step));

        renderT0 = toc(simTimer);
        % Always a full forced drawnow, not 'limitrate'. 'limitrate' was
        % tried for the live (non-export) path as a lighter update path,
        % but its internal partial-update/culling heuristic can
        % inconsistently skip redrawing elements that aren't "primary"
        % triggers of that particular update -- which is the likely cause
        % of the minimap vanishing (it's only touched by one small
        % set() call per frame) and the subplot legends rendering
        % garbled/overlapping (repositioning a legend needs a full layout
        % pass, which 'limitrate' can skip). Reverted for both branches:
        % correctness over a small speed difference.
        drawnow;
        % TIMING FIX #12: renderDuration is measured HERE, immediately
        % after drawnow and BEFORE video capture -- previously this
        % measurement included getframe()+buffer-append, so video-capture
        % overhead (and its occasional reallocation spikes) was directly
        % corrupting the adaptive pacing below, making the LIVE view
        % itself patchy whenever exportVideo was on, not just the file.
        renderDuration = toc(simTimer) - renderT0;

        if exportVideo && videoFrameCount < videoMaxBufferedFrames
            videoFrameCount = videoFrameCount + 1;
            videoFrameBuffer{videoFrameCount} = getframe(fig);
            videoFrameTimes(videoFrameCount)  = t;
        end

        % Real-time throttle: if the sim is running AHEAD of real
        % elapsed time (this machine can comfortably keep up), pause
        % just long enough to bring it back to playbackSpeed*realtime.
        % Deliberately one-directional -- if the sim is BEHIND (this
        % machine can't keep up), we never try to force a catch-up;
        % that asymmetry is exactly what fixes the old "perpetual
        % slow-motion" bug, since forcing catch-up when the actual
        % bottleneck is computation (not rendering) is structurally
        % impossible and only made things look more broken. Applies
        % regardless of exportVideo now -- video capture piggybacks on
        % whatever pace this settles into, it doesn't dictate it.
        if playbackSpeed > 0
            aheadBy = t/playbackSpeed - toc(simTimer);
            if aheadBy > 0
                pause(min(aheadBy, 0.25));
            end
        end

        % TIMING FIX #13 (still-patchy motion): renderDuration from a
        % SINGLE render is noisy (OS scheduling, background processes, an
        % occasional GC pause) even with heavy smoothing on what's done
        % with it afterward -- one outlier sample can still visibly kick
        % the cadence. Filtered through a small rolling MEDIAN first
        % (robust to a single spike/dip in a way a moving average isn't),
        % before that heavier smoothing+deadband is even applied.
        renderDurationHistory(mod(renderDurationHistIdx, renderDurationHistN) + 1) = renderDuration;
        renderDurationHistIdx = renderDurationHistIdx + 1;
        nHistValid = min(renderDurationHistIdx, renderDurationHistN);
        medianRenderDuration = median(renderDurationHistory(1:nHistValid));

        % TIMING FIX #8 (adaptive live pacing): rather than relying on
        % a manually-guessed liveTargetFPS, self-tune the number of
        % physics steps run between renders so the SIMULATED time
        % between renders tracks how long a render actually just took
        % in real time -- if this machine renders in 0.3s, we aim for
        % ~0.3s of sim-time per frame; if it renders in 2s, we aim for
        % ~2s of sim-time per frame. Smoothed (not applied instantly)
        % so one unusually slow/fast frame (e.g. first-frame JIT/driver
        % warmup) doesn't cause a big cadence swing on its own. Also
        % scaled by playbackSpeed -- at 1.5x we deliberately want MORE
        % sim-time to elapse per render (renderDuration*playbackSpeed),
        % consistent with the throttle above targeting t/playbackSpeed.
        targetStepsPerFrame = max(1, round(medianRenderDuration*playbackSpeed/dt));
        renderEventCount = renderEventCount + 1;
        % TIMING FIX #14 (slow start): the heavy smoothing/deadband below
        % is deliberately conservative for long-run STABILITY, but that
        % means it also takes a while to climb out of the initial
        % liveTargetFPS guess -- and the first few seconds of any run are
        % exactly what every screenshot and video in this project has
        % captured, so that slow climb was reading as "still too slow"
        % even on runs that would have settled in fine. For the first 6
        % render events, jump straight to the measured target (fast,
        % un-smoothed) to find the right ballpark almost immediately;
        % only after that does it switch to the stable, deadbanded
        % behavior used from then on.
        if renderEventCount <= 6
            liveStepsPerFrame = targetStepsPerFrame;
        elseif targetStepsPerFrame > 1.25*liveStepsPerFrame || targetStepsPerFrame < 0.75*liveStepsPerFrame
            % Heavily smoothed (0.75 old / 0.25 new) plus a deadband that
            % ignores small fluctuations entirely -- a single render's
            % timing is noisy (OS scheduling, background processes,
            % garbage collection, etc.), and reacting strongly to that
            % noise is what was making the cadence visibly hunt up and
            % down, producing the "moving in patches" look.
            liveStepsPerFrame = round(0.75*liveStepsPerFrame + 0.25*targetStepsPerFrame);
        end
        liveStepsPerFrame = min(max(liveStepsPerFrame, 1), 3000);

        % Live, honest readout of the ACTUAL achieved playback rate
        % (not a hardcoded "1.0x" claim) -- if this machine can't
        % sustain real-time for this scene, the panel will say so
        % (e.g. "0.4x realtime") instead of silently appearing stuck.
        % Accurate content to bake into the video too, if recording.
        achievedRate = t/max(toc(simTimer), 1e-6);
        set(txtClockSub, 'String', sprintf('%.2fx realtime', achievedRate));

        if exportVideo && mod(step, max(1,round(nStepsNominal/10))) < liveStepsPerFrame
            fprintf('  Captured %d video frames so far (t=%.1fs, %.0f%% of target)...\n', ...
                    videoFrameCount, t, min(100, 100*step/nStepsNominal));
        end

        lastRenderWallTime = toc(simTimer);
        if ~ishandle(fig); break; end
    end

    if stopAfterThisStep
        break;   % the lap in progress when tEnd was reached has now
                 % finished, and this step's frame has already been fully
                 % processed/rendered above -- stop here instead of
                 % running the full padded nSteps.
    end
end

if exportVideo
    % All rendering is already done -- this is just encoding
    % already-captured frames to disk, which is fast.
    if videoFrameCount == 0
        warning('exportVideo was on, but no frames were captured -- nothing to save.');
    else
        videoOutFile = fullfile(scriptOutDir, 'av_simulation.mp4');
        vidWriter = VideoWriter(videoOutFile, 'MPEG-4');
        % TIMING FIX #11 (fast/patchy video): writing captured frames
        % back-to-back at a single constant rate (frameCount/totalTime)
        % got the TOTAL duration right but not the PACING within it --
        % the adaptive live cadence captures frames at UNEVEN simulated-
        % time spacing (dense right after it speeds up, sparse right
        % after it slows down), and playing those back at one constant
        % rate stretches the dense patches into slow motion and
        % compresses the sparse patches into a fast-forward jump. Fixed
        % by resampling onto an EVENLY-SPACED timeline at a fixed output
        % rate: for each output frame's target simulated time, pick the
        % nearest captured frame at or before it (duplicating a frame
        % when the live capture was sparser than the output rate needs).
        % Every output frame now represents an equal slice of simulated
        % time, so playback speed is uniform throughout -- the tradeoff
        % is that a genuinely sparse stretch holds on one frame slightly
        % longer rather than jumping, which reads as a brief pause, not
        % a wrong speed.
        outputFPS = 20;
        nCaptured = videoFrameCount;
        finalSimT = videoFrameTimes(nCaptured);
        nOutFrames = max(1, round(finalSimT*outputFPS));
        vidWriter.FrameRate = outputFPS;
        open(vidWriter);
        srcIdx = 1;
        for k = 1:nOutFrames
            tTarget = (k-1)/outputFPS;
            while srcIdx < nCaptured && videoFrameTimes(srcIdx+1) <= tTarget
                srcIdx = srcIdx + 1;
            end
            writeVideo(vidWriter, videoFrameBuffer{srcIdx});
        end
        close(vidWriter);
        fprintf('Video saved: %s\n', videoOutFile);
        fprintf('  %d frames captured live, resampled to %d frames at a steady %d fps -> video duration = %.1f s (target was tEnd = %.1f s)\n', ...
                nCaptured, nOutFrames, outputFPS, nOutFrames/outputFPS, tEnd);
    end
end

%% ======================= SECTION 6: FINAL SUMMARY ========================
% Wrapped in try/catch like Sections 7-10 below -- previously, an error
% HERE would still halt everything downstream before those sections' own
% protection ever got a chance to run, which defeated the point of
% isolating them individually.
try
validIdx = ~isnan(logE1);
% BUGFIX: rmsPosGps used to be computed over this SAME validIdx mask,
% which is only about logE1's validity, not logPosErrGps's own. Before
% the very first GPS fix arrives, logPosErrGps(step) is still its
% preallocated NaN (nothing has set it yet), while logE1(step) is
% already valid (it's computed every step regardless of sensors) -- so
% validIdx included those early indices, mean() saw the leftover NaNs
% mixed in, and rmsPosGps (and everything downstream of it, like the
% EKF-improvement percentage) came out as NaN every single run. Each
% RMS metric now uses a mask matching its OWN array's validity.
validGpsIdx = validIdx & ~isnan(logPosErrGps);
rmsE1      = sqrt(mean(logE1(validIdx).^2));
rmsPosEst  = sqrt(mean(logPosErrEst(validIdx).^2));
rmsPosGps  = sqrt(mean(logPosErrGps(validGpsIdx).^2));

fprintf('\n=== Simulation complete ===\n');
fprintf('RMS cross-track error            : %.3f m\n', rmsE1);
fprintf('RMS EKF position estimation error: %.3f m\n', rmsPosEst);
fprintf('RMS raw GPS position error       : %.3f m\n', rmsPosGps);
fprintf('EKF error reduction vs raw GPS    : %.1f%%\n', 100*(1 - rmsPosEst/rmsPosGps));

if estimatorCompare
    validUKF = ~isnan(logPosErrUKF);
    rmsPosUKF = sqrt(mean(logPosErrUKF(validUKF).^2));
    fprintf('\n--- EKF vs UKF comparison (identical sensors, side-by-side) ---\n');
    fprintf('RMS EKF position error : %.3f m\n', rmsPosEst);
    fprintf('RMS UKF position error : %.3f m\n', rmsPosUKF);
    if rmsPosUKF < rmsPosEst
        fprintf('-> UKF was %.1f%% more accurate than the EKF on this run.\n', 100*(1-rmsPosUKF/rmsPosEst));
    else
        fprintf('-> EKF was %.1f%% more accurate than the UKF on this run.\n', 100*(1-rmsPosEst/rmsPosUKF));
    end
end

fprintf('\n--- Lap timing ---\n');
fprintf('Laps completed : %d\n', lapCount);
if isinf(bestLapTime)
    fprintf('Best lap time  : -- (no full lap completed)\n');
else
    fprintf('Best lap time  : %.2f s\n', bestLapTime);
end
fprintf('Skid-mark points logged (|slip| > %.1f deg): %d\n', skidThreshold, numel(skidX));

% NEW FEATURE: control-effort summary. The LQR weight-sensitivity sweep
% (Section 10) already demonstrates the classical "tighter tracking costs
% more steering effort" trade-off across many R values -- this reports
% the SAME kind of number, but for the actual weights used THIS run, so
% the headline error metrics above are never read in isolation from what
% it cost in steering activity to achieve them. logDeltaCmd/logDeltaFB
% are already logged every step during the main loop; this only
% aggregates data that already exists, no new state.
rmsSteerCmd  = sqrt(mean(logDeltaCmd(validIdx).^2));
rmsSteerRate = sqrt(mean(diff(logDeltaCmd(validIdx)).^2)) / dt;
fprintf('\n--- Control effort (this run''s Q=%s, R=%g) ---\n', mat2str(diag(Qlqr)'), Rlqr);
fprintf('RMS commanded steering angle : %.2f deg\n', rmsSteerCmd);
fprintf('RMS steering rate            : %.2f deg/s\n', rmsSteerRate);

catch summaryErr
    warning('Section 6 (final summary) failed: %s', summaryErr.message);
end


%% ================ SECTION 6B: MACHINE-READABLE RESULTS EXPORT =============
% NEW FEATURE: writes this run's key numbers to a small JSON file
% alongside the video/PDF/PPTX, so results can be diffed across runs,
% parsed by another script, or dropped straight into a portfolio/README
% without retyping numbers off a screenshot. Uses jsonencode (base
% MATLAB since R2016b; the 'PrettyPrint' option needs R2021a but this
% whole script already assumes R2020a+ for exportgraphics elsewhere) --
% no toolbox required either way. Wrapped in try/catch and placed in its
% own section so a failure here can't affect the video/PDF/PPTX exports
% that follow, and vice versa.
try
results = struct();
results.generatedAt              = datestr(now, 'yyyy-mm-dd HH:MM:SS');
results.simulatedDuration_s      = t;
results.targetDuration_s         = tEnd;
results.lapsCompleted            = lapCount;
results.bestLapTime_s            = ternary(isinf(bestLapTime), [], bestLapTime);
results.rmsCrossTrackError_m     = rmsE1;
results.rmsEkfPositionError_m    = rmsPosEst;
results.rmsRawGpsPositionError_m = rmsPosGps;
results.ekfImprovementPct        = 100*(1 - rmsPosEst/rmsPosGps);
results.rmsCommandedSteering_deg = rmsSteerCmd;
results.rmsSteeringRate_degps    = rmsSteerRate;
results.tireModel                = veh.tireModel;
results.controllerType           = controllerType;
results.vehicleMass_kg           = veh.m;
results.wheelbase_m              = veh.L;
results.integrationStep_s        = dt;
results.gpsNoiseStd_m            = noise.gpsPos;
results.imuYawNoiseStd_radps     = noise.imuYaw;
results.imuAccelNoiseStd_mps2    = noise.imuAccel;
results.lqrQ                     = diag(Qlqr)';
results.lqrR                     = Rlqr;

jsonFile = fullfile(scriptOutDir, 'av_simulation_results.json');
try
    jsonText = jsonencode(results, 'PrettyPrint', true);
catch
    jsonText = jsonencode(results);   % older MATLAB without PrettyPrint support
end
writeTextFile(jsonFile, jsonText);
fprintf('Machine-readable results saved: %s\n', jsonFile);
catch jsonErr
    warning('Results JSON export failed (%s).', jsonErr.message);
end

%% ======================= SECTION 7: STATIC DIAGNOSTIC FIGURE ==============
% A second, non-animated figure summarizing tire behavior and estimator
% convergence -- the kind of plots that would appear in a controls report.
% Wrapped in try/catch: previously, an error anywhere in this section (or
% Section 9/10 below) would silently halt the ENTIRE rest of the script,
% which is why sometimes only one extra window (or none) appeared instead
% of the usual four, and the PDF report could silently fail to generate
% too -- each diagnostic section now stands on its own.
try
figDiag = figure('Color','k','Position',[80 60 1100 750]);

subplot(2,2,1);
plot(logT, logAlphaF, 'Color',[0.3 0.9 1.0], 'LineWidth',1.1); hold on;
plot(logT, logAlphaR, 'Color',[1 0.55 0.2], 'LineWidth',1.1);
ax1 = gca; set(ax1,'Color','k','XColor','w','YColor','w'); grid on; ax1.GridColor = [0.3 0.3 0.3];
title('Tire slip angles','Color','w'); xlabel('time [s]','Color','w'); ylabel('\alpha [deg]','Color','w');
legend({'front \alpha_f','rear \alpha_r'},'TextColor','w','Color','none','EdgeColor',[0.4 0.4 0.4]);

subplot(2,2,2);
plot(logT, logAyTrue, 'Color',[1 0.85 0.1], 'LineWidth',1.1);
set(gca,'Color','k','XColor','w','YColor','w'); grid on;
title('True lateral acceleration','Color','w'); xlabel('time [s]','Color','w'); ylabel('a_y [m/s^2]','Color','w');
yline(lims.ayMax,'--','Color',[1 0.3 0.3]); yline(-lims.ayMax,'--','Color',[1 0.3 0.3]);

subplot(2,2,3);
semilogy(logT, logPtrace, 'Color',[0.9 0.3 1.0], 'LineWidth',1.2);
set(gca,'Color','k','XColor','w','YColor','w'); grid on;
title('EKF covariance trace (uncertainty shrinking)','Color','w');
xlabel('time [s]','Color','w'); ylabel('trace(P)  (log scale)','Color','w');

subplot(2,2,4);
plot(logT, logPosErrEst, 'Color',[0.9 0.3 1.0], 'LineWidth',1.2); hold on;
plot(logT, logPosErrGps, ':', 'Color',[0.7 0.7 0.7], 'LineWidth',1.1);
set(gca,'Color','k','XColor','w','YColor','w'); grid on;
title('Position error: EKF vs raw GPS','Color','w');
xlabel('time [s]','Color','w'); ylabel('error [m]','Color','w');
legend({'EKF estimate','raw GPS'},'TextColor','w','Color','none','EdgeColor',[0.4 0.4 0.4]);

sgtitle(figDiag, 'Estimator & Vehicle Diagnostics', 'Color','w','FontSize',13);
catch diagErr
    warning('Section 7 (diagnostic figure) failed: %s', diagErr.message);
end

%% ======================= SECTION 8: NO-EKF BASELINE COMPARISON ============
% To demonstrate why sensor fusion matters, the SAME control law is now
% driven directly off the raw, noisy, low-rate GPS position (with heading
% integrated open-loop from the noisy gyro and speed integrated open-loop
% from the commanded acceleration) instead of the EKF estimate. This is
% representative of a naive "GPS-only" controller with no filtering.
% Wrapped in try/catch, independent of Section 7 and Section 10, so a
% failure here doesn't prevent the sweep or PDF report from still running.
%
try
fprintf('\n=== Running NO-EKF baseline comparison (headless, no animation) ===\n');

tCompare = 40;   % shorter horizon is enough for a clear comparison
[rmsE1_ekf, rmsPos_ekf]  = runTrial(11, true,  veh, lims, trackX, trackY, trackKappa, trackPsi, ...
                                     noise, Qekf, Rgps, Rimu, Qlqr, Rlqr, lqrIters, dt, tCompare, gpsDecim);
[rmsE1_raw, rmsPos_raw]  = runTrial(11, false, veh, lims, trackX, trackY, trackKappa, trackPsi, ...
                                     noise, Qekf, Rgps, Rimu, Qlqr, Rlqr, lqrIters, dt, tCompare, gpsDecim);

fprintf('With EKF     : RMS cross-track error = %.3f m,  RMS position error = %.3f m\n', rmsE1_ekf, rmsPos_ekf);
fprintf('Raw GPS only : RMS cross-track error = %.3f m,  RMS position error = %.3f m\n', rmsE1_raw, rmsPos_raw);
fprintf('Cross-track error improvement from EKF: %.1f%%\n', 100*(1 - rmsE1_ekf/rmsE1_raw));
posImprovePct = 100*(1 - rmsPos_ekf/rmsPos_raw);
fprintf('Position error improvement from EKF   : %.1f%%\n', posImprovePct);
catch mcErr
    warning('Section 8 (baseline comparison) failed: %s', mcErr.message);
    tCompare = 40;      % fallback so Section 10 below (independent otherwise) still has a sane value if referenced
    posImprovePct = NaN; % fallback so Section 11B (PPTX) below can still run and just show "--"
end


%% ======================= SECTION 9: MONTE CARLO TRIALS =====================
% A single run (however carefully seeded) is one draw from a random
% process -- Section 8's baseline comparison could, in principle, have
% been a lucky or unlucky sample for either controller. Repeating the
% same EKF-vs-raw-GPS comparison across many independent random seeds
% turns that single anecdote into a distribution, so "the EKF wins"
% reflects consistent behavior rather than one favorable run.
%
% RESTORED: this section previously existed, produced a chart with a
% title and axes but no bars and no console error, and was deleted
% rather than left in a persistently broken state. The empty chart was
% only a downstream symptom -- the real bug was inside runTrial's
% no-EKF baseline path (see the BUGFIX comment there): it estimated
% speed by finite-differencing two noisy GPS fixes 0.1 s apart, which
% amplifies into roughly 8.5 m/s of noise on a 5-14 m/s signal. Fed
% into the quadratic feedforward term, that noise was enough to send
% the raw-GPS trials into full runaways (hundreds of meters of error)
% on effectively every trial -- a results array dominated by wildly
% inconsistent outliers, which is exactly the kind of data that can
% make a histogram/bar call silently draw nothing sensible. With that
% root cause fixed at the source, this section now runs cleanly and
% produces the intended distribution.
% Wrapped in try/catch, independent of Sections 8 and 10 -- a failure
% here doesn't prevent the sweep or PDF report from still running.
try
fprintf('\n=== Monte Carlo trials (randomized sensor-noise seeds) ===\n');

nMonteCarlo = 24;   % trials per controller -- enough for a stable-looking
                     % distribution without making this section the
                     % slowest part of the run
tMonteCarlo = 20;   % shorter horizon per trial, same reasoning as tCompare/tSweep
mcPosEKF = nan(1, nMonteCarlo);
mcPosRaw = nan(1, nMonteCarlo);

for i = 1:nMonteCarlo
    seedMC = 5000 + i;   % offset well clear of any other seed used in this script
    [~, mcPosEKF(i)] = runTrial(seedMC, true,  veh, lims, trackX, trackY, trackKappa, trackPsi, ...
                                 noise, Qekf, Rgps, Rimu, Qlqr, Rlqr, lqrIters, dt, tMonteCarlo, gpsDecim);
    [~, mcPosRaw(i)] = runTrial(seedMC, false, veh, lims, trackX, trackY, trackKappa, trackPsi, ...
                                 noise, Qekf, Rgps, Rimu, Qlqr, Rlqr, lqrIters, dt, tMonteCarlo, gpsDecim);
end

% Guard against any individual trial coming back NaN/Inf -- one bad draw
% should not take down the whole summary or plotting call (this is the
% same class of validity-mask lesson learned from the Section 6 fix).
validEKF = ~isnan(mcPosEKF) & ~isinf(mcPosEKF);
validRaw = ~isnan(mcPosRaw) & ~isinf(mcPosRaw);

if any(validEKF) && any(validRaw)
    meanEKF = mean(mcPosEKF(validEKF)); stdEKF = std(mcPosEKF(validEKF));
    meanRaw = mean(mcPosRaw(validRaw)); stdRaw = std(mcPosRaw(validRaw));
    fprintf('  EKF     : mean RMS position error = %.3f m  (std %.3f m, %d/%d valid trials)\n', ...
            meanEKF, stdEKF, nnz(validEKF), nMonteCarlo);
    fprintf('  Raw GPS : mean RMS position error = %.3f m  (std %.3f m, %d/%d valid trials)\n', ...
            meanRaw, stdRaw, nnz(validRaw), nMonteCarlo);
    bothValid = validEKF & validRaw;
    nBetter = nnz(mcPosEKF(bothValid) < mcPosRaw(bothValid));
    fprintf('  EKF was more accurate in %d / %d matched trials.\n', nBetter, nnz(bothValid));

    edges = linspace(0, max([mcPosEKF(validEKF), mcPosRaw(validRaw)])*1.05, 16);
    % This chart went through several rendering attempts before landing
    % here -- worth leaving the history in case it recurs. histogram()
    % with FaceAlpha rendered empty; histogram() without alpha plus a
    % stairs() display style also rendered empty; bar() also rendered
    % empty. A screenshot at that point was the turning point: switching
    % to histcounts() + stairs() + plot() markers rendered the MARKERS
    % correctly (which also proved the underlying data was fine all
    % along -- e.g. 7+17=24 EKF trials, tightly clustered as expected)
    % but the stairs() line itself still didn't render. The one figure
    % in this whole script nobody ever reported as blank is the LQR
    % sweep (figSweep below), which draws with plot(..., '-o', ...) --
    % marker and line in a single call. Reusing that exact call pattern
    % here, instead of stairs(), is what finally rendered correctly.
    countsEKF = histcounts(mcPosEKF(validEKF), edges);
    countsRaw = histcounts(mcPosRaw(validRaw), edges);
    binCenters = edges(1:end-1) + diff(edges)/2;
    fprintf('  Bin centers  : '); fprintf('%.3f ', binCenters); fprintf('\n');
    fprintf('  EKF counts   : '); fprintf('%d ', countsEKF); fprintf('\n');
    fprintf('  Raw GPS counts: '); fprintf('%d ', countsRaw); fprintf('\n');

    figMC = figure('Color','k', 'Position',[220 170 780 480]);
    axMC = axes(figMC); hold(axMC,'on');
    set(axMC,'Color','k','XColor','w','YColor','w');
    plot(axMC, binCenters, countsEKF, '-o', 'Color',[0.4 0.95 1.0], ...
         'MarkerFaceColor',[0.4 0.95 1.0], 'LineWidth',2.2, 'MarkerSize',5);
    plot(axMC, binCenters, countsRaw, '-o', 'Color',[0.95 0.4 0.4], ...
         'MarkerFaceColor',[0.95 0.4 0.4], 'LineWidth',2.2, 'MarkerSize',5);
    grid(axMC,'on'); axMC.GridColor = [0.3 0.3 0.3];
    xlabel(axMC, 'RMS position error per trial [m]', 'Color','w');
    ylabel(axMC, 'Number of trials', 'Color','w');
    title(axMC, sprintf('Monte Carlo: %d trials per controller, randomized sensor-noise seeds', nMonteCarlo), ...
          'Color','w', 'FontSize',11);
    legend(axMC, {'EKF','Raw GPS'}, 'TextColor','w', 'Color','k', 'EdgeColor',[0.4 0.4 0.4]);
else
    warning('Monte Carlo trials: too many invalid results to summarize (EKF valid=%d, raw valid=%d of %d).', ...
            nnz(validEKF), nnz(validRaw), nMonteCarlo);
end
catch mcTrialsErr
    warning('Section 9 (Monte Carlo trials) failed: %s', mcTrialsErr.message);
end


%% ======================= SECTION 10: LQR WEIGHT SENSITIVITY SWEEP =========
% Classical LQR tuning trade-off: increasing the steering-effort penalty R
% relative to the tracking-error penalty Q produces a smoother, gentler
% controller at the cost of larger tracking error. Sweeping R and plotting
% the resulting RMS cross-track error reproduces the textbook "tuning
% curve" that guides the Q/R weight selection in the first place.
% Wrapped in try/catch, independent of Sections 7-9 -- a failure here
% still lets the PDF report (Section 11) run afterward.
try
fprintf('\n=== LQR weight sensitivity sweep (R penalty on steering effort) ===\n');

Rsweep = [50, 150, 400, 700, 1500, 3000, 6000];
sweepE1 = zeros(size(Rsweep));
tSweep = 30;   % short horizon per point, this is just for a trend curve

for i = 1:numel(Rsweep)
    [sweepE1(i), ~] = runTrial(200+i, true, veh, lims, trackX, trackY, trackKappa, trackPsi, ...
                               noise, Qekf, Rgps, Rimu, Qlqr, Rsweep(i), lqrIters, dt, tSweep, gpsDecim);
    fprintf('  R = %6.0f  ->  RMS cross-track error = %.3f m\n', Rsweep(i), sweepE1(i));
end

figSweep = figure('Color','k','Position',[200 150 700 480]);
axSweep = axes(figSweep); hold(axSweep,'on');
set(axSweep,'Color','k','XColor','w','YColor','w','XScale','log');
plot(axSweep, Rsweep, sweepE1, '-o', 'Color',[1 0.85 0.1], ...
     'MarkerFaceColor',[1 0.85 0.1], 'LineWidth',1.6, 'MarkerSize',6);
grid(axSweep,'on'); axSweep.GridColor = [0.3 0.3 0.3];
xlabel(axSweep,'LQR steering-effort weight R (log scale)','Color','w');
ylabel(axSweep,'RMS cross-track error [m]','Color','w');
title(axSweep,'LQR tuning trade-off: tracking accuracy vs. steering effort', ...
      'Color','w','FontSize',11);
text(axSweep, Rsweep(1), sweepE1(1), '  aggressive  ', 'Color',[0.6 1 0.6], 'FontSize',9);
text(axSweep, Rsweep(end), sweepE1(end), '  gentle  ', 'Color',[1 0.6 0.6], ...
     'FontSize',9, 'HorizontalAlignment','right');
catch sweepErr
    warning('Section 10 (LQR sweep) failed: %s', sweepErr.message);
end

fprintf('\nAll analyses complete. Four figures were generated (unless a warning above says one failed):\n');
fprintf('  1. Live animated dashboard (track / speed / steering / error)\n');
fprintf('  2. Static diagnostics (tire slip, lateral accel, EKF covariance)\n');
fprintf('  3. Monte Carlo trials (EKF vs. raw-GPS RMS position error distribution)\n');
fprintf('  4. LQR weight sensitivity (tuning trade-off curve)\n');

%% ======================= SECTION 11: PDF REPORT EXPORT ====================
% Builds a proper multi-page report -- title page, an executive summary
% with methodology and visual stat cards, then one page per chart with
% its caption directly above it -- instead of just stapling raw
% screenshots together with no narrative at all.
if exportReport
    reportFile = fullfile(scriptOutDir, 'av_simulation_report.pdf');
    if exist(reportFile, 'file'); delete(reportFile); end
    try
        % REPORT FIX (washed-out colors): exportgraphics defaults to a
        % WHITE backdrop for alpha-blending unless told otherwise -- and
        % this whole dashboard is built from semi-transparent, dark-
        % themed elements (trails, low-grip zone tint, confidence
        % ellipse, etc.). Without an explicit BackgroundColor matching
        % the actual figure, every one of those blended toward white
        % instead of black, which is exactly why the exported report
        % looked pale/washed-out compared to the real on-screen dashboard.
        exportOpts = {'ContentType','image', 'BackgroundColor','black', 'Resolution',200};

        % REPORT FIX (inconsistent page sizes): every page in the report
        % now shares the SAME width and height. Previously the chart
        % pages were built at this width but the caption pages ahead of
        % them used a much shorter height (see renderReportChartPage's
        % header comment for why that was a problem on its own), so the
        % report's pages weren't even a consistent size. fig/figDiag/
        % figSweep/figMC are each temporarily resized to REPORT_PAGE_W
        % before export (preserving their own aspect ratio, so nothing
        % gets stretched/distorted) and restored to their original
        % on-screen size immediately after.
        REPORT_PAGE_W = 1200;
        REPORT_PAGE_H = round(REPORT_PAGE_W*1100/850);
        REPORT_PAGE_SZ = [REPORT_PAGE_W, REPORT_PAGE_H];

        % ---- Page 1: Title page ----
        lapStr = 'no full lap completed';
        if ~isinf(bestLapTime), lapStr = sprintf('%d laps completed, best %.2f s', lapCount, bestLapTime); end
        fTitle = renderReportTextPage({ ...
            struct('type','space'), struct('type','space'), struct('type','space'), ...
            struct('type','title',    'text','AUTONOMOUS VEHICLE'), ...
            struct('type','title',    'text','STATE ESTIMATION & OPTIMAL CONTROL'), ...
            struct('type','subtitle', 'text','Simulation Report'), ...
            struct('type','space'), struct('type','space'), ...
            struct('type','rule'), struct('type','space'), ...
            struct('type','subtitle', 'text','Dynamic Bicycle Model   |   Extended Kalman Filter   |   LQR Path Tracking'), ...
            struct('type','space'), struct('type','space'), struct('type','space'), ...
            struct('type','stat', 'text',sprintf('Report generated       : %s', datestr(now,'dd mmm yyyy, HH:MM'))), ...
            struct('type','stat', 'text',sprintf('Simulated duration     : %.1f s (target %.0f s)', t, tEnd)), ...
            struct('type','stat', 'text',sprintf('Lap summary            : %s', lapStr)), ...
            struct('type','stat', 'text',sprintf('Estimator              : Extended Kalman Filter%s', ...
                   ternary(estimatorCompare,' (+ parallel UKF comparison)',''))), ...
            }, REPORT_PAGE_SZ);
        exportgraphics(fTitle, reportFile, exportOpts{:});
        close(fTitle);

        % ---- Page 2: Executive summary / methodology / visual key results ----
        ekfImprovePct = 100*(1-rmsPosEst/rmsPosGps);
        bestLapStr = ternary(isinf(bestLapTime), '--', sprintf('%.2f s', bestLapTime));
        fSummary = renderReportTextPage({ ...
            struct('type','header', 'text','1.  Methodology'), ...
            struct('type','body', 'text',sprintf([ ...
                'The vehicle is modeled with a nonlinear dynamic bicycle model (linear tire\n' ...
                'model by default) driving a closed rounded-rectangle circuit. Position and\n' ...
                'heading are estimated by an Extended Kalman Filter fusing low-rate, noisy\n' ...
                'GPS position fixes with higher-rate IMU yaw-rate and acceleration readings.\n' ...
                'Path tracking uses an LQR controller on the linearized error dynamics, with\n' ...
                'curvature feedforward added for the steering command. A small physically-\n' ...
                'motivated disturbance (~%.2f deg steering, ~%.2f m/s^2 longitudinal) is applied\n' ...
                'to the TRUE vehicle every step, representing real-world effects no controller\n' ...
                'fully cancels -- crosswind gusts, road-surface irregularities, and imperfect\n' ...
                'actuators -- so that repeated laps and runs are not perfectly identical.'], ...
                procNoise.steer*180/pi, procNoise.accel)), ...
            struct('type','space'), ...
            struct('type','header', 'text','2.  Key Results'), ...
            struct('type','space'), ...
            struct('type','cards', 'items',{{ ...
                struct('label','RMS CROSS-TRACK ERROR', 'value',sprintf('%.3f m',rmsE1), 'color',[1 0.85 0.3]), ...
                struct('label','EKF POSITION ERROR',     'value',sprintf('%.3f m',rmsPosEst), 'color',[0.9 0.3 1.0]), ...
                struct('label','RAW GPS ERROR',           'value',sprintf('%.3f m',rmsPosGps), 'color',[0.7 0.7 0.7]), ...
                struct('label','EKF IMPROVEMENT',         'value',sprintf('%.1f %%',ekfImprovePct), 'color',[0.4 1 0.6]), ...
                struct('label','LAPS COMPLETED',          'value',sprintf('%d',lapCount), 'color',[0.4 0.95 1.0]), ...
                struct('label','BEST LAP TIME',           'value',bestLapStr, 'color',[1 0.55 0.85]), ...
                }}), ...
            struct('type','space'), ...
            struct('type','header', 'text','3.  Vehicle & Simulation Parameters'), ...
            struct('type','stat', 'text',sprintf('Mass / wheelbase          : %.0f kg / %.2f m', veh.m, veh.L)), ...
            struct('type','stat', 'text',sprintf('Understeer gradient Kus   : %.6f rad*s^2/m', veh.Kus)), ...
            struct('type','stat', 'text',sprintf('Integration step dt       : %.3f s (%.0f Hz)', dt, 1/dt)), ...
            struct('type','stat', 'text',sprintf('GPS / IMU noise (1-sigma) : %.2f m  /  %.3f rad/s, %.2f m/s^2', ...
                   noise.gpsPos, noise.imuYaw, noise.imuAccel)), ...
            struct('type','stat', 'text',sprintf('Skid-mark events (|slip| > %.1f deg) : %d', skidThreshold, numel(skidX))), ...
            }, REPORT_PAGE_SZ);
        exportgraphics(fSummary, reportFile, exportOpts{:}, 'Append',true);
        close(fSummary);

        % ---- Page 3: Live dashboard, caption + chart together ----
        fPage3 = renderReportChartPage({ ...
            struct('type','header', 'text','4.  Live Simulation Dashboard'), ...
            struct('type','body', 'text',sprintf([ ...
                'Snapshot of the animated dashboard at the end of the run: the track view\n' ...
                '(true path, EKF estimate, short-horizon predicted rollout, and raw GPS\n' ...
                'fixes), live gauges, lap timer, and the three time-history plots at right\n' ...
                '(commanded speed tracking, steering decomposition, and tracking/estimation\n' ...
                'error over the full %.0f-second run).'], tEnd)), ...
            }, fig, exportOpts, REPORT_PAGE_SZ);
        exportgraphics(fPage3, reportFile, exportOpts{:}, 'Append',true);
        close(fPage3);

        % ---- Page 4: Diagnostic figure, caption + chart together ----
        fPage4 = renderReportChartPage({ ...
            struct('type','header', 'text','5.  Estimator & Vehicle Diagnostics'), ...
            struct('type','body', 'text',sprintf([ ...
                'Front/rear tire slip angles and true lateral acceleration (with the\n' ...
                'vehicle''s handling-limit threshold marked) characterize how hard the\n' ...
                'vehicle is being driven. The EKF covariance trace shows estimator\n' ...
                'confidence converging over time, and the bottom-right panel compares\n' ...
                'the filtered position estimate directly against raw, unfiltered GPS.']) ), ...
            }, figDiag, exportOpts, REPORT_PAGE_SZ);
        exportgraphics(fPage4, reportFile, exportOpts{:}, 'Append',true);
        close(fPage4);

        % ---- Page 5: Monte Carlo trials, caption + chart together ----
        % (only if Section 9 actually produced the figure -- it's wrapped
        % in its own try/catch and independently optional, same as the
        % diagnostics and sweep figures)
        if exist('figMC','var') && isvalid(figMC)
            fPage5 = renderReportChartPage({ ...
                struct('type','header', 'text','6.  Monte Carlo Trials'), ...
                struct('type','body', 'text',sprintf([ ...
                    'RMS position error distribution across %d independent, randomly-seeded\n' ...
                    'trials per controller, isolating how much of Section 8''s single-run\n' ...
                    'comparison reflects consistent behavior rather than one favorable draw.\n' ...
                    'The EKF distribution sits tighter and lower than the raw-GPS distribution\n' ...
                    'across the sampled seeds.'], nMonteCarlo)), ...
                }, figMC, exportOpts, REPORT_PAGE_SZ);
            exportgraphics(fPage5, reportFile, exportOpts{:}, 'Append',true);
            close(fPage5);
        end

        % ---- Page 6: LQR sweep, caption + chart together ----
        fPage6 = renderReportChartPage({ ...
            struct('type','header', 'text','7.  LQR Weight Sensitivity Sweep'), ...
            struct('type','body', 'text',sprintf([ ...
                'Classical LQR tuning trade-off: sweeping the steering-effort penalty R\n' ...
                'while holding the tracking-error weighting Q fixed reproduces the\n' ...
                'textbook accuracy-vs-effort curve that guides Q/R weight selection --\n' ...
                'small R (aggressive) tracks tightly at the cost of harsher steering;\n' ...
                'large R (gentle) trades tracking accuracy for smoother control effort.']) ), ...
            }, figSweep, exportOpts, REPORT_PAGE_SZ);
        exportgraphics(fPage6, reportFile, exportOpts{:}, 'Append',true);
        close(fPage6);

        fprintf('\nPDF report saved: %s\n', reportFile);
    catch reportErr
        warning('PDF report export failed (%s). This feature needs MATLAB R2020a or later.', reportErr.message);
    end
end


%% ===================== SECTION 11B: POWERPOINT (PPTX) REPORT ==============
% Builds a real, 17-slide .pptx project showcase alongside the PDF,
% generated automatically every run: title, motivation, architecture,
% three methodology deep-dives, test environment, a snapshot of THIS
% run's live dashboard, a dashboard walkthrough, key results (using this
% run's actual numbers), an EKF-vs-raw-GPS comparison, LQR tuning
% trade-off, validation methodology, engineering highlights, a project
% roadmap, and a closing slide.
%
% A .pptx file is just a ZIP archive of XML parts (Office Open XML) --
% this writes those parts directly with plain fopen/fprintf and zips them
% with MATLAB's built-in zip(), so there's no PowerPoint/COM automation
% dependency (which would only work on Windows with Office installed) and
% no Report Generator toolbox dependency (a separate paid add-on that may
% not be installed). This is plain MATLAB, works on any platform.
if exportReport
    pptxFile = fullfile(scriptOutDir, 'AV_Simulation_Showcase.pptx');
    try
        pptxTmp = fullfile(tempdir, sprintf('av_pptx_build_%d', round(rand*1e6)));
        if exist(pptxTmp,'dir'); rmdir(pptxTmp,'s'); end
        dirs = {'', '_rels', 'ppt', 'ppt/_rels', 'ppt/slides', 'ppt/slides/_rels', ...
                'ppt/slideLayouts', 'ppt/slideLayouts/_rels', 'ppt/slideMasters', ...
                'ppt/slideMasters/_rels', 'ppt/theme', 'ppt/media'};
        for i = 1:numel(dirs)
            mkdir(fullfile(pptxTmp, dirs{i}));
        end

        % ---- Capture THIS run's actual dashboard as the embedded image ----
        % Same dark-background fix as the PDF export -- exportgraphics
        % defaults to a white alpha-blend backdrop otherwise.
        dashImgFile = fullfile(pptxTmp, 'ppt/media/image1.png');
        exportgraphics(fig, dashImgFile, 'BackgroundColor','black', 'Resolution',150);
        imInfo = imfinfo(dashImgFile);
        imAspect = imInfo.Width / imInfo.Height;

        % ---- This run's stats, formatted once for the results slide ----
        ekfImprovePctPP = 100*(1-rmsPosEst/rmsPosGps);
        bestLapStrPP = ternary(isinf(bestLapTime), '--', sprintf('%.2f s', bestLapTime));

        % ---- Ordered slide list. dashboardSlideIdx marks which one gets
        %      the image relationship -- keep this in sync with its
        %      position below if the order ever changes. ----
        slideBuilders = { ...
            @() pptxSlideTitle(), ...
            @() pptxSlideChallenge(), ...
            @() pptxSlideArchitecture(), ...
            @() pptxSlideMethodology(), ...
            @() pptxSlideBicycleModel(), ...
            @() pptxSlideEKFDeepDive(), ...
            @() pptxSlideLQRDeepDive(), ...
            @() pptxSlideTrackEnv(), ...
            @() pptxSlideDashboard(imAspect), ...
            @() pptxSlideDashboardWalkthrough(), ...
            @() pptxSlideResults(sprintf('%.3f m', rmsE1), sprintf('%.1f %%', ekfImprovePctPP), ...
                sprintf('%d', lapCount), bestLapStrPP, sprintf('%.3f m', rmsPosEst), sprintf('%.3f m', rmsPosGps)), ...
            @() pptxSlideFusionWins(posImprovePct), ...
            @() pptxSlideLQRTradeoff(), ...
            @() pptxSlideValidation(), ...
            @() pptxSlideEngineering(), ...
            @() pptxSlideRoadmap(), ...
            @() pptxSlideClosing() ...
        };
        dashboardSlideIdx = 9;
        nSlides = numel(slideBuilders);

        % ---- Static parts (content types, relationships, theme, master, layout) ----
        writeTextFile(fullfile(pptxTmp,'[Content_Types].xml'), pptxContentTypesXml(nSlides));
        writeTextFile(fullfile(pptxTmp,'_rels/.rels'), pptxRootRelsXml());
        writeTextFile(fullfile(pptxTmp,'ppt/presentation.xml'), pptxPresentationXml(nSlides));
        writeTextFile(fullfile(pptxTmp,'ppt/_rels/presentation.xml.rels'), pptxPresentationRelsXml(nSlides));
        writeTextFile(fullfile(pptxTmp,'ppt/theme/theme1.xml'), pptxThemeXml());
        writeTextFile(fullfile(pptxTmp,'ppt/slideMasters/slideMaster1.xml'), pptxSlideMasterXml());
        writeTextFile(fullfile(pptxTmp,'ppt/slideMasters/_rels/slideMaster1.xml.rels'), pptxSlideMasterRelsXml());
        writeTextFile(fullfile(pptxTmp,'ppt/slideLayouts/slideLayout1.xml'), pptxSlideLayoutXml());
        writeTextFile(fullfile(pptxTmp,'ppt/slideLayouts/_rels/slideLayout1.xml.rels'), pptxSlideLayoutRelsXml());

        % ---- Slide content + per-slide relationships ----
        emptyRels = pptxEmptyRelsXml();
        imageRels = pptxImageRelXml();
        for i = 1:nSlides
            writeTextFile(fullfile(pptxTmp, sprintf('ppt/slides/slide%d.xml', i)), slideBuilders{i}());
            if i == dashboardSlideIdx
                writeTextFile(fullfile(pptxTmp, sprintf('ppt/slides/_rels/slide%d.xml.rels', i)), imageRels);
            else
                writeTextFile(fullfile(pptxTmp, sprintf('ppt/slides/_rels/slide%d.xml.rels', i)), emptyRels);
            end
        end

        % ---- Zip and rename to .pptx ----
        zipTmp = fullfile(tempdir, sprintf('av_pptx_bundle_%d.zip', round(rand*1e6)));
        zip(zipTmp, {'[Content_Types].xml', '_rels', 'ppt'}, pptxTmp);
        if exist(pptxFile,'file'); delete(pptxFile); end
        movefile(zipTmp, pptxFile);
        rmdir(pptxTmp, 's');

        fprintf('PowerPoint report saved: %s (%d slides)\n', pptxFile, nSlides);
    catch pptxErr
        warning('PPTX report export failed (%s).', pptxErr.message);
    end
end


%% =========================================================================
%  SECTION 12: LOCAL FUNCTIONS
%
%  All functions below are pure (no globals), unit-testable in isolation,
%  and are reused identically by both the animated main loop and the
%  headless runTrial() used for baseline comparison, Monte Carlo trials,
%  and the LQR sensitivity sweep -- ensuring every analysis in this script
%  is evaluating the exact same underlying model and controller.
% =========================================================================

function [cx, cy, curvature, tangentAngle, s] = generateTrack(nPts, R0)
    % GENERATETRACK  Builds a smooth, closed "rounded rectangle" race
    %   track using a superellipse parametrization, then resamples it to
    %   uniform arc-length spacing for even point density all the way
    %   around (no bunching near the corners).
    %
    %   WHY A SUPERELLIPSE (not the old sine-harmonic shape):
    %   The previous version built the loop from radius = R0 + sine
    %   harmonics. For some harmonic combinations that produces a
    %   heart/clover shape with a concave NOTCH -- a point where
    %   curvature spikes sharply and can even flip sign abruptly. That
    %   caused two visible problems: (1) an ugly, unintended track shape,
    %   and (2) the nearest-point path search could jump across the notch
    %   to the "wrong" far side of the track, making the LQR feedback and
    %   the predictive rollout chase a bogus reference and shoot off
    %   wildly (exactly the runaway yellow dashed line you saw).
    %
    %   A superellipse  |x/a|^n + |y/b|^n = 1  with n > 2 has bounded,
    %   smoothly-varying curvature everywhere -- including at the
    %   "corners" -- and no concave notches, which structurally prevents
    %   both problems at once.
    %
    %   Inputs:
    %     nPts - number of (uniformly arc-length-spaced) output points
    %     R0   - nominal size scale [m]; the track spans roughly
    %            [-1.5*R0, 1.5*R0] x [-R0, R0]
    %
    %   Outputs:
    %     cx, cy       - centerline coordinates [m]
    %     curvature    - signed curvature kappa at each point [1/m]
    %     tangentAngle - heading of the path tangent at each point [rad]
    %     s            - cumulative arc length from the start point [m]

    a = 1.35*R0;    % half-length (long straights run along this axis)
    b = 0.85*R0;    % half-width
    n = 4;          % superellipse exponent: higher = squarer corners,
                     % 2 = pure ellipse. n=4 gives a classic rounded-
                     % rectangle race-track silhouette.

    thetaFine = linspace(0, 2*pi, 4000+1); thetaFine(end) = [];
    xFine = a * sign(cos(thetaFine)) .* abs(cos(thetaFine)).^(2/n);
    yFine = b * sign(sin(thetaFine)) .* abs(sin(thetaFine)).^(2/n);

    % --- resample to UNIFORM ARC LENGTH so points are evenly spaced
    %     around the whole loop (avoids clustering near the corners) ---
    dxFine = gradient(xFine); dyFine = gradient(yFine);
    segFine = hypot(dxFine, dyFine);
    sFine = [0, cumsum(segFine(1:end-1))];
    totalLen = sFine(end) + segFine(end);

    sTarget = linspace(0, totalLen, nPts+1); sTarget(end) = [];
    % wrap-safe interpolation: append the first point at the end with
    % s = totalLen so interp1 can wrap cleanly
    sFineWrap = [sFine, totalLen];
    xFineWrap = [xFine, xFine(1)];
    yFineWrap = [yFine, yFine(1)];

    cx = interp1(sFineWrap, xFineWrap, sTarget, 'pchip');
    cy = interp1(sFineWrap, yFineWrap, sTarget, 'pchip');

    dx  = gradient(cx);  dy  = gradient(cy);
    ddx = gradient(dx);  ddy = gradient(dy);
    curvature = (dx.*ddy - dy.*ddx) ./ (dx.^2 + dy.^2).^1.5;
    tangentAngle = atan2(dy, dx);

    segLen = hypot(dx, dy);
    s = cumsum(segLen) - segLen(1);
end


function stateDot = vehicleDynamicsCT(state, input, veh)
    % VEHICLEDYNAMICSCT  Nonlinear dynamic (not merely kinematic) bicycle model.
    %
    %   Unlike a kinematic model, this one includes lateral tire forces,
    %   so it captures understeer/oversteer behavior, sideslip, and the
    %   coupling between yaw rate and lateral velocity that a real car
    %   exhibits -- important because the LQR controller is designed
    %   against a LINEARIZATION of exactly this physics.
    %
    %   Tire model: EITHER a linear (small-slip-angle) lateral force law,
    %   Fy = C * alpha (accurate below ~4-5 deg slip), OR the nonlinear
    %   Pacejka "Magic Formula" (veh.tireModel='pacejka'), which captures
    %   the realistic SATURATING grip curve as slip grows toward and past
    %   the friction limit -- the linear model would keep producing more
    %   and more force forever, which is physically wrong at high slip.
    %   A friction scale veh.muScale (default 1.0) multiplies whichever
    %   tire force is used, representing reduced grip (rain, ice, gravel).
    %
    %   Inputs:
    %     state - [X; Y; psi; vx; vy; r]   (position, heading, body-frame
    %             longitudinal/lateral velocity, yaw rate)
    %     input - [delta; ax]              (steering angle, commanded
    %             longitudinal acceleration)
    %     veh   - struct with fields m, Iz, lf, lr, Cf, Cr, tireModel,
    %             muScale, and (if tireModel='pacejka') pacB/pacC/pacE/pacDf/pacDr
    %
    %   Output:
    %     stateDot - time derivative of state
    vxSafe = max(state(4), 0.5);   % guard against division near-zero speed

    alpha_f = input(1) - atan2(state(5) + veh.lf*state(6), vxSafe);
    alpha_r =           - atan2(state(5) - veh.lr*state(6), vxSafe);

    if isfield(veh,'muScale'); mu = veh.muScale; else; mu = 1.0; end

    if isfield(veh,'tireModel') && strcmp(veh.tireModel,'pacejka')
        Fyf = mu*veh.pacDf*sin(veh.pacC*atan(veh.pacB*alpha_f - ...
                veh.pacE*(veh.pacB*alpha_f - atan(veh.pacB*alpha_f))));
        Fyr = mu*veh.pacDr*sin(veh.pacC*atan(veh.pacB*alpha_r - ...
                veh.pacE*(veh.pacB*alpha_r - atan(veh.pacB*alpha_r))));
    else
        Fyf = mu*veh.Cf*alpha_f;
        Fyr = mu*veh.Cr*alpha_r;
    end

    vxdot  = input(2) + state(5)*state(6) - (Fyf*sin(input(1)))/veh.m;
    vydot  = (Fyf*cos(input(1)) + Fyr)/veh.m - state(4)*state(6);
    rdot   = (veh.lf*Fyf*cos(input(1)) - veh.lr*Fyr)/veh.Iz;
    psidot = state(6);
    Xdot   = state(4)*cos(state(3)) - state(5)*sin(state(3));
    Ydot   = state(4)*sin(state(3)) + state(5)*cos(state(3));

    stateDot = [Xdot; Ydot; psidot; vxdot; vydot; rdot];
end


function stateNext = rk4Step(state, input, dt, veh)
    % Classical 4th-order Runge-Kutta integration of the vehicle model.
    k1 = vehicleDynamicsCT(state,               input, veh);
    k2 = vehicleDynamicsCT(state + dt/2*k1,     input, veh);
    k3 = vehicleDynamicsCT(state + dt/2*k2,     input, veh);
    k4 = vehicleDynamicsCT(state + dt*k3,       input, veh);
    stateNext = state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
end


function imuMeas = simulateIMU(trueState, input, veh, noise)
    % Synthetic IMU: gyro (yaw rate) + accelerometer (longitudinal accel),
    % both corrupted with Gaussian noise.
    trueDot = vehicleDynamicsCT(trueState, input, veh);
    axTrue  = trueDot(4);
    rTrue   = trueState(6);
    imuMeas = [rTrue + noise.imuYaw*randn; axTrue + noise.imuAccel*randn];
end


function J = numJacobian(f, x0)
    % Generic central-difference numerical Jacobian of f() at x0.
    n = numel(x0);
    fx0 = f(x0);
    m = numel(fx0);
    J = zeros(m, n);
    for i = 1:n
        h = 1e-6*max(1, abs(x0(i)));
        dx = zeros(n,1); dx(i) = h;
        J(:,i) = (f(x0+dx) - f(x0-dx)) / (2*h);
    end
end


function [stateEst, P] = ekfPredict(stateEst, P, input, dt, veh, Q)
    % EKFPREDICT  Extended Kalman Filter time-update (prediction step).
    %
    %   Propagates both the state estimate and its error covariance P
    %   through the full NONLINEAR vehicle model. Because the model is
    %   nonlinear, the covariance propagation needs a local linearization
    %   (the Jacobian F = df/dstate); rather than hand-deriving this
    %   6x6 Jacobian analytically, it is computed numerically via central
    %   differences (numJacobian), which is more error-prone to derive by
    %   hand than to simply evaluate -- a common and perfectly valid
    %   engineering shortcut for EKFs on complex models.
    %
    %   Inputs:  stateEst, P (prior estimate/covariance), input, dt, veh, Q
    %   Outputs: stateEst, P (predicted estimate/covariance)
    f = @(s) rk4Step(s, input, dt, veh);
    F = numJacobian(f, stateEst);
    stateEst = f(stateEst);
    P = F*P*F' + Q;
end


function [stateEst, P] = ekfUpdateGPS(stateEst, P, zGps, Rgps)
    % EKF measurement-update using a GPS position fix.
    H = [1 0 0 0 0 0;
         0 1 0 0 0 0];
    yInnov = zGps - H*stateEst;
    S = H*P*H' + Rgps;
    K = P*H'/S;
    stateEst = stateEst + K*yInnov;
    P = (eye(6) - K*H)*P;
end


function [stateEst, P] = ekfUpdateIMU(stateEst, P, zImu, input, veh, Rimu)
    % EKF measurement-update using yaw-rate + longitudinal-accel from IMU.
    h = @(s) [s(6); imuAccelModel(s, input, veh)];
    H = numJacobian(h, stateEst);
    yInnov = zImu - h(stateEst);
    S = H*P*H' + Rimu;
    K = P*H'/S;
    stateEst = stateEst + K*yInnov;
    P = (eye(6) - K*H)*P;
end


function ax = imuAccelModel(state, input, veh)
    % Predicted accelerometer reading (longitudinal accel) for a given state.
    sDot = vehicleDynamicsCT(state, input, veh);
    ax = sDot(4);
end


function [sigmaPts, Wm, Wc] = generateSigmaPoints(meanVec, P, alphaUKF, betaUKF, kappaUKF)
    % GENERATESIGMAPOINTS  Scaled unscented transform sigma points and
    %   weights (van der Merwe / Wan & Merwe formulation), the standard
    %   basis of the Unscented Kalman Filter. Rather than linearizing the
    %   nonlinear model with a Jacobian (as the EKF does), the UKF
    %   propagates a small deterministic set of "sigma points" THROUGH
    %   the exact nonlinear model and reconstructs the mean/covariance
    %   from the results -- often more accurate for strongly nonlinear
    %   systems since no linearization error is introduced.
    n = numel(meanVec);
    lambda = alphaUKF^2*(n+kappaUKF) - n;
    Pscaled = (n+lambda)*P;
    Pscaled = (Pscaled+Pscaled')/2 + 1e-9*eye(n);   % symmetrize + jitter
    [S, flag] = chol(Pscaled, 'lower');
    if flag ~= 0
        S = chol(Pscaled + 1e-6*eye(n), 'lower');   % fallback jitter if not PD
    end
    sigmaPts = zeros(n, 2*n+1);
    sigmaPts(:,1) = meanVec;
    for i = 1:n
        sigmaPts(:,i+1)   = meanVec + S(:,i);
        sigmaPts(:,i+1+n) = meanVec - S(:,i);
    end
    Wm = zeros(1,2*n+1); Wc = zeros(1,2*n+1);
    Wm(1) = lambda/(n+lambda);
    Wc(1) = Wm(1) + (1 - alphaUKF^2 + betaUKF);
    Wm(2:end) = 1/(2*(n+lambda));
    Wc(2:end) = Wm(2:end);
end


function [stateEst, P] = ukfPredict(stateEst, P, input, dt, veh, Q)
    % UKFPREDICT  UKF time-update: propagate sigma points through the
    %   exact nonlinear model, then reconstruct the predicted mean/covariance.
    n = 6;
    [sigmaPts, Wm, Wc] = generateSigmaPoints(stateEst, P, 1e-3, 2, 0);
    propPts = zeros(n, 2*n+1);
    for i = 1:2*n+1
        propPts(:,i) = rk4Step(sigmaPts(:,i), input, dt, veh);
    end
    stateEst = propPts*Wm';
    P = Q;
    for i = 1:2*n+1
        dx = propPts(:,i) - stateEst;
        P = P + Wc(i)*(dx*dx');
    end
end


function [stateEst, P] = ukfUpdateGPS(stateEst, P, zGps, Rgps)
    % UKFUPDATEGPS  UKF measurement-update for a direct GPS position fix.
    n = 6;
    [sigmaPts, Wm, Wc] = generateSigmaPoints(stateEst, P, 1e-3, 2, 0);
    Zpts = sigmaPts(1:2,:);
    zPred = Zpts*Wm';
    Pzz = Rgps; Pxz = zeros(n,2);
    for i = 1:2*n+1
        dz = Zpts(:,i) - zPred; dx = sigmaPts(:,i) - stateEst;
        Pzz = Pzz + Wc(i)*(dz*dz');
        Pxz = Pxz + Wc(i)*(dx*dz');
    end
    K = Pxz/Pzz;
    stateEst = stateEst + K*(zGps - zPred);
    P = P - K*Pzz*K';
end


function [stateEst, P] = ukfUpdateIMU(stateEst, P, zImu, input, veh, Rimu)
    % UKFUPDATEIMU  UKF measurement-update for yaw-rate + longitudinal accel.
    n = 6;
    [sigmaPts, Wm, Wc] = generateSigmaPoints(stateEst, P, 1e-3, 2, 0);
    Zpts = zeros(2, 2*n+1);
    for i = 1:2*n+1
        Zpts(:,i) = [sigmaPts(6,i); imuAccelModel(sigmaPts(:,i), input, veh)];
    end
    zPred = Zpts*Wm';
    Pzz = Rimu; Pxz = zeros(n,2);
    for i = 1:2*n+1
        dz = Zpts(:,i) - zPred; dx = sigmaPts(:,i) - stateEst;
        Pzz = Pzz + Wc(i)*(dz*dz');
        Pxz = Pxz + Wc(i)*(dx*dz');
    end
    K = Pxz/Pzz;
    stateEst = stateEst + K*(zImu - zPred);
    P = P - K*Pzz*K';
end


function [alphaF_deg, alphaR_deg] = tireSlipAngles(state, input, veh)
    % Computes front/rear tire slip angles (in degrees) for a given state
    % and input, using the same small-slip kinematic relations as the
    % dynamics model. Useful for checking that the linear-tire assumption
    % remains valid (slip angles should generally stay within a few degrees
    % for a road car operating well below the friction limit).
    vxSafe = max(state(4), 0.5);
    alpha_f = input(1) - atan2(state(5) + veh.lf*state(6), vxSafe);
    alpha_r =           - atan2(state(5) - veh.lr*state(6), vxSafe);
    alphaF_deg = rad2deg(alpha_f);
    alphaR_deg = rad2deg(alpha_r);
end


function [idx, psiPath, kappa, e1] = projectToPath(X, Y, trackX, trackY, trackKappa, trackPsi, idxGuess, windowHalf)
    % PROJECTTOPATH  Frenet-style projection onto the reference track.
    %
    %   Finds the nearest track point and returns the signed lateral
    %   (cross-track) error relative to its tangent, plus the local
    %   heading and curvature there.
    %
    %   SEARCH MODE (this is the key robustness fix):
    %     - If idxGuess is omitted/empty, a GLOBAL nearest-neighbor search
    %       is done over the whole track (used once, to initialize).
    %     - If idxGuess is supplied, the search is restricted to a small
    %       WINDOW of +/- windowHalf points around it (default 40; on a
    %       500-point closed track that is roughly +/- 15% of the lap).
    %       Since the vehicle only moves a short distance between calls,
    %       the correct match is always nearby -- a windowed "sticky"
    %       search prevents the projection from ever jumping to a distant,
    %       unrelated part of the track even momentarily (which is what
    %       caused the wild, far-off predicted trajectory previously: a
    %       global search occasionally locked onto the wrong branch of a
    %       track with a tight concave section). Callers should pass the
    %       PREVIOUS call's returned idx back in as idxGuess to keep the
    %       search "stuck" to the correct region of the track.
    %
    %   Inputs:
    %     X, Y                    - query point [m]
    %     trackX, trackY          - track centerline coordinates [m]
    %     trackKappa, trackPsi    - track curvature [1/m] and tangent [rad]
    %     idxGuess  (optional)    - previous index, enables windowed search
    %     windowHalf (optional)   - half-width of the search window [pts]
    %
    %   Outputs:
    %     idx     - matched track index (feed back in as idxGuess next call)
    %     psiPath - tangent heading at the matched point [rad]
    %     kappa   - curvature at the matched point [1/m], clamped to a
    %               sane physical range as a safety net against any
    %               residual numerical spikes
    %     e1      - signed lateral (cross-track) error [m]

    n = numel(trackX);
    if nargin < 7 || isempty(idxGuess) || isnan(idxGuess)
        d2 = (trackX - X).^2 + (trackY - Y).^2;
        [~, idx] = min(d2);
    else
        if nargin < 8 || isempty(windowHalf)
            windowHalf = 40;
        end
        idxRange = mod((idxGuess-windowHalf-1):(idxGuess+windowHalf-1), n) + 1;
        d2local = (trackX(idxRange) - X).^2 + (trackY(idxRange) - Y).^2;
        [~, relIdx] = min(d2local);
        idx = idxRange(relIdx);
    end

    psiPath = trackPsi(idx);
    kappaMax = 0.3;   % clamp: ~3.3 m minimum turn radius, generous safety net
    kappa = max(min(trackKappa(idx), kappaMax), -kappaMax);

    vecX = X - trackX(idx);
    vecY = Y - trackY(idx);
    e1 = -vecX*sin(psiPath) + vecY*cos(psiPath);
end


function [A, B] = lateralErrorModel(veh, vx)
    % LATERALERRORMODEL  Linearized 4-state path-tracking error model.
    %
    %   This is the classical error-dynamics linearization used throughout
    %   the lane-keeping / path-following control literature (e.g. Rajamani,
    %   "Vehicle Dynamics and Control"). Rather than regulating [X,Y,psi]
    %   directly, the controller regulates the error RELATIVE to the path:
    %
    %     e1 - lateral (cross-track) distance from the path
    %     e2 - rate of change of e1
    %     e3 - heading error relative to the path tangent
    %     e4 - rate of change of e3 (related to yaw-rate error)
    %
    %   The model assumes constant longitudinal speed vx (valid locally,
    %   since the gain is recomputed online at the current speed -- this
    %   is a form of gain scheduling).
    %
    %   Inputs:  veh (vehicle parameter struct), vx (design speed) [m/s]
    %   Outputs: A (4x4), B (4x1) continuous-time state-space matrices
    Cf = veh.Cf; Cr = veh.Cr; m = veh.m; Iz = veh.Iz; lf = veh.lf; lr = veh.lr;

    A = [0, 1, 0, 0;
         0, -(Cf+Cr)/(m*vx), (Cf+Cr)/m, (-Cf*lf+Cr*lr)/(m*vx);
         0, 0, 0, 1;
         0, -(Cf*lf-Cr*lr)/(Iz*vx), (Cf*lf-Cr*lr)/Iz, -(Cf*lf^2+Cr*lr^2)/(Iz*vx)];

    B = [0; Cf/m; 0; Cf*lf/Iz];
end


function deltaFF = feedforwardSteering(kappa, vx, L, Kus)
    % Classical steady-state cornering feedforward (Ackermann + understeer).
    deltaFF = kappa*L + Kus*vx^2*kappa;
end


function vTarget = speedPlanner(kappa, vmin, vmax, ayMax)
    % Limits target speed so that lateral acceleration stays under ayMax:
    % ay = v^2 * kappa  =>  v <= sqrt(ayMax / |kappa|)
    vCurveLimit = sqrt(ayMax / max(abs(kappa), 1e-4));
    vTarget = max(min(vCurveLimit, vmax), vmin);
end


function K = computeLQRGainDARE(A, B, Q, R, dt, maxIter)
    % COMPUTELQRGAINDARE  Discrete-time LQR gain via Riccati value iteration.
    %
    %   Solves  min sum( e'*Q*e + u'*R*u )  subject to e_{k+1} = Ad*e_k + Bd*u_k
    %   by iterating the discrete algebraic Riccati equation (DARE):
    %
    %       P_{i+1} = Q + Ad'*P_i*Ad - Ad'*P_i*Bd*K_i*Ad,
    %       K_i     = (R + Bd'*P_i*Bd)^-1 * Bd'*P_i*Ad
    %
    %   until P converges, which is mathematically equivalent to the
    %   value-iteration form of dynamic programming applied to an infinite-
    %   horizon LQR problem. This avoids any dependency on the Control
    %   System Toolbox's `dlqr` function -- everything here is plain
    %   linear algebra available in base MATLAB.
    %
    %   Inputs:  A, B (continuous-time), Q, R (cost weights), dt, maxIter
    %   Output:  K (1x4 state feedback gain, u = -K*e)
    Ad = eye(size(A)) + A*dt;   % simple Euler discretization (small dt)
    Bd = B*dt;

    P = Q;
    for i = 1:maxIter
        K = (R + Bd'*P*Bd) \ (Bd'*P*Ad);
        P = Q + Ad'*P*Ad - Ad'*P*Bd*K;
    end
end


function pts = roundedRectPoints(halfLen, halfWid, cornerRadius, nArcPts)
    % ROUNDEDRECTPOINTS  Generates the perimeter of a rectangle with
    %   smoothly filleted (rounded) corners, centered at the origin --
    %   used for the car body/cabin silhouettes so they read as a real
    %   vehicle rather than a sharp-cornered box. Standard construction:
    %   four quarter-circle arcs (one per corner), each swept exactly 90
    %   degrees; the straight edges between corners are implicit (a patch
    %   or line simply connects consecutive points directly).
    %
    %   Inputs:
    %     halfLen, halfWid - half-length and half-width of the rectangle
    %     cornerRadius     - fillet radius (clamped to never exceed the
    %                        smaller half-dimension, so it can't self-intersect)
    %     nArcPts          - points per corner arc (more = smoother curve)
    %
    %   Output:
    %     pts - Nx2 matrix of [x,y] perimeter points, traversed
    %           counter-clockwise starting from the top-right corner
    r = min(cornerRadius, 0.95*min(halfLen, halfWid));
    cornerCenters = [ halfLen-r,   halfWid-r;    % top-right
                      -(halfLen-r), halfWid-r;    % top-left
                      -(halfLen-r), -(halfWid-r); % bottom-left
                       halfLen-r,  -(halfWid-r)]; % bottom-right
    startAngles = [0, pi/2, pi, 3*pi/2];
    pts = zeros(4*nArcPts, 2);
    for c = 1:4
        ang = startAngles(c) + linspace(0, pi/2, nArcPts);
        idxRange = (c-1)*nArcPts+1 : c*nArcPts;
        pts(idxRange,1) = cornerCenters(c,1) + r*cos(ang);
        pts(idxRange,2) = cornerCenters(c,2) + r*sin(ang);
    end
end


function out = ternary(cond, a, b)
    % TERNARY  MATLAB has no built-in ternary operator; small helper used
    %   by the report generation code for compact conditional strings.
    if cond
        out = a;
    else
        out = b;
    end
end


%% ---------------------- PPTX generation helpers ---------------------------
% A .pptx file is a ZIP archive of XML "parts". The functions below build
% those parts directly as plain text (no toolbox, no COM automation), and
% Section 11B zips them together into the final file. Coordinates are in
% inches; pptxEmu() converts to the English Metric Units OOXML expects.

function id = pptxNextId()
    persistent counter
    if isempty(counter), counter = 1; end
    counter = counter + 1;
    id = counter;
end

function e = pptxEmu(inches)
    e = round(inches * 914400);
end

function writeTextFile(path, content)
    fid = fopen(path, 'w');
    fwrite(fid, content, 'char');
    fclose(fid);
end

function esc = xmlEscape(txt)
    % XMLESCAPE  Escapes the handful of characters that are structurally
    %   significant in XML element content (&, <, >) so arbitrary slide
    %   text can never break the surrounding markup.
    %
    %   BUGFIX: pptxTextBox used to splice its 'txt' argument straight
    %   into '<a:t>%s</a:t>' with no escaping at all. Two of the static
    %   strings elsewhere in this file contain a literal '&' ("Model &
    %   Disturbance Uncertainty", "linear & Pacejka tire options"), which
    %   is not valid XML on its own (a bare '&' starts an entity
    %   reference) -- it corrupted those two slides' XML, and because a
    %   PPTX-repairing viewer's response to one malformed part can affect
    %   how it renders the rest of the package, that one typo-sized bug
    %   was the root cause of an entire dashboard-image slide appearing
    %   blank elsewhere in the deck. Every string reaching the XML now
    %   goes through here first, so this can't recur even if future
    %   slide text happens to contain '&', '<', or '>'.
    esc = strrep(txt, '&', '&amp;');
    esc = strrep(esc, '<', '&lt;');
    esc = strrep(esc, '>', '&gt;');
end

function xml = pptxTextBox(x, y, w, h, txt, sz, color, bold, italic, align)
    b = ternary(bold, '1', '0'); it = ternary(italic, '1', '0');
    id = pptxNextId();
    xml = sprintf(['<p:sp><p:nvSpPr><p:cNvPr id="%d" name="TextBox %d"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>' ...
        '<p:spPr><a:xfrm><a:off x="%d" y="%d"/><a:ext cx="%d" cy="%d"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>' ...
        '<p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0"><a:noAutofit/></a:bodyPr><a:lstStyle/>' ...
        '<a:p><a:pPr algn="%s"/><a:r><a:rPr lang="en-US" sz="%d" b="%s" i="%s" dirty="0"><a:solidFill><a:srgbClr val="%s"/></a:solidFill><a:latin typeface="Calibri"/></a:rPr><a:t>%s</a:t></a:r></a:p>' ...
        '</p:txBody></p:sp>'], id, id, pptxEmu(x), pptxEmu(y), pptxEmu(w), pptxEmu(h), align, round(sz*100), b, it, color, xmlEscape(txt));
end

function xml = pptxRect(x, y, w, h, fillColor, lineColor, lineW, radius)
    id = pptxNextId();
    if radius > 0
        prst = 'roundRect'; avlst = sprintf('<a:gd name="adj" fmla="val %d"/>', round(radius*100000));
    else
        prst = 'rect'; avlst = '';
    end
    if isempty(fillColor), fillXml = '<a:noFill/>'; else, fillXml = sprintf('<a:solidFill><a:srgbClr val="%s"/></a:solidFill>', fillColor); end
    if isempty(lineColor), lineXml = '<a:ln><a:noFill/></a:ln>'; else, lineXml = sprintf('<a:ln w="%d"><a:solidFill><a:srgbClr val="%s"/></a:solidFill></a:ln>', round(lineW*12700), lineColor); end
    xml = sprintf(['<p:sp><p:nvSpPr><p:cNvPr id="%d" name="Rect %d"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>' ...
        '<p:spPr><a:xfrm><a:off x="%d" y="%d"/><a:ext cx="%d" cy="%d"/></a:xfrm><a:prstGeom prst="%s"><a:avLst>%s</a:avLst></a:prstGeom>%s%s</p:spPr>' ...
        '<p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody></p:sp>'], ...
        id, id, pptxEmu(x), pptxEmu(y), pptxEmu(w), pptxEmu(h), prst, avlst, fillXml, lineXml);
end

function xml = pptxEllipse(x, y, w, h, fillColor, lineColor, lineW)
    id = pptxNextId();
    if isempty(fillColor), fillXml = '<a:noFill/>'; else, fillXml = sprintf('<a:solidFill><a:srgbClr val="%s"/></a:solidFill>', fillColor); end
    if isempty(lineColor), lineXml = '<a:ln><a:noFill/></a:ln>'; else, lineXml = sprintf('<a:ln w="%d"><a:solidFill><a:srgbClr val="%s"/></a:solidFill></a:ln>', round(lineW*12700), lineColor); end
    xml = sprintf(['<p:sp><p:nvSpPr><p:cNvPr id="%d" name="Ellipse %d"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>' ...
        '<p:spPr><a:xfrm><a:off x="%d" y="%d"/><a:ext cx="%d" cy="%d"/></a:xfrm><a:prstGeom prst="ellipse"><a:avLst/></a:prstGeom>%s%s</p:spPr>' ...
        '<p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody></p:sp>'], ...
        id, id, pptxEmu(x), pptxEmu(y), pptxEmu(w), pptxEmu(h), fillXml, lineXml);
end

function xml = pptxLine(x1, y1, x2, y2, color, lineW)
    id = pptxNextId();
    cx = max(pptxEmu(abs(x2-x1)), 1); cy = max(pptxEmu(abs(y2-y1)), 1);
    xml = sprintf(['<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="%d" name="Line %d"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>' ...
        '<p:spPr><a:xfrm><a:off x="%d" y="%d"/><a:ext cx="%d" cy="%d"/></a:xfrm><a:prstGeom prst="line"><a:avLst/></a:prstGeom>' ...
        '<a:ln w="%d"><a:solidFill><a:srgbClr val="%s"/></a:solidFill></a:ln></p:spPr></p:cxnSp>'], ...
        id, id, pptxEmu(min(x1,x2)), pptxEmu(min(y1,y2)), cx, cy, round(lineW*12700), color);
end

function xml = pptxPicture(relId, x, y, w, h)
    id = pptxNextId();
    xml = sprintf(['<p:pic><p:nvPicPr><p:cNvPr id="%d" name="Picture %d"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>' ...
        '<p:blipFill><a:blip r:embed="%s"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>' ...
        '<p:spPr><a:xfrm><a:off x="%d" y="%d"/><a:ext cx="%d" cy="%d"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'], ...
        id, id, relId, pptxEmu(x), pptxEmu(y), pptxEmu(w), pptxEmu(h));
end

function xml = pptxSlideWrap(shapesXml)
    xml = sprintf(['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n' ...
        '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">' ...
        '<p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="0A0E14"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>' ...
        '<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>' ...
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>' ...
        '%s</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'], shapesXml);
end

function xml = pptxContentTypesXml(nSlides)
    overrides = '';
    for i = 1:nSlides
        overrides = [overrides sprintf('<Override PartName="/ppt/slides/slide%d.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>', i)]; %#ok<AGROW>
    end
    xml = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' ...
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' ...
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>' ...
        '<Default Extension="xml" ContentType="application/xml"/>' ...
        '<Default Extension="png" ContentType="image/png"/>' ...
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>' ...
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>' ...
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>' ...
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>' ...
        overrides ...
        '</Types>'];
end

function xml = pptxRootRelsXml()
    xml = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' ...
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' ...
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>' ...
        '</Relationships>'];
end

function xml = pptxPresentationXml(nSlides)
    sldIds = '';
    for i = 1:nSlides
        sldIds = [sldIds sprintf('<p:sldId id="%d" r:id="rId%d"/>', 255+i, i+1)]; %#ok<AGROW>
    end
    xml = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' ...
        '<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">' ...
        '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>' ...
        '<p:sldIdLst>' sldIds '</p:sldIdLst>' ...
        '<p:sldSz cx="12192000" cy="6858000" type="screen16x9"/><p:notesSz cx="6858000" cy="9144000"/>' ...
        '</p:presentation>'];
end

function xml = pptxPresentationRelsXml(nSlides)
    slideRels = '';
    for i = 1:nSlides
        slideRels = [slideRels sprintf('<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide%d.xml"/>', i+1, i)]; %#ok<AGROW>
    end
    xml = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' ...
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' ...
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>' ...
        slideRels ...
        sprintf('<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>', nSlides+2) ...
        '</Relationships>'];
end

function xml = pptxThemeXml()
    xml = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' ...
        '<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="AVReportTheme"><a:themeElements>' ...
        '<a:clrScheme name="AVReport">' ...
        '<a:dk1><a:srgbClr val="000000"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>' ...
        '<a:dk2><a:srgbClr val="0A0E14"/></a:dk2><a:lt2><a:srgbClr val="D7DCE6"/></a:lt2>' ...
        '<a:accent1><a:srgbClr val="3DDCFF"/></a:accent1><a:accent2><a:srgbClr val="E063FF"/></a:accent2>' ...
        '<a:accent3><a:srgbClr val="FFC93D"/></a:accent3><a:accent4><a:srgbClr val="5CE68C"/></a:accent4>' ...
        '<a:accent5><a:srgbClr val="8B95A8"/></a:accent5><a:accent6><a:srgbClr val="121927"/></a:accent6>' ...
        '<a:hlink><a:srgbClr val="3DDCFF"/></a:hlink><a:folHlink><a:srgbClr val="E063FF"/></a:folHlink>' ...
        '</a:clrScheme>' ...
        '<a:fontScheme name="AVReport"><a:majorFont><a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont>' ...
        '<a:minorFont><a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme>' ...
        '<a:fmtScheme name="AVReport">' ...
        '<a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>' ...
        '<a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="12700"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="19050"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>' ...
        '<a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>' ...
        '<a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>' ...
        '</a:fmtScheme></a:themeElements></a:theme>'];
end

function xml = pptxSlideMasterXml()
    xml = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' ...
        '<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">' ...
        '<p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="0A0E14"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>' ...
        '<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>' ...
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>' ...
        '<p:clrMap bg1="lt1" tx1="dk1" bg2="dk2" tx2="lt2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>' ...
        '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>' ...
        '</p:sldMaster>'];
end

function xml = pptxSlideMasterRelsXml()
    xml = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' ...
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' ...
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>' ...
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>' ...
        '</Relationships>'];
end

function xml = pptxSlideLayoutXml()
    xml = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' ...
        '<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">' ...
        '<p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>' ...
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>' ...
        '<p:clrMapOvr><a:overrideClrMapping bg1="lt1" tx1="dk1" bg2="dk2" tx2="lt2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/></p:clrMapOvr>' ...
        '</p:sldLayout>'];
end

function xml = pptxSlideLayoutRelsXml()
    xml = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' ...
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' ...
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>' ...
        '</Relationships>'];
end

function xml = pptxEmptyRelsXml()
    xml = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' ...
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"></Relationships>'];
end

function xml = pptxImageRelXml()
    xml = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' ...
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' ...
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/image1.png"/>' ...
        '</Relationships>'];
end

function xml = pptxSlideTitle()
    s = {};
    s{end+1} = pptxEllipse(9.4,0.3,3.8,3.8,'','232B3D',1);
    s{end+1} = pptxEllipse(9.85,0.75,2.9,2.9,'','3DDCFF',1.5);
    s{end+1} = pptxEllipse(11.16,2.06,0.28,0.28,'FFC93D','',0);
    s{end+1} = pptxTextBox(0.9,2.5,9.5,0.85,'AUTONOMOUS VEHICLE',40,'FFFFFF',true,false,'l');
    s{end+1} = pptxTextBox(0.9,3.25,10.5,0.7,'STATE ESTIMATION &amp; OPTIMAL CONTROL',30,'3DDCFF',true,false,'l');
    s{end+1} = pptxTextBox(0.9,4.05,9,0.45,'A MATLAB simulation framework for sensor fusion and closed-loop path tracking',15,'8B95A8',false,true,'l');
    s{end+1} = pptxRect(0.9,4.75,2.9,0.42,'121927','232B3D',1,0.18);
    s{end+1} = pptxTextBox(0.9,4.83,2.9,0.32,'DYNAMIC BICYCLE MODEL',10.5,'D7DCE6',true,false,'ctr');
    s{end+1} = pptxRect(3.95,4.75,3.1,0.42,'121927','232B3D',1,0.18);
    s{end+1} = pptxTextBox(3.95,4.83,3.1,0.32,'EXTENDED KALMAN FILTER',10.5,'D7DCE6',true,false,'ctr');
    s{end+1} = pptxRect(7.2,4.75,2.5,0.42,'121927','232B3D',1,0.18);
    s{end+1} = pptxTextBox(7.2,4.83,2.5,0.32,'LQR PATH TRACKING',10.5,'D7DCE6',true,false,'ctr');
    s{end+1} = pptxLine(0.9,6.55,5.1,6.55,'232B3D',1);
    s{end+1} = pptxTextBox(0.9,6.65,8,0.35,'MATLAB   |   Live Animated Dashboard   |   Auto-Generated Report',11,'8B95A8',false,false,'l');
    xml = pptxSlideWrap(strjoin(s,''));
end

function xml = pptxSlideMethodology()
    s = {};
    s{end+1} = pptxTextBox(0.9,0.5,6,0.35,'METHODOLOGY',12,'3DDCFF',true,false,'l');
    s{end+1} = pptxTextBox(0.9,0.85,11.5,0.6,'Three models working together',27,'FFFFFF',true,false,'l');
    cols = {
        {'1','3DDCFF','Dynamic Bicycle Model','Nonlinear two-wheel vehicle model with lateral and yaw dynamics, driven by tire slip forces (linear or Pacejka curves).'}, ...
        {'2','E063FF','Extended Kalman Filter','Predicts vehicle state forward, then corrects it against noisy GPS and IMU measurements every step.'}, ...
        {'3','FFC93D','LQR Path Tracking','Solves for the optimal steering feedback gain, balancing tracking accuracy against effort, plus curvature feedforward.'} ...
    };
    cw = 3.75; cgap = 0.35; cx0 = 0.9; cy0 = 1.85; ch = 4.3;
    for i = 1:3
        x = cx0 + (i-1)*(cw+cgap);
        col = cols{i};
        s{end+1} = pptxRect(x,cy0,cw,ch,'121927','232B3D',1,0.05); %#ok<AGROW>
        s{end+1} = pptxEllipse(x+0.3,cy0+0.3,0.55,0.55,'0F1520',col{2},1.8); %#ok<AGROW>
        s{end+1} = pptxTextBox(x+0.3,cy0+0.4,0.55,0.4,col{1},20,col{2},true,false,'ctr'); %#ok<AGROW>
        s{end+1} = pptxTextBox(x+0.3,cy0+1.05,cw-0.6,0.6,col{3},15.5,'FFFFFF',true,false,'l'); %#ok<AGROW>
        s{end+1} = pptxTextBox(x+0.3,cy0+1.7,cw-0.6,1.8,col{4},11,'D7DCE6',false,false,'l'); %#ok<AGROW>
    end
    xml = pptxSlideWrap(strjoin(s,''));
end

function xml = pptxSlideChallenge()
    s = {};
    s{end+1} = pptxTextBox(0.9,0.5,6,0.35,'THE CHALLENGE',12,'3DDCFF',true,false,'l');
    s{end+1} = pptxTextBox(0.9,0.9,11.5,0.9,'Real sensors are noisy, slow, and never enough on their own',27,'FFFFFF',true,false,'l');
    s{end+1} = pptxTextBox(0.9,1.85,7.1,1.9, [ ...
        'An autonomous vehicle only ever sees the world through imperfect sensors. GPS ' ...
        'fixes arrive slowly and wander by half a meter or more; inertial sensors drift ' ...
        'over time. A controller that trusts raw readings directly will track poorly and ' ...
        'behave unpredictably -- the vehicle needs a best estimate of where it really is, ' ...
        'fused from everything available, updated every step.'], 13,'D7DCE6',false,false,'l');
    items = {
        {'1','3DDCFF','Noisy, Low-Rate GPS','Fixes arrive at ~10 Hz with meter-scale noise -- too coarse to steer on directly.'}, ...
        {'2','E063FF','Real-Time Control','Steering must be computed every 20 ms, using only what has been estimated so far.'}, ...
        {'3','FFC93D','Model & Disturbance Uncertainty','Crosswind, road surface, and actuator imperfections mean reality never quite matches the model.'} ...
    };
    iy = 1.8;
    for i = 1:3
        it = items{i};
        s{end+1} = pptxEllipse(8.35,iy,0.42,0.42,'121927',it{2},1.5); %#ok<AGROW>
        s{end+1} = pptxTextBox(8.35,iy,0.42,0.42,it{1},15,it{2},true,false,'ctr'); %#ok<AGROW>
        s{end+1} = pptxTextBox(8.95,iy-0.02,3.6,0.4,it{3},13.5,'FFFFFF',true,false,'l'); %#ok<AGROW>
        s{end+1} = pptxTextBox(8.95,iy+0.42,3.6,0.85,it{4},10,'8B95A8',false,false,'l'); %#ok<AGROW>
        iy = iy + 1.5;
    end
    xml = pptxSlideWrap(strjoin(s,''));
end

function xml = pptxSlideArchitecture()
    s = {};
    s{end+1} = pptxTextBox(0.9,0.5,6,0.35,'SYSTEM ARCHITECTURE',12,'3DDCFF',true,false,'l');
    s{end+1} = pptxTextBox(0.9,0.9,11.5,0.6,'A closed sense -> estimate -> control -> act loop',27,'FFFFFF',true,false,'l');
    stages = {
        {'SENSORS','GPS (10 Hz) + IMU (yaw rate, accel) at 50 Hz','8B95A8'}, ...
        {'EXTENDED KALMAN FILTER','Fuses noisy, async measurements into one state estimate','3DDCFF'}, ...
        {'LQR CONTROLLER','Curvature feedforward + linear feedback on tracking error','E063FF'}, ...
        {'VEHICLE DYNAMICS','Nonlinear bicycle model + real-world disturbance','FFC93D'} ...
    };
    boxW = 2.55; gap = 0.55; startX = 0.9; boxY = 2.5; boxH = 2.0;
    for i = 1:4
        x = startX + (i-1)*(boxW+gap);
        st = stages{i};
        s{end+1} = pptxRect(x,boxY,boxW,boxH,'121927',st{3},1.6,0.09); %#ok<AGROW>
        s{end+1} = pptxRect(x+0.18,boxY+0.2,boxW-0.36,0.06,st{3},'',0,0); %#ok<AGROW>
        s{end+1} = pptxTextBox(x+0.15,boxY+0.42,boxW-0.3,0.6,st{1},13,'FFFFFF',true,false,'ctr'); %#ok<AGROW>
        s{end+1} = pptxTextBox(x+0.15,boxY+1.05,boxW-0.3,0.85,st{2},10,'8B95A8',false,false,'ctr'); %#ok<AGROW>
        if i < 4
            s{end+1} = pptxTextBox(x+boxW,boxY+boxH/2-0.2,gap,0.4,'>',16,'232B3D',false,false,'ctr'); %#ok<AGROW>
        end
    end
    loopY = boxY + boxH + 0.55;
    s{end+1} = pptxLine(startX+boxW/2, loopY, startX+3*(boxW+gap)+boxW/2, loopY, '232B3D', 1.2);
    s{end+1} = pptxTextBox(startX,loopY+0.15,11.5,0.35,'estimate feeds back into the filter every step -- the loop runs closed, in real time',10.5,'8B95A8',false,true,'ctr');
    xml = pptxSlideWrap(strjoin(s,''));
end

function xml = pptxSlideBicycleModel()
    s = {};
    s{end+1} = pptxTextBox(0.9,0.5,6,0.35,'METHODOLOGY  --  DEEP DIVE 1 OF 3',12,'3DDCFF',true,false,'l');
    s{end+1} = pptxTextBox(0.9,0.9,11.5,0.6,'Dynamic Bicycle Model',27,'FFFFFF',true,false,'l');
    s{end+1} = pptxTextBox(0.9,1.75,7.0,3.8, [ ...
        'The vehicle is represented as a two-wheel "bicycle" -- one effective front tire, ' ...
        'one effective rear -- tracking lateral velocity, yaw rate, heading, and position. ' ...
        'Tire forces come from a slip-angle-based tire model, selectable between a simple ' ...
        'linear model (fast, accurate near the limits of normal driving) and the full ' ...
        'nonlinear Pacejka "magic formula" curve (captures grip saturation at the limit).\n\n' ...
        'Low-grip zones on the track scale down the available friction locally, and a ' ...
        'small real-world disturbance -- representing crosswind gusts and road-surface ' ...
        'irregularities -- is applied to the TRUE vehicle every step, so the plant the ' ...
        'controller has to handle is never perfectly clean or repeatable.'], 12.5,'D7DCE6',false,false,'l');
    facts = {
        {'STATE VECTOR','6 states: y, vy, r, psi, X, Y','3DDCFF'}, ...
        {'TIRE MODEL','Linear (default) or Pacejka','E063FF'}, ...
        {'INTEGRATION','RK4, fixed 20 ms timestep','FFC93D'}, ...
        {'DISTURBANCE','Steering + longitudinal noise on the true plant','5CE68C'} ...
    };
    fy = 1.85;
    for i = 1:4
        f = facts{i};
        s{end+1} = pptxRect(8.3,fy,4.15,0.85,'121927',f{3},1.2,0.07); %#ok<AGROW>
        s{end+1} = pptxTextBox(8.5,fy+0.12,3.8,0.3,f{1},9.5,'8B95A8',true,false,'l'); %#ok<AGROW>
        s{end+1} = pptxTextBox(8.5,fy+0.42,3.8,0.4,f{2},12,f{3},true,false,'l'); %#ok<AGROW>
        fy = fy + 1.05;
    end
    xml = pptxSlideWrap(strjoin(s,''));
end

function xml = pptxSlideEKFDeepDive()
    s = {};
    s{end+1} = pptxTextBox(0.9,0.5,6,0.35,'METHODOLOGY  --  DEEP DIVE 2 OF 3',12,'3DDCFF',true,false,'l');
    s{end+1} = pptxTextBox(0.9,0.9,11.5,0.6,'Extended Kalman Filter',27,'FFFFFF',true,false,'l');
    steps = {
        {'PREDICT','Propagate the state estimate forward using the vehicle motion model, growing the uncertainty covariance to reflect what could have happened since the last update.','3DDCFF'}, ...
        {'UPDATE (IMU)','Correct heading rate and acceleration against the gyro/accelerometer at 50 Hz, tightening the estimate between GPS fixes.','E063FF'}, ...
        {'UPDATE (GPS)','Whenever a GPS fix arrives (~10 Hz), correct position directly, weighted by how much the filter currently trusts it.','FFC93D'} ...
    };
    cw=3.75; cgap=0.35; cx0=0.9; cy0=1.85; ch=3.3;
    for i=1:3
        x = cx0 + (i-1)*(cw+cgap); st = steps{i};
        s{end+1} = pptxRect(x,cy0,cw,ch,'121927',st{3},1.4,0.06); %#ok<AGROW>
        s{end+1} = pptxTextBox(x+0.3,cy0+0.25,cw-0.6,0.4,st{1},14,st{3},true,false,'l'); %#ok<AGROW>
        s{end+1} = pptxTextBox(x+0.3,cy0+0.8,cw-0.6,2.3,st{2},10.5,'D7DCE6',false,false,'l'); %#ok<AGROW>
        if i < 3
            s{end+1} = pptxTextBox(x+cw,cy0+ch/2-0.2,cgap,0.4,'>',14,'232B3D',false,false,'ctr'); %#ok<AGROW>
        end
    end
    s{end+1} = pptxTextBox(0.9,5.45,11.5,0.9, [ ...
        'Fusing two asynchronous, differently-trusted sources like this is exactly why the ' ...
        'estimated position tracks far more smoothly than raw GPS alone -- the covariance ' ...
        'update naturally down-weights whichever source is currently less certain.'], 11.5,'8B95A8',false,true,'l');
    xml = pptxSlideWrap(strjoin(s,''));
end

function xml = pptxSlideLQRDeepDive()
    s = {};
    s{end+1} = pptxTextBox(0.9,0.5,6,0.35,'METHODOLOGY  --  DEEP DIVE 3 OF 3',12,'3DDCFF',true,false,'l');
    s{end+1} = pptxTextBox(0.9,0.9,11.5,0.6,'LQR Path Tracking',27,'FFFFFF',true,false,'l');
    s{end+1} = pptxTextBox(0.9,1.8,11.5,1.3, [ ...
        'The tracking-error dynamics (cross-track error, heading error, and their rates) are ' ...
        'linearized around the current estimated speed, and a discrete-time Riccati equation ' ...
        'is solved for the steering feedback gain that minimizes a weighted combination of ' ...
        'tracking error and steering effort. A curvature feedforward term is added on top, ' ...
        'so the controller does not have to rely on feedback alone to take a known corner.'], 12.5,'D7DCE6',false,false,'l');
    cards = {
        {'Q WEIGHT','Penalizes tracking error -- larger Q tracks tighter','FFC93D'}, ...
        {'R WEIGHT','Penalizes steering effort -- larger R drives gentler','E063FF'}, ...
        {'GAIN CACHING','Riccati re-solved only when speed drifts, not every step','3DDCFF'}, ...
        {'FEEDFORWARD','Curvature term anticipates corners instead of reacting late','5CE68C'} ...
    };
    cw=2.75; gap=0.25; x0=0.9; y0=3.4; ch=2.2;
    for i=1:4
        x = x0+(i-1)*(cw+gap); c = cards{i};
        s{end+1} = pptxRect(x,y0,cw,ch,'121927',c{3},1.2,0.07); %#ok<AGROW>
        s{end+1} = pptxTextBox(x+0.2,y0+0.2,cw-0.4,0.4,c{1},12.5,c{3},true,false,'l'); %#ok<AGROW>
        s{end+1} = pptxTextBox(x+0.2,y0+0.68,cw-0.4,1.4,c{2},10,'D7DCE6',false,false,'l'); %#ok<AGROW>
    end
    xml = pptxSlideWrap(strjoin(s,''));
end

function xml = pptxSlideTrackEnv()
    s = {};
    s{end+1} = pptxTextBox(0.9,0.5,6,0.35,'TEST ENVIRONMENT',12,'3DDCFF',true,false,'l');
    s{end+1} = pptxTextBox(0.9,0.9,11.5,0.6,'A track designed to stress-test the whole stack',27,'FFFFFF',true,false,'l');
    items = {
        {'CLOSED CIRCUIT','A cusp-free, rounded-rectangle superellipse track, generating continuous curvature the LQR feedforward term has to anticipate through every corner.','3DDCFF'}, ...
        {'LOW-GRIP ZONES','Two marked "slippery" segments scale down available tire grip locally, testing how the controller and estimator behave when the plant briefly departs from its nominal model.','FFC93D'}, ...
        {'REALISTIC DISTURBANCE','A small steering and longitudinal disturbance is applied to the true vehicle every step, so no two laps -- or two runs -- play out identically.','E063FF'}, ...
        {'SECOND ("GHOST") VEHICLE','An optional independently-driven second car with live collision-risk prediction, exercising the same estimation/control stack twice in parallel.','5CE68C'} ...
    };
    cw=5.6; ch=2.05; gapx=0.35; gapy=0.3; x0=0.9; y0=1.85;
    for i=1:4
        col = mod(i-1,2); row = floor((i-1)/2);
        x = x0+col*(cw+gapx); y = y0+row*(ch+gapy);
        it = items{i};
        s{end+1} = pptxRect(x,y,cw,ch,'121927',it{3},1.3,0.06); %#ok<AGROW>
        s{end+1} = pptxTextBox(x+0.25,y+0.18,cw-0.5,0.35,it{1},12.5,it{3},true,false,'l'); %#ok<AGROW>
        s{end+1} = pptxTextBox(x+0.25,y+0.56,cw-0.5,1.4,it{2},10,'D7DCE6',false,false,'l'); %#ok<AGROW>
    end
    xml = pptxSlideWrap(strjoin(s,''));
end

function xml = pptxSlideDashboard(imAspect)
    imgW = 9.8; imgH = imgW/imAspect; imgX = (13.333-imgW)/2; imgY = 1.5;
    s = {};
    s{end+1} = pptxTextBox(0.9,0.45,6,0.35,'LIVE SIMULATION',12,'3DDCFF',true,false,'l');
    s{end+1} = pptxTextBox(0.9,0.78,11.5,0.55,'A fully animated, real-time dashboard -- this run',27,'FFFFFF',true,false,'l');
    s{end+1} = pptxRect(imgX-0.05,imgY-0.05,imgW+0.1,imgH+0.1,'','232B3D',1,0);
    s{end+1} = pptxPicture('rId2',imgX,imgY,imgW,imgH);
    s{end+1} = pptxTextBox(0.9,imgY+imgH+0.18,11.5,0.5, ...
        'Track view, live gauges, lap timer, and three time-history plots update together in real time.',11,'8B95A8',false,true,'ctr');
    xml = pptxSlideWrap(strjoin(s,''));
end

function xml = pptxSlideDashboardWalkthrough()
    s = {};
    s{end+1} = pptxTextBox(0.9,0.5,6,0.35,'DASHBOARD WALKTHROUGH',12,'3DDCFF',true,false,'l');
    s{end+1} = pptxTextBox(0.9,0.9,11.5,0.6,'Every panel, explained',27,'FFFFFF',true,false,'l');
    items = {
        {'TRACK VIEW','True path, EKF estimate, short-horizon predicted rollout, and raw GPS fixes overlaid on the circuit, live.','3DDCFF'}, ...
        {'MINIMAP','A compact always-visible overview showing exactly where the car is on the full circuit.','E063FF'}, ...
        {'LAP / ELAPSED TIME','Current lap, best lap, and a live "Xx realtime" readout showing how closely the animation is tracking the wall clock.','FFC93D'}, ...
        {'SPEED / STEERING / G-FORCE','Analog-style gauges giving an at-a-glance read on the vehicle''s instantaneous state.','5CE68C'}, ...
        {'TIME-HISTORY PLOTS','Speed tracking, steering decomposition, and tracking/estimation error over the full run, updating alongside the animation.','8B95A8'} ...
    };
    iy = 1.85;
    for i = 1:5
        it = items{i};
        s{end+1} = pptxEllipse(0.9,iy+0.03,0.14,0.14,it{3},'',0); %#ok<AGROW>
        s{end+1} = pptxTextBox(1.25,iy-0.05,3.0,0.4,it{1},12.5,it{3},true,false,'l'); %#ok<AGROW>
        s{end+1} = pptxTextBox(4.5,iy-0.08,7.9,0.7,it{2},11,'D7DCE6',false,false,'l'); %#ok<AGROW>
        iy = iy + 0.85;
    end
    xml = pptxSlideWrap(strjoin(s,''));
end

function xml = pptxSlideResults(cxErr, ekfImprove, laps, bestLap, ekfPos, gpsPos)
    s = {};
    s{end+1} = pptxTextBox(0.9,0.5,6,0.35,'RESULTS',12,'3DDCFF',true,false,'l');
    s{end+1} = pptxTextBox(0.9,0.85,11.5,0.6,'This run''s key numbers',27,'FFFFFF',true,false,'l');
    cards = {
        {'RMS CROSS-TRACK ERROR', cxErr, 'FFC93D'}, ...
        {'EKF POSITION ERROR', ekfPos, '3DDCFF'}, ...
        {'RAW GPS ERROR', gpsPos, '8B95A8'}, ...
        {'EKF IMPROVEMENT', ekfImprove, '5CE68C'}, ...
        {'LAPS COMPLETED', laps, 'E063FF'}, ...
        {'BEST LAP TIME', bestLap, 'FFFFFF'} ...
    };
    cw = 3.55; ch = 1.65; gapx = 0.35; gapy = 0.35; x0 = 0.9; y0 = 1.85;
    for i = 1:6
        col = mod(i-1,3); row = floor((i-1)/3);
        x = x0 + col*(cw+gapx); y = y0 + row*(ch+gapy);
        c = cards{i};
        s{end+1} = pptxRect(x,y,cw,ch,'121927',c{3},1.3,0.06); %#ok<AGROW>
        s{end+1} = pptxTextBox(x+0.25,y+0.18,cw-0.5,0.35,c{1},10,'8B95A8',true,false,'l'); %#ok<AGROW>
        s{end+1} = pptxTextBox(x+0.25,y+0.52,cw-0.5,0.65,c{2},28,c{3},true,false,'l'); %#ok<AGROW>
    end
    s{end+1} = pptxTextBox(0.9,y0+2*(ch+gapy)+0.05,11.5,0.3,'Computed from this simulation run.',9,'8B95A8',false,true,'l');
    xml = pptxSlideWrap(strjoin(s,''));
end

function xml = pptxSlideFusionWins(posImprovePct)
    s = {};
    s{end+1} = pptxTextBox(0.9,0.5,6,0.35,'WHY FUSION WINS',12,'3DDCFF',true,false,'l');
    s{end+1} = pptxTextBox(0.9,0.85,11.5,0.6,'Filtered estimate vs. trusting GPS alone',27,'FFFFFF',true,false,'l');
    colW = 5.55; colH = 4.55; gap = 0.5; y0 = 1.85;
    leftPts = {'10 Hz updates -- the vehicle is "blind" between fixes','Meter-scale position noise fed straight to the controller', ...
        'Heading and speed drift open-loop between fixes, uncorrected','Steering reacts to noise, not the true path','Position error large and inconsistent run to run'};
    rightPts = {'50 Hz effective rate via IMU propagation between fixes','GPS and IMU cross-check each other every step', ...
        'Full state (position, heading, speed) continuously available','Steering tracks the smoothed, trusted estimate', ...
        ternaryStr(isnan(posImprovePct), 'Position error consistently lower (see Results)', ...
                   sprintf('Position error reduced by roughly %.0f%% on this run', posImprovePct))};
    s{end+1} = pptxRect(0.9,y0,colW,colH,'121927','8B95A8',1.5,0.08);
    s{end+1} = pptxTextBox(1.25,y0+0.3,colW-0.7,0.5,'Raw GPS Only',17,'8B95A8',true,false,'l');
    iy = y0+1.0;
    for i=1:5
        s{end+1} = pptxTextBox(1.25,iy,colW-0.7,0.6,['-  ' leftPts{i}],11.5,'D7DCE6',false,false,'l'); %#ok<AGROW>
        iy = iy+0.68;
    end
    s{end+1} = pptxRect(0.9+colW+gap,y0,colW,colH,'121927','3DDCFF',1.5,0.08);
    s{end+1} = pptxTextBox(0.9+colW+gap+0.35,y0+0.3,colW-0.7,0.5,'EKF Fused Estimate',17,'3DDCFF',true,false,'l');
    iy = y0+1.0;
    for i=1:5
        s{end+1} = pptxTextBox(0.9+colW+gap+0.35,iy,colW-0.7,0.6,['-  ' rightPts{i}],11.5,'D7DCE6',false,false,'l'); %#ok<AGROW>
        iy = iy+0.68;
    end
    xml = pptxSlideWrap(strjoin(s,''));
end

function xml = pptxSlideLQRTradeoff()
    s = {};
    s{end+1} = pptxTextBox(0.9,0.5,6,0.35,'CONTROLLER TUNING',12,'3DDCFF',true,false,'l');
    s{end+1} = pptxTextBox(0.9,0.85,11.5,0.6,'LQR weight sensitivity: accuracy vs. effort',27,'FFFFFF',true,false,'l');
    s{end+1} = pptxTextBox(0.9,1.7,11.5,0.9, [ ...
        'Sweeping the steering-effort penalty R while holding the tracking-error weight Q ' ...
        'fixed reproduces the textbook LQR tuning curve that guided the weight selection ' ...
        'used throughout this project.'], 12.5,'D7DCE6',false,false,'l');
    s{end+1} = pptxRect(0.9,2.75,5.6,3.5,'121927','5CE68C',1.4,0.07);
    s{end+1} = pptxTextBox(1.2,3.0,5.0,0.45,'SMALL R  --  Aggressive',14,'5CE68C',true,false,'l');
    s{end+1} = pptxTextBox(1.2,3.5,5.0,2.5,'Tracks the path tightly, responds fast to error -- at the cost of sharper, more frequent steering corrections and higher control effort.',11,'D7DCE6',false,false,'l');
    s{end+1} = pptxRect(6.85,2.75,5.6,3.5,'121927','E063FF',1.4,0.07);
    s{end+1} = pptxTextBox(7.15,3.0,5.0,0.45,'LARGE R  --  Gentle',14,'E063FF',true,false,'l');
    s{end+1} = pptxTextBox(7.15,3.5,5.0,2.5,'Smoother, lower-effort steering -- at the cost of looser tracking accuracy, especially through sharp corners.',11,'D7DCE6',false,false,'l');
    xml = pptxSlideWrap(strjoin(s,''));
end

function xml = pptxSlideValidation()
    s = {};
    s{end+1} = pptxTextBox(0.9,0.5,6,0.35,'VALIDATION',12,'3DDCFF',true,false,'l');
    s{end+1} = pptxTextBox(0.9,0.85,11.5,0.6,'Trusting a result means testing it more than once',27,'FFFFFF',true,false,'l');
    items = {
        {'NO-EKF BASELINE','The same controller re-run driven directly off raw GPS, headlessly, quantifying exactly how much the filter is buying in cross-track and position error.','3DDCFF'}, ...
        {'MONTE CARLO TRIALS','Repeated runs across many random sensor-noise seeds, so results reflect a distribution rather than one lucky (or unlucky) run.','E063FF'}, ...
        {'LQR SENSITIVITY SWEEP','A sweep of the steering-effort weight R, tracing the accuracy/effort trade-off curve rather than trusting one hand-picked tuning.','FFC93D'}, ...
        {'REALISTIC DISTURBANCE','Every run includes a physical process disturbance on the true vehicle, so validation reflects imperfect reality, not an idealized noiseless plant.','5CE68C'} ...
    };
    cw=5.6; ch=2.05; gapx=0.35; gapy=0.3; x0=0.9; y0=1.85;
    for i=1:4
        col = mod(i-1,2); row = floor((i-1)/2);
        x = x0+col*(cw+gapx); y = y0+row*(ch+gapy);
        it = items{i};
        s{end+1} = pptxRect(x,y,cw,ch,'121927',it{3},1.3,0.06); %#ok<AGROW>
        s{end+1} = pptxTextBox(x+0.25,y+0.18,cw-0.5,0.35,it{1},12.5,it{3},true,false,'l'); %#ok<AGROW>
        s{end+1} = pptxTextBox(x+0.25,y+0.56,cw-0.5,1.4,it{2},10,'D7DCE6',false,false,'l'); %#ok<AGROW>
    end
    xml = pptxSlideWrap(strjoin(s,''));
end

function xml = pptxSlideEngineering()
    s = {};
    s{end+1} = pptxTextBox(0.9,0.5,6,0.35,'ENGINEERING HIGHLIGHTS',12,'3DDCFF',true,false,'l');
    s{end+1} = pptxTextBox(0.9,0.85,11.5,0.6,'Built to actually run smoothly, not just simulate correctly',27,'FFFFFF',true,false,'l');
    items = {
        {'ADAPTIVE REAL-TIME PACING','The animation measures its own render cost every frame and self-tunes how much simulated time to advance between frames -- staying paced to real time on whatever machine it runs on, no manual tuning required.','3DDCFF'}, ...
        {'DETERMINISTIC VIDEO EXPORT','Recorded frames are resampled onto an evenly-spaced timeline before encoding, so exported video plays at a uniform, correct speed regardless of how the live render cadence varied while capturing it.','E063FF'}, ...
        {'SELF-CONTAINED REPORT EXPORT','Both the PDF report and this PowerPoint are generated directly by the script every run -- no external toolbox, no manual export step, no dependency beyond base MATLAB.','FFC93D'}, ...
        {'RESILIENT ANALYSIS PIPELINE','Each analysis section (diagnostics, baseline comparison, sensitivity sweep, report export) is independently fault-tolerant, so one failing section can''t silently take the others down with it.','5CE68C'} ...
    };
    cw=5.6; ch=2.05; gapx=0.35; gapy=0.3; x0=0.9; y0=1.85;
    for i=1:4
        col = mod(i-1,2); row = floor((i-1)/2);
        x = x0+col*(cw+gapx); y = y0+row*(ch+gapy);
        it = items{i};
        s{end+1} = pptxRect(x,y,cw,ch,'121927',it{3},1.3,0.06); %#ok<AGROW>
        s{end+1} = pptxTextBox(x+0.25,y+0.18,cw-0.5,0.35,it{1},12.5,it{3},true,false,'l'); %#ok<AGROW>
        s{end+1} = pptxTextBox(x+0.25,y+0.56,cw-0.5,1.4,it{2},10,'D7DCE6',false,false,'l'); %#ok<AGROW>
    end
    xml = pptxSlideWrap(strjoin(s,''));
end

function xml = pptxSlideRoadmap()
    s = {};
    s{end+1} = pptxTextBox(0.9,0.5,6,0.35,'PROJECT ROADMAP',12,'3DDCFF',true,false,'l');
    s{end+1} = pptxTextBox(0.9,0.85,11.5,0.6,'From physics model to full analysis suite',27,'FFFFFF',true,false,'l');
    steps = {
        {'PHASE 1','Vehicle Physics','Nonlinear bicycle model, linear & Pacejka tire options','3DDCFF'}, ...
        {'PHASE 2','State Estimation','EKF sensor fusion; parallel UKF for comparison','E063FF'}, ...
        {'PHASE 3','Optimal Control','LQR path tracking with curvature feedforward','FFC93D'}, ...
        {'PHASE 4','Live Visualization','Animated dashboard, minimap, gauges, video export','5CE68C'}, ...
        {'PHASE 5','Validation Suite','Monte Carlo trials, LQR sweep, PDF + PPTX report','FFFFFF'} ...
    };
    laneY = 3.9; x0 = 1.3; x1 = 12.0; n = 5;
    s{end+1} = pptxLine(x0,laneY,x1,laneY,'232B3D',2);
    spacing = (x1-x0)/(n-1);
    for i = 1:n
        cx = x0 + (i-1)*spacing;
        st = steps{i};
        s{end+1} = pptxEllipse(cx-0.13,laneY-0.13,0.26,0.26,'0A0E14',st{4},2.2); %#ok<AGROW>
        s{end+1} = pptxEllipse(cx-0.045,laneY-0.045,0.09,0.09,st{4},'',0); %#ok<AGROW>
        up = mod(i,2)==1;
        if up, textY = laneY-1.55; else, textY = laneY+0.35; end
        s{end+1} = pptxTextBox(cx-1.1,textY,2.2,0.3,st{1},9.5,st{4},true,false,'ctr'); %#ok<AGROW>
        s{end+1} = pptxTextBox(cx-1.1,textY+0.28,2.2,0.4,st{2},12.5,'FFFFFF',true,false,'ctr'); %#ok<AGROW>
        s{end+1} = pptxTextBox(cx-1.1,textY+0.68,2.2,0.85,st{3},9,'8B95A8',false,false,'ctr'); %#ok<AGROW>
    end
    xml = pptxSlideWrap(strjoin(s,''));
end

function xml = pptxSlideClosing()
    % BUGFIX: the decorative rings used to sit at x=[0.1,3.9], directly
    % underneath the "Thank You" headline text (which starts at x=0.9,
    % sz=46 -- easily wide enough to reach into that same region) and the
    % tagline below it, so the accent graphic visibly collided with the
    % text it was meant to sit clear of. The Title slide (pptxSlideTitle)
    % never has this problem because its rings live on the OPPOSITE side
    % of the slide from its text; mirroring that same left-text/
    % right-graphic split here fixes it the same way.
    s = {};
    s{end+1} = pptxEllipse(9.4,3.0,3.8,3.8,'','232B3D',1);
    s{end+1} = pptxEllipse(9.85,3.45,2.9,2.9,'','3DDCFF',1.5);
    s{end+1} = pptxEllipse(11.16,4.76,0.28,0.28,'FFC93D','',0);
    s{end+1} = pptxTextBox(0.9,2.5,8,0.6,'Built entirely in MATLAB',18,'8B95A8',false,false,'l');
    s{end+1} = pptxTextBox(0.9,3.05,8,1.1,'Thank You',46,'FFFFFF',true,false,'l');
    s{end+1} = pptxTextBox(0.9,4.15,8,0.45,'Dynamic Bicycle Model  |  Extended Kalman Filter  |  LQR Path Tracking  |  Live Simulation',13,'3DDCFF',false,false,'l');
    xml = pptxSlideWrap(strjoin(s,''));
end


function fH = renderReportTextPage(blocks, pageSize)
    % RENDERREPORTTEXTPAGE  Builds one dark-themed, styled text page for
    %   the PDF report (title page, methodology/summary) using plain
    %   text() objects -- no toolbox dependencies. 'blocks' is a cell
    %   array of structs, each with fields:
    %     .type  : 'title' | 'subtitle' | 'header' | 'body' | 'stat' | 'rule' | 'space'
    %     .text  : the string to display (unused for 'rule'/'space')
    %   Rendered top-to-bottom in a single column with consistent spacing.
    fH = figure('Color','k', 'Position',[100 60 pageSize(1) pageSize(2)], 'Visible','on');
    ax = axes(fH, 'Position',[0 0 1 1]); hold(ax,'on');
    axis(ax,[0 1 0 1]); axis(ax,'off');
    renderReportBlocks(ax, blocks, 0.94);
end


function fH = renderReportChartPage(blocks, figHandle, exportOpts, pageSize)
    % RENDERREPORTCHARTPAGE  Combines a short caption (header + one
    %   paragraph) with its chart image on ONE full page, instead of the
    %   caption getting its own separate, short, mostly-empty page
    %   immediately before the chart's own page.
    %
    %   BUGFIX (half-filled pages): the previous approach rendered each
    %   caption alone onto a page roughly a third of the normal page
    %   height, but the caption text itself -- using the same fixed-
    %   fraction-of-page-height spacing as every other text block --
    %   only ever filled about a fifth of even that reduced height,
    %   leaving a large blank dark rectangle below two or three lines of
    %   text. That happened once per chart (three of the report's eight
    %   pages were like this). Putting the caption directly above its
    %   own chart on one page -- sized the SAME as every other page in
    %   the report -- uses the space properly and also makes the whole
    %   PDF a consistent page size throughout (previously the caption
    %   pages were a different height than the title/summary pages).
    %
    %   figHandle is temporarily resized to the target page width (the
    %   same resize/export/restore pattern used throughout this report), exported to a temporary PNG so its
    %   colors/quality match the rest of the report, read back in, and
    %   drawn with image() -- NOT imshow(), which needs the Image
    %   Processing Toolbox this project deliberately avoids -- then
    %   figHandle is restored to its original size, unaffected by having
    %   been used here.
    tmpPng = [tempname() '.png'];
    origPos = figHandle.Position;
    scale = pageSize(1) / origPos(3);
    figHandle.Position = [origPos(1), origPos(2), pageSize(1), origPos(4)*scale];
    drawnow;
    exportgraphics(figHandle, tmpPng, exportOpts{:});
    figHandle.Position = origPos;
    drawnow;
    imgData = imread(tmpPng);
    delete(tmpPng);

    fH = figure('Color','k', 'Position',[100 60 pageSize(1) pageSize(2)], 'Visible','on');

    textFrac = 0.20;   % top fraction of the page reserved for the caption
    axTxt = axes(fH, 'Position',[0 (1-textFrac) 1 textFrac]); hold(axTxt,'on');
    axis(axTxt,[0 1 0 1]); axis(axTxt,'off');
    renderReportBlocks(axTxt, blocks, 0.82);

    axImg = axes(fH, 'Position',[0.03 0.02 0.94 (1-textFrac-0.04)]);
    image(axImg, imgData);
    axis(axImg, 'image'); axis(axImg, 'off');
end


function yEnd = renderReportBlocks(ax, blocks, yStart)
    % RENDERREPORTBLOCKS  Shared block-layout engine used by both
    %   renderReportTextPage (a full page of blocks) and
    %   renderReportChartPage (a short caption region above a chart).
    %   Draws 'blocks' top-to-bottom into axes 'ax' starting at
    %   normalized y = yStart, and returns the y position after the
    %   last block.
    %
    %   BUGFIX (content clipped off the bottom of the page): the spacing
    %   constants below were tightened from their original values. The
    %   Methodology/Summary page's blocks (a 10-line paragraph, a 6-card
    %   grid, several stat lines) added up to MORE vertical space than
    %   the page actually has -- starting at y=0.94 and subtracting every
    %   block's height landed at y=-0.005, i.e. past the bottom edge of
    %   the visible page, silently clipping the last parameter line off
    %   entirely. The constants here were tightened enough that the same
    %   worst-case page now ends around y=0.13 (a comfortable bottom
    %   margin) instead of running off the page.
    y = yStart;
    for i = 1:numel(blocks)
        b = blocks{i};
        switch b.type
            case 'title'
                text(ax, 0.5, y, b.text, 'Color','w', 'FontSize',22, 'FontWeight','bold', ...
                     'HorizontalAlignment','center', 'Interpreter','none');
                y = y - 0.075;
            case 'subtitle'
                text(ax, 0.5, y, b.text, 'Color',[0.65 0.75 0.95], 'FontSize',13, ...
                     'HorizontalAlignment','center', 'Interpreter','none');
                y = y - 0.05;
            case 'header'
                text(ax, 0.06, y, b.text, 'Color',[0.4 0.95 1.0], 'FontSize',14, 'FontWeight','bold', ...
                     'HorizontalAlignment','left', 'Interpreter','none');
                y = y - 0.045;
            case 'body'
                text(ax, 0.06, y, b.text, 'Color',[0.88 0.88 0.9], 'FontSize',10.5, ...
                     'HorizontalAlignment','left', 'VerticalAlignment','top', ...
                     'Interpreter','none', 'Units','normalized');
                nLines = max(1, sum(b.text==newline)+1);
                y = y - 0.024*nLines - 0.012;
            case 'stat'
                text(ax, 0.08, y, b.text, 'Color',[1 0.85 0.3], 'FontSize',10.5, ...
                     'FontName','Consolas', 'HorizontalAlignment','left', 'Interpreter','none');
                y = y - 0.028;
            case 'rule'
                plot(ax, [0.06 0.94], [y y], 'Color',[0.35 0.35 0.4], 'LineWidth',0.8);
                y = y - 0.03;
            case 'cards'
                % Grid of bordered stat cards: b.items is a cell array of
                % structs with .label, .value, .color (RGB triplet).
                % Visually ties the report back to the dashboard's own
                % LAP/CLOCK panel language (bordered box, colored big
                % number, small caption) instead of plain text lines.
                nCards = numel(b.items);
                nCols = min(3, nCards);
                nRows = ceil(nCards/nCols);
                cardW = (0.88)/nCols; cardGap = 0.016;
                cardH = 0.085;
                for ci = 1:nCards
                    row = floor((ci-1)/nCols); col = mod(ci-1,nCols);
                    cx = 0.06 + col*cardW;
                    cy = y - row*(cardH+cardGap) - cardH;
                    item = b.items{ci};
                    rectangle(ax, 'Position',[cx cy cardW-cardGap cardH], 'Curvature',0.15, ...
                              'EdgeColor',item.color, 'LineWidth',1.3, 'FaceColor',[0.06 0.07 0.09]);
                    text(ax, cx+0.012, cy+cardH-0.024, item.label, 'Color',[0.7 0.72 0.76], ...
                         'FontSize',8, 'FontWeight','bold', 'HorizontalAlignment','left', 'Interpreter','none');
                    text(ax, cx+0.012, cy+0.028, item.value, 'Color',item.color, ...
                         'FontSize',14, 'FontWeight','bold', 'FontName','Consolas', ...
                         'HorizontalAlignment','left', 'Interpreter','none');
                end
                y = y - nRows*(cardH+cardGap) - 0.016;
            case 'space'
                y = y - 0.022;
        end
    end
    yEnd = y;
end


function a = wrapAnglePi(a)
    % Wraps an angle to the interval [-pi, pi].
    a = atan2(sin(a), cos(a));
end


function drawHudCorners(ax, x0, y0, w, h, armLen, color)
    % DRAWHUDCORNERS  Draws four small L-shaped accent brackets just
    %   inside the four corners of a [x0 y0 w h] rectangle, on top of the
    %   given axes -- a lightweight "sci-fi HUD" accent used to dress up
    %   the LAP and CLOCK digital panels beyond a plain bordered box.
    inset = 0.05;
    corners = [x0+inset,   y0+inset;    % bottom-left
               x0+w-inset, y0+inset;    % bottom-right
               x0+inset,   y0+h-inset;  % top-left
               x0+w-inset, y0+h-inset]; % top-right
    signsX = [1 -1 1 -1];
    signsY = [1 1 -1 -1];
    for c = 1:4
        cx = corners(c,1); cy = corners(c,2);
        plot(ax, [cx cx+signsX(c)*armLen], [cy cy], 'Color',color, 'LineWidth',1.7);
        plot(ax, [cx cx], [cy cy+signsY(c)*armLen], 'Color',color, 'LineWidth',1.7);
    end
end


function mu = frictionScaleAt(s, zones, lowMu)
    % FRICTIONSCALEAT  Returns the grip multiplier (1.0 normally, lowMu
    %   inside a defined low-grip zone) for an arc-length position s.
    mu = 1.0;
    for i = 1:size(zones,1)
        if s >= zones(i,1) && s <= zones(i,2)
            mu = lowMu;
            return;
        end
    end
end


function [ex, ey] = confidenceEllipsePoints(muX, muY, Pxy, nSigma, nPts)
    % CONFIDENCEELLIPSEPOINTS  Standard nSigma confidence ellipse for a
    %   2D Gaussian with mean (muX,muY) and covariance Pxy: eigen-
    %   decompose the covariance, scale the eigenvector axes by
    %   nSigma*sqrt(eigenvalue), and map a unit circle through them.
    theta = linspace(0, 2*pi, nPts);
    Pxy = (Pxy + Pxy')/2;   % enforce symmetry (guards tiny numerical drift)
    [V, D] = eig(Pxy);
    radii = nSigma*sqrt(max(diag(D), 0));
    circlePts = [cos(theta); sin(theta)];
    ellipsePts = V*diag(radii)*circlePts;
    ex = muX + ellipsePts(1,:);
    ey = muY + ellipsePts(2,:);
end


function J = mpcCost(e0, Ad, Bd, Q, R, N, U)
    % MPCCOST  Quadratic cost of a candidate control sequence U (Nx1)
    %   over the linear error-dynamics model, starting from error state e0.
    e = e0;
    J = 0;
    for k = 1:N
        e = Ad*e + Bd*U(k);
        J = J + e'*Q*e + R*U(k)^2;
    end
end


function grad = mpcCostGradient(e0, Ad, Bd, Q, R, N, U)
    % MPCCOSTGRADIENT  Central-difference gradient of mpcCost wrt each
    %   element of U. The linear rollout is cheap (4x4 mat-vec ops), so a
    %   full finite-difference gradient over a ~15-step horizon costs only
    %   a few thousand trivial floating point operations -- fast enough
    %   to solve fresh every single control step in real time.
    eps_ = 1e-4;
    grad = zeros(N,1);
    for i = 1:N
        Up = U; Up(i) = Up(i) + eps_;
        Um = U; Um(i) = Um(i) - eps_;
        grad(i) = (mpcCost(e0,Ad,Bd,Q,R,N,Up) - mpcCost(e0,Ad,Bd,Q,R,N,Um)) / (2*eps_);
    end
end


function U = solveLinearMPC(e0, Ad, Bd, Q, R, N, maxU, Uwarm)
    % SOLVELINEARMPC  Finite-horizon linear MPC solved via projected
    %   gradient descent: minimize sum(e_k'*Q*e_k + R*u_k^2) subject to
    %   e_{k+1}=Ad*e_k+Bd*u_k and the HARD constraint |u_k| <= maxU,
    %   enforced by clipping after every gradient step (a standard,
    %   simple, and reliably-convergent method for convex box-constrained
    %   QPs). Warm-started from the previous solution (shifted by one
    %   step) for both speed and smoother receding-horizon behavior.
    if isempty(Uwarm) || numel(Uwarm) ~= N
        U = zeros(N,1);
    else
        U = [Uwarm(2:end); Uwarm(end)];
    end
    stepSize = 0.02;
    nIter = 20;
    for iter = 1:nIter
        grad = mpcCostGradient(e0, Ad, Bd, Q, R, N, U);
        U = U - stepSize*grad;
        U = max(min(U, maxU), -maxU);
    end
end


function out = rateLimitStep(newVal, prevVal, maxStep)
    % RATELIMITSTEP  Slews a command toward newVal by at most maxStep per
    %   call, relative to prevVal. Used to cap steering-angle rate of
    %   change so a sudden reference jump (e.g. from a curvature
    %   discontinuity) produces a smooth transition instead of an
    %   instantaneous jerky spike.
    diffVal = newVal - prevVal;
    diffVal = max(min(diffVal, maxStep), -maxStep);
    out = prevVal + diffVal;
end


function ticks = niceTickValues(maxVal, approxCount)
    % NICETICKVALUES  Picks "nice" round tick values from 0 to maxVal,
    %   the same kind of algorithm behind automatic axis tick placement
    %   (a simplified version of Heckbert's "nice numbers" method).
    %   Produces steps of 1, 2, or 5 (times a power of 10) rather than
    %   arbitrary numbers, so a gauge reads like a real speedometer
    %   (0, 5, 10, 15...) instead of odd fractional intervals.
    %
    %   Inputs:  maxVal (the top of the range to cover), approxCount
    %            (roughly how many ticks are desired)
    %   Output:  ticks (row vector from 0 up to the first nice value >= maxVal,
    %            trimmed to not exceed maxVal by more than one step)
    if maxVal <= 0
        ticks = 0;
        return;
    end
    roughStep = maxVal/approxCount;
    magnitude = 10^floor(log10(roughStep));
    residual = roughStep/magnitude;
    if residual < 1.5
        niceStep = 1*magnitude;
    elseif residual < 3
        niceStep = 2*magnitude;
    elseif residual < 7
        niceStep = 5*magnitude;
    else
        niceStep = 10*magnitude;
    end
    ticks = 0:niceStep:(ceil(maxVal/niceStep)*niceStep);
    ticks = ticks(ticks <= maxVal + 1e-9);
    if isempty(ticks)
        ticks = 0;
    end
end


function s = ternaryStr(cond, sTrue, sFalse)
    % Tiny ternary-style helper since MATLAB has no inline ?: operator.
    if cond
        s = sTrue;
    else
        s = sFalse;
    end
end


function [rmsE1, rmsPosErr] = runTrial(seed, useEKF, veh, lims, trackX, trackY, trackKappa, trackPsi, ...
                                        noise, Qekf, Rgps, Rimu, Qlqr, Rlqr, lqrIters, dt, tEndTrial, gpsDecim)
    % Runs one complete, HEADLESS (no plotting) trial of the full pipeline
    % for a given random seed, and returns tracking/estimation performance.
    %
    % useEKF = true  -> the controller is driven by the Extended Kalman
    %                   Filter estimate (full sensor-fusion pipeline).
    % useEKF = false -> the controller is driven directly by the raw,
    %                   noisy, low-rate GPS position, with heading and
    %                   speed obtained by simple finite differencing
    %                   between consecutive GPS fixes (a naive baseline
    %                   representative of an unfiltered GPS-only system).
    %
    % This function reuses the exact same dynamics/controller building
    % blocks as the main animated loop, so the comparison is apples-to-apples.

    rng(seed);
    nStepsTrial = round(tEndTrial/dt);

    X0 = trackX(1); Y0 = trackY(1); psi0 = trackPsi(1);
    trueState = [X0; Y0; psi0; lims.vmin; 0; 0];

    % --- EKF-side state (only meaningfully used if useEKF = true) ---
    estState = trueState + [0.8; -0.8; 0.05; 0; 0; 0];
    P = diag([1.0 1.0 0.05 1.0 1.0 0.05]);

    % --- naive raw-GPS-based state used when useEKF = false ---
    rawState = trueState;          % best available initial guess

    speedIntegral = 0;
    e1Sum2 = 0; posErrSum2 = 0; nSamples = 0;
    trialIdx = [];          % sticky path-search index (see projectToPath)
    prevDeltaTrial = 0;     % for steering rate limiting
    trialCachedK = [];
    trialCachedVx = NaN;

    for step = 1:nStepsTrial
        t = step*dt;

        if useEKF
            controlState = estState;
        else
            controlState = rawState;
        end

        vxDesign = max(controlState(4), 2.0);
        [trialIdx, psiPath, kappaPath, e1] = projectToPath(controlState(1), controlState(2), ...
                                                      trackX, trackY, trackKappa, trackPsi, trialIdx);
        e3 = wrapAnglePi(controlState(3) - psiPath);
        e2 = controlState(4)*sin(e3) + controlState(5)*cos(e3);
        e4 = controlState(6) - vxDesign*kappaPath;
        errState = [e1; e2; e3; e4];

        if isempty(trialCachedK) || abs(vxDesign - trialCachedVx) > 0.25
            [A_lin, B_lin] = lateralErrorModel(veh, vxDesign);
            trialCachedK = computeLQRGainDARE(A_lin, B_lin, Qlqr, Rlqr, dt, lqrIters);
            trialCachedVx = vxDesign;
        end
        K = trialCachedK;
        deltaFF = feedforwardSteering(kappaPath, vxDesign, veh.L, veh.Kus);
        deltaCmd = deltaFF - K*errState;
        deltaCmd = max(min(deltaCmd, lims.maxSteer), -lims.maxSteer);
        deltaCmd = rateLimitStep(deltaCmd, prevDeltaTrial, lims.maxSteerRate*dt);
        prevDeltaTrial = deltaCmd;

        vTarget = speedPlanner(kappaPath, lims.vmin, lims.vmax, lims.ayMax);
        vErr = vTarget - controlState(4);
        speedIntegral = max(min(speedIntegral + vErr*dt, 5), -5);
        axCmd = 1.2*vErr + 0.3*speedIntegral;
        axCmd = max(min(axCmd, lims.maxAccel), -lims.maxDecel);

        inputCmd = [deltaCmd; axCmd];

        % advance ground truth
        trueState = rk4Step(trueState, inputCmd, dt, veh);

        % sensors
        imuMeas = simulateIMU(trueState, inputCmd, veh, noise);
        haveGps = (mod(step, gpsDecim) == 0);
        if haveGps
            gpsMeas = trueState(1:2) + noise.gpsPos*randn(2,1);
        end

        if useEKF
            [estState, P] = ekfPredict(estState, P, inputCmd, dt, veh, Qekf);
            [estState, P] = ekfUpdateIMU(estState, P, imuMeas, inputCmd, veh, Rimu);
            if haveGps
                [estState, P] = ekfUpdateGPS(estState, P, gpsMeas, Rgps);
            end
            posErrSample = hypot(trueState(1)-estState(1), trueState(2)-estState(2));
        else
            % naive baseline: propagate kinematically between GPS fixes,
            % using the (noisy) IMU yaw rate for heading and integrating
            % the COMMANDED acceleration for speed, then snap POSITION
            % ONLY to GPS on fixes.
            %
            % BUGFIX: this used to re-derive speed at every GPS fix by
            % finite-differencing two consecutive noisy GPS positions
            % 0.1s apart: estVel = (gpsMeas - lastGpsForDiff)/dtGps. That
            % looks like reasonable dead reckoning, but the noise math
            % doesn't work out -- differencing two independent ~0.6 m GPS
            % fixes 0.1 s apart amplifies into a velocity estimate with
            % roughly sqrt(2)*0.6/0.1 = 8.5 m/s of standard deviation,
            % LARGER than the entire 5-14 m/s speed range it's supposed
            % to measure. That noise fed straight into the quadratic
            % understeer feedforward term (Kus*vx^2*kappa) produced huge,
            % erratic steering spikes right as the car entered a corner,
            % which is what actually drove the "no-EKF" baseline physically
            % off the track -- not just a noisier trace, a full runaway
            % (RMS cross-track error in the hundreds of meters). Simply
            % integrating the known commanded accel (the same open-loop
            % approach already used for heading above) avoids that noise
            % amplification entirely, while still being a genuinely
            % "naive" estimate with no proper filtering -- it drifts
            % between fixes and never fuses anything, it just isn't
            % gratuitously noisier than the signal it's tracking.
            rawState(3) = rawState(3) + imuMeas(1)*dt;
            rawState(4) = max(rawState(4) + axCmd*dt, 0.1);
            rawState(1) = rawState(1) + rawState(4)*cos(rawState(3))*dt;
            rawState(2) = rawState(2) + rawState(4)*sin(rawState(3))*dt;
            rawState(5) = 0; rawState(6) = imuMeas(1);
            if haveGps
                rawState(1) = gpsMeas(1);
                rawState(2) = gpsMeas(2);
                lastGpsForDiff = gpsMeas;
                lastGpsTime = t;
            end
            posErrSample = hypot(trueState(1)-rawState(1), trueState(2)-rawState(2));
        end

        e1Sum2 = e1Sum2 + e1^2;
        posErrSum2 = posErrSum2 + posErrSample^2;
        nSamples = nSamples + 1;
    end

    rmsE1 = sqrt(e1Sum2/nSamples);
    rmsPosErr = sqrt(posErrSum2/nSamples);
end
