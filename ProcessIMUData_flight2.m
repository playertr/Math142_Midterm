% ProcessIMUDataJ825R.m
% Primarily written by Erik Spjut
% Last modified by Tim Player 3/28/2019 tplayer@hmc.edu
% In cooperation with the marine brokers at Wells Fargo Insurance
% (That is a joke -- that message is broadcast by  NOAA Weather Radio 
% Station WWG24 in Seattle)
% Actually in collobation with the E178 class

%% Toggle separate figures
separateFigs = true;

%% LoadFile
% The easiest way to read in the XTRA or BASE .csv is to read it in as
% a table.
AIM_Data=readtable('4_27_flight2_trim.csv');
% You have to actually look at the column headings to make sure
% you have the correct column names.
% Because the AIM XTRA uses different sample rates for the different
% sensors, most columns (except for the z-axis accelerometer and associated
% time) will have NaNs for much of the column. They show up once the actual
% column data cease. THe easiest way to remove them is with a conditional
% using isfinite.

% The data in a column heading may be accessed by appending with a dot, ".",
% and either the column name, e.g., .time, or the column number in
% parentheses, e.g., .(1). The example below shows how to plot two columns
% using either method of addressing.

% %% PlotDotName
% plot(AIM_Data.time,AIM_Data.GyroZ)
% title('Test Plot Table Columns .name')
% xlabel('time (s)')
% ylabel('\omega_z (°/s)')
% %% PlotDotNumber
% plot(AIM_Data.(1),AIM_Data.(4))
% title('Test Plot Table Columns .(#)')
% xlabel('time (s)')
% ylabel('\omega_z (°/s)')

%% SearchApogee
% Find the apogee time
accelTimePost=AIM_Data.time(AIM_Data.time>=0);
accelZPost=AIM_Data.acceleration(AIM_Data.time>=0);

if(separateFigs) figure(); end
plot(accelTimePost,accelZPost)
title('z-axis acceleration')
xlabel('time (s)')
ylabel('a_z (g)')

%% SetApogee
apogeeTime = 22.98; % Determined by looking at the above plot and zooming in.
% A clever person may be able to automate the selection process.
sp = 0.001; % Set the sample period
timeArr = 0:sp:apogeeTime; % The Z accelerometer has an approximate 
% sample rate of 100 SPS, or a period of 0.01 s. Using a precise time
% array makes a lot of things simpler. We'll resample later using timeArr.

% Table data are column vectors.
% You should transpose most table data before using.

%% DisplayTemperatures
% Below is the procedure for calculating the temperatures of the
% thermistors from the voltages in Channels C & D. Note the transposition
% and the isfinite to remove NaNs.
VoltC=transpose(AIM_Data.CVoltage(isfinite(AIM_Data.CVoltage)));
[TCC,TCK]=TfromV(VoltC);
TtimeC=transpose(AIM_Data.time_6(isfinite(AIM_Data.time_6)));
VoltD=transpose(AIM_Data.DVoltage(isfinite(AIM_Data.DVoltage)));
[TDC,TDK]=TfromV(VoltD);
TtimeD=transpose(AIM_Data.time_7(isfinite(AIM_Data.time_7)));
if(separateFigs) figure(); end
plot(TtimeC,TCC,TtimeD,TDC)
title('Temperature vs. Time Profile')
xlabel('time (s)')
ylabel('Temperature (°C)')
legend('Channel C','Channel D')

%% GetGPSdata
timeGPS=transpose(AIM_Data.time_14(isfinite(AIM_Data.time_14)));
timeGPSDesc = timeGPS(timeGPS>apogeeTime);
lat=transpose(AIM_Data.lat(isfinite(AIM_Data.lat)));
long=transpose(AIM_Data.long(isfinite(AIM_Data.long)));
GPSMSL=transpose(AIM_Data.GPSMSL(isfinite(AIM_Data.GPSMSL)));
GPSMSLDesc = GPSMSL(timeGPS>apogeeTime);

if(separateFigs) figure(); end
plot(timeGPS,lat, 'g*', timeGPS,long, 'b*', timeGPS,GPSMSL, 'r*')
title('GPS Data')
xlabel('time (s)')
ylabel('Latitude (°), Longitude (°), Altitude (m MSL)')
legend('lat', 'long', 'alt')

%% CalcGPSBaseline
latBase = mean(lat(timeGPS<0)); % Remove ; to get for mag vec & gravity
longBase = mean(long(timeGPS<0)); % Remove ; to get for mag vec
GPSMSLBase = mean(GPSMSL(timeGPS<0)); % Remove ; to get for mag vec & gravity

%% CalcGPSPositionChanges
latDel = 110574*(lat-latBase); % Distance in m
longDel = 111320*(long-longBase).*cos(pi/180*(lat+latBase)/2); % Distance in m
GPSAGL = GPSMSL - GPSMSLBase;

if(separateFigs) figure(); end
plot(timeGPS,longDel, 'g*', timeGPS,latDel, 'b*', timeGPS,GPSAGL, 'r*') % 
title('GPS Data converted to Position Change')
xlabel('time (s)')
ylabel('Distance (m)')
legend('x (East)', 'y (North)', 'AGL') % 

%% GetMagnetometerData
% The magnetometer chip is a left-handed coordinate system. To
% align X, Y, and Z with ENU, we need to flip the sign on X and Y.
timeMag=transpose(AIM_Data.time_8(isfinite(AIM_Data.time_8)));
magX = -transpose(AIM_Data.MagX(isfinite(AIM_Data.MagX))); % Flip X
magY = -transpose(AIM_Data.MagY(isfinite(AIM_Data.MagY))); % Flip Y
magZ =  transpose(AIM_Data.MagZ(isfinite(AIM_Data.MagZ))); % No flip

if(separateFigs) figure(); end
plot(timeMag,magX, timeMag,magY, timeMag,magZ)
title('Raw Magnetometer Data')
xlabel('time (s)')
ylabel('Magnetic Field Strength')
legend('m_x', 'm_y', 'm_z')

%% ScaleMagData
magXMax = max(magX);
magXMin = min(magX);
magYMax = max(magY);
magYMin = min(magY);
magZMax = max(magZ);
magZMin = min(magZ);
magXsc = (2*magX-(magXMax+magXMin))/(magXMax-magXMin);
magYsc = (2*magY-(magYMax+magYMin))/(magYMax-magYMin);
magZsc = (2*magZ-(magZMax+magZMin))/(magZMax-magZMin);
magMagsc = sqrt(magXsc.^2 + magYsc.^2 + magZsc.^2);

if(separateFigs) figure(); end
plot(timeMag,magXsc, timeMag,magYsc, timeMag,magZsc, timeMag,magMagsc)
title('Scaled Magnetometer Data')
xlabel('time (s)')
ylabel('Magnetic Field Strength')
legend('m_x', 'm_y', 'm_z','mag')

%% RenormalizeMagData
% The previous graph shows that the normalized magnitude isn't always 1.00
% so we renormalize so that all of the vectors have magnitude 1.
magXsc = magXsc./magMagsc;
magYsc = magYsc./magMagsc;
magZsc = magZsc./magMagsc;

if(separateFigs) figure(); end
plot(timeMag,magXsc, timeMag,magYsc, timeMag,magZsc)
title('Scaled Renormalized Magnetometer Data')
xlabel('time (s)')
ylabel('Magnetic Field Strength')
legend('m_x', 'm_y', 'm_z')

%% Get Magnetometer Initial Orientation
% Get the magnetometer pre-launch data and calculate the baselines.
% We've skipped getting the pre-launch data as separate variables.
magX0 = mean(magXsc(timeMag<0));
magY0 = mean(magYsc(timeMag<0));
magZ0 = mean(magZsc(timeMag<0));
mag0 = norm([magX0 magY0 magZ0]);
magX0 = magX0/mag0;
magY0 = magY0/mag0;
magZ0 = magZ0/mag0;
magVec0 = [magX0; magY0; magZ0];

% Include the earth magnetic vector from https://www.ngdc.noaa.gov/geomag-web/#igrfwmm
% Use the pre-launch GPS Baselines from above to look up the vectors.
% N 34.4962, W 116.9593, MSL 861.9322 m, 9 July 2017
magN = 23431.2;
magE = 4864.6;
magD = 40623.4;
magUp = - magD;
magEarthVec = [magE; magN; magUp];
magEarthNorm = norm(magEarthVec);
magEarthVecNorm = magEarthVec/magEarthNorm;

% Calculate the variance in roll angle measurements
rollVar = var(atan2(magYsc(timeMag<0),magXsc(timeMag<0)));

%% ResampleMag
% Get the magnetometer data between launch and apogee by resamling at the timeArr
% values.
magXrs = interp1(timeMag,magXsc,timeArr,'pchip');
magYrs = interp1(timeMag,magYsc,timeArr,'pchip');
magZrs = interp1(timeMag,magZsc,timeArr,'pchip');

% Create the set of local measured magnetometer vectors. They are column
% vectors of the form [mx; my; mz] and the column index corresponds to the
% index of a given timeArr value
magVec = [magXrs; magYrs; magZrs]; % Set of magnetometer vectors
%renormalize again
n = length(timeArr);
for i = 1:n
    magVec(1:3,i) = magVec(1:3,i)/norm(magVec(1:3,i));
    roll(i) = atan2(magVec(2,i),magVec(1,i)); % May want to move this
    %tilt(i) = atan2(magVec(3,i),norm([magVec(2,i) magVec(1,i)]));
end

if(separateFigs) figure(); end

plot(timeArr,magVec(1,1:n),timeArr,magVec(2,1:n),timeArr,magVec(3,1:n))
title('Scaled and Normalized Magnetometer Data')
xlabel('time (s)')
ylabel('Magnetic Field Strength')
legend('m_x', 'm_y', 'm_z')

%% RollVariance
% We're trying to determine typical jump sizes from one time step to the
% next.
droll = (unwrap(roll(2:n))-unwrap(roll(1:n-1)));

if(separateFigs) figure(); end
plot(timeArr(2:n),droll);
title('numerical derivative of roll')
xlabel('time (s)')
ylabel('droll/dt')
actRollVar = var(droll);

%% GetGyroData
% Getting IMU data. Note the transposition and NaN removal.
timeGyro=transpose(AIM_Data.time(isfinite(AIM_Data.time)));
gyroX = transpose(AIM_Data.GyroX(isfinite(AIM_Data.GyroX)));
gyroY = transpose(AIM_Data.GyroY(isfinite(AIM_Data.GyroY)));
gyroZ = transpose(AIM_Data.GyroZ(isfinite(AIM_Data.GyroZ)));

if(separateFigs) figure(); end
plot(timeGyro,gyroZ, timeGyro,gyroX, timeGyro,gyroY)
title('Gyro Data')
xlabel('time (s)')
ylabel('Rotation Rate (°/s)')
legend('\omega_z', '\omega_x', '\omega_y')

%% GetAccelData
timeAccel=transpose(AIM_Data.time_1(isfinite(AIM_Data.time_1)));
accelZ=transpose(AIM_Data.acceleration(isfinite(AIM_Data.acceleration)));
timeAccelLat=transpose(AIM_Data.time_2(isfinite(AIM_Data.time_2)));
accelX=transpose(AIM_Data.lat_XAccel_(isfinite(AIM_Data.lat_XAccel_)));
accelY=transpose(AIM_Data.lat_YAccel_(isfinite(AIM_Data.lat_YAccel_)));

if(separateFigs) figure(); end
plot(timeAccel,accelZ, timeAccelLat,accelX, timeAccelLat,accelY)
title('Accelerometer Data')
xlabel('time (s)')
ylabel('Acceleration (g''s)')
legend('a_z', 'a_x', 'a_y')

% Get local gravitational field from https://www.sensorsone.com/local-gravity-calculator/
%  N 34.4962, GPS alt 861.9322 m

localg = 9.79424;

%% ConvertUnits
% Convert IMU data to radians per second and meters per second.
gyroX=gyroX*pi/180;
gyroY=gyroY*pi/180;
gyroZ=gyroZ*pi/180;
accelX=accelX*localg;
accelY=accelY*localg;
accelZ=accelZ*localg;

%% AccelPreLaunch
% Get the accel pre-launch data. Unlike the gyro data, it's not easy to use
% these data to calculate baselines or offsets unless you're certain that
% the rocket is perfectly vertical on the pad. However, if you've
% determined the baselines separately, these are really useful for the
% TRIAD method gravity vector.
accelZTimePre=timeAccel(timeAccel<0);
accelZPre=accelZ(timeAccel<0);
accelXYTimePre=timeAccelLat(timeAccelLat<0);
accelXPre=accelX(timeAccelLat<0);
accelYPre=accelY(timeAccelLat<0);

if(separateFigs) figure(); end
plot(accelZTimePre,accelZPre,accelXYTimePre,accelXPre,accelXYTimePre,accelYPre)
title('Pre-Launch Accelerometer Data')
xlabel('time (s)')
ylabel('Acceleration (m/s)')
legend('a_z', 'a_x', 'a_y')

% AccelPadVector
accelPreVec = [mean(accelXPre); mean(accelXPre); mean(accelZPre)];

%% Resample
% Get the accel data between launch and apogee by resamling at the timeArr
% values.
accelZrs = interp1(timeAccel,accelZ,timeArr,'pchip');
accelXrs = interp1(timeAccelLat,accelX,timeArr,'pchip');
accelYrs = interp1(timeAccelLat,accelY,timeArr,'pchip');

% Create the set of local measured acceleration vectors. They are column
% vectors of the form [ax; ay; az] and the column index corresponds to the
% index of a given timeArr value
accelVec = [accelXrs; accelYrs; accelZrs]; % Set of acceleration vectors

% Get the gyro pre-launch data and calculate the baselines or offsets.
% We've skipped getting the pre-launch data as separate variables.
gyroXbase = mean(gyroX(timeGyro<0));
gyroYbase = mean(gyroY(timeGyro<0));
gyroZbase = mean(gyroZ(timeGyro<0));

% Get the variance of the roll gyro
rgVar = var(gyroZ(timeGyro<0));

% Get the gyro data between launch and apogee by resamling at the timeArr
% values.
gyroXrs = interp1(timeGyro,gyroX,timeArr,'pchip');
gyroYrs = interp1(timeGyro,gyroY,timeArr,'pchip');
gyroZrs = interp1(timeGyro,gyroZ,timeArr,'pchip');

% Get z-gyro variance
% We're trying to determine typical jump sizes from one time step to the
% next.
dGyroZ = (gyroZrs(2:n) - gyroZrs(1:n-1));

if(separateFigs) figure(); end
plot(timeArr(2:n),dGyroZ)
title('numerical derivative of gyro z')
xlabel('time (s)')
ylabel('dwz/dt')
actGyroZVar = var(dGyroZ);

%% RollAngle
% This section can be removed, but remove references in further plotting
rGyro(1) = roll(1);
rGyrocd(1) = roll(1);
% pGyro(1) = 0; % tilt(1); % Fix starting angle
% yGyro(1) = 0; % tilt(1); % Fix starting angle
for i = 1:n-1
    rGyro(i+1) = rGyro(i) + sp*(gyroZrs(i)-gyroZbase);
    rGyrocd(i+1) = rGyrocd(i) + 0.5*sp * ...
        ((gyroZrs(i+1)-gyroZbase)+(gyroZrs(i)-gyroZbase));
%     pGyro(i+1) = pGyro(i)+ sp*(gyroXrs(i)-gyroXbase);
%     yGyro(i+1) = yGyro(i)+ sp*(gyroYrs(i)-gyroYbase);
end

%% RollSmoothing
roll = unwrap(roll);
gyTry = (gyroZrs - gyroZbase);
% Be sure to change the sample rate and the weight in smooth if you change
% the sample rate in the resampling
[res, theta] = smooth(roll, rollVar ,gyTry, rgVar, sp, roll(1), 10*sp);

%% CheckRollSmothing
% Replace GyroZrs with smoothed rates
gyroZrsSmooth(1:n-1) = theta(2:n)-theta(1:n-1);
gyroZrsSmooth(n) = gyroZrsSmooth(n-1);
gyroZrsSmooth = gyroZrsSmooth/sp;
% Create the set of local measured rotation-rate vectors. They are column
% vectors of the form [wx; wy; wz] and the column index corresponds to the
% index of a given timeArr value
rotRateVec = [gyroXrs-gyroXbase; gyroYrs-gyroYbase; gyroZrsSmooth];

%% RollKalmanFilter
roll = unwrap(roll);
Fk = [1 0; sp 1];
FkT = transpose(Fk);
Hk = [1 0; 0 1];
HkT = transpose(Hk);
Pk1k1(1:2,1:2,1) = [0 0; 0 0];
xk1k1(1:2,1) = [gyTry(1); roll(1)];
Rk = [rgVar 0; 0 rollVar];
Qk = [actGyroZVar 0; 0 actRollVar]; % Have to guess these values for now
II = [1 0; 0 1];
yk(1) = 0;
ykk(1) = 0;
for i = 1:n-1
    % Predict
    xkk1(1:2,i) = Fk*xk1k1(1:2,i);
    Pkk1(1:2,1:2,i) = Fk*Pk1k1(i)*FkT + Qk;
    % Update
    zk(1:2,i+1) = [gyTry(i+1) roll(i+1)];
    yk(1:2,i+1) = zk(1:2,i+1) - Hk*xkk1(1:2,i);
    Sk(1:2,1:2,i+1) = Rk + Hk*Pkk1(1:2,1:2,i)*HkT;
    Kk(1:2,1:2,i+1) = Pkk1(1:2,1:2,i)*HkT*inv(Sk(1:2,1:2,i+1)); % Kalman gain
    xk1k1(1:2,i+1) = xkk1(1:2,i) + Kk(1:2,1:2,i+1)*yk(1:2,i+1);
    Pk1k1(1:2,1:2,i+1) = (II - Kk(1:2,1:2,i+1)*Hk)*Pkk1(1:2,1:2,i)* ...
        transpose(II - Kk(1:2,1:2,i+1)*Hk)...
        + Kk(1:2,1:2,i+1)*Rk*transpose(Kk(1:2,1:2,i+1));
    ykk(1:2,i+1) = zk(1:2,i+1) - Hk*xk1k1(1:2,i+1); % Measurement post-fit residual
end
rgKalman = xk1k1(2,1:n); % The roll angle put out by the Kalman Filter
rdotKalman = xk1k1(1,1:n); % The roll rotation rate from the Kalman Filter.
rotRateVecK = [gyroXrs-gyroXbase; gyroYrs-gyroYbase; rdotKalman];

%% CheckRollRates
if(separateFigs) figure(); end
plot(timeArr, 180/pi*gyTry, timeArr, 180/pi*gyroZrsSmooth, timeArr, 180/pi*rdotKalman,'.')
title('New vs Old Rates')
xlabel('time (s)')
ylabel('Roll Rate (°/s)')
legend('rGyro', 'smooth', 'Kalman')

%% Check Rolls
if(separateFigs) figure(); end
plot(timeArr, 180/pi*roll, timeArr, 180/pi*rGyrocd, timeArr, 180/pi*theta ...
    ,timeArr, 180/pi*rgKalman)
title('Roll Method Comparison')
xlabel('time (s)')
ylabel('Roll Angle (°)')
legend('roll', 'rGyro', 'smooth', 'Kalman')

%% Check roll using Quiver

if(separateFigs) figure(); end

plot(timeArr,magVec(1,1:n),timeArr,magVec(2,1:n),timeArr,magVec(3,1:n))
clf
hold on
for i = 50000:n
    scatter(magVec(1,i), magVec(2,i), 'r')
    scatter(cos(roll(i)), sin(roll(i)), 'b')
    axis([-1 1 -1 1])
    drawnow update
    pause(sp)
end
title('Scaled and Normalized Magnetometer Data')
xlabel('time (s)')
ylabel('Magnetic Field Strength')
legend('m_x', 'm_y', 'm_z')


%% InitiateLoop
% Initiate quaternion. This is the point to apply the TRIAD method to
% determine the initial alignment of the rocket on the pad with ENU.
[qr(1:4,1), RM0] = TRIAD([0;0;localg], magEarthVec, accelPreVec, magVec0);
%qr(1:4,1) = [0 0 0 0]; %TIM: dummy values, missing TRIAD()
RM0 = 0; 

% Initialize the rotation quaternion array

% Initiate the global acceleration, velocity and position vectors.
accelGlobVec(1:3,1) = [0;0;0];
velGlobVec(1:3,1) = [0;0;0];
posGlobVec(1:3,1) = [0;0;0];

% Initiate the local corrected acceleration and velocity vectors.
accelLocVec(1:3,1) = [0;0;0]; % Used for thrust calculations.
velLocVec(1:3,1) = [0;0;0]; % Used for drag calculations.
velLoc(1) = 0;
vel1DLoc(1) = 0;
vel1DRaw(1) = 0;

%% RunLoop
% Prep for the for loop.
for i =1:n-1
    % Calculate the qdot quaternion
    qdot = qdotCalc(qr(1:4,i),rotRateVec(1:3,i));
    % Take one explicit Euler step forward with rotation quaternion
    qr(1:4,i+1)=qr(1:4,i)+sp*qdot;
    % Renormalize quaternion
    qr(1:4,i+1) = qr(1:4,i+1)/norm(qr(1:4,i+1));
    % The slightly faster but less accurate method of normalization is
    % commented out below.
    %qr(1:4,i+1)=qr(1:4,i+1)*(1.5-0.5*(qr(1,i+1)^2+qr(2,i+1)^2+qr(3,i+1)^2+...
    %    qr(4,i+1)^2));
    accelGlobVec(1:3,i+1)=rotVecToGlob(accelVec(1:3,i+1),qr(1:4,i+1)) ...
        - [0;0;localg]; %We are using the local ground value of g, but
        % g varies with altitude, and we're not accounting for the change.
    accelLocVec(1:3,i+1) = rotVecToLoc(accelGlobVec(1:3,i+1),qr(1:4,i+1));
    velGlobVec(1:3,i+1) = velGlobVec(1:3,i) + sp*accelGlobVec(1:3,i+1);
    velLocVec(1:3,i+1) = rotVecToLoc(velGlobVec(1:3,i+1),qr(1:4,i+1));
    velLoc(i+1)= norm(velLocVec(1:3,i+1));
    vel1DLoc(i+1) = vel1DLoc(i) +  sp*accelLocVec(3,i+1);
    vel1DRaw(i+1) = vel1DRaw(i) +  sp*(accelZrs(i+1)-localg);
    posGlobVec(1:3,i+1) = posGlobVec(1:3,i)+ sp*velGlobVec(1:3,i+1);
end

%% InitiateLoopK
% Initiate quaternion. This is the point to apply the TRIAD method to
% determine the initial alignment of the rocket on the pad with ENU.

[qrK(1:4,1), RM0] = TRIAD([0;0;localg], magEarthVec, accelPreVec, magVec0);
%qrK(1:4,1) = [0 0 0 0 ]; %TIM: I initialized these to zero bc we're missing TRIAD
RM0 = 0;

% Initialize the rotation quaternion array

% Initiate the global acceleration, velocity and position vectors.
accelGlobVecK(1:3,1) = [0;0;0];
velGlobVecK(1:3,1) = [0;0;0];
posGlobVecK(1:3,1) = [0;0;0];

% Initiate the local corrected acceleration and velocity vectors.
accelLocVecK(1:3,1) = [0;0;0]; % Used for thrust calculations.
velLocVecK(1:3,1) = [0;0;0]; % Used for drag calculations.
velLocK(1) = 0;
vel1DLocK(1) = 0;

%% RunLoopK
% Prep for the for loop.
for i =1:n-1
    % Calculate the qdot quaternion
    qdotK = qdotCalc(qrK(1:4,i),rotRateVecK(1:3,i));
    % Take one explicit Euler step forward with rotation quaternion
    qrK(1:4,i+1)=qrK(1:4,i)+sp*qdotK;
    % Renormalize quaternion
    qrK(1:4,i+1) = qrK(1:4,i+1)/norm(qrK(1:4,i+1));
    % The slightly faster but less accurate method of normalization is
    % commented out below.
    %qr(1:4,i+1)=qr(1:4,i+1)*(1.5-0.5*(qr(1,i+1)^2+qr(2,i+1)^2+qr(3,i+1)^2+...
    %    qr(4,i+1)^2));
    accelGlobVecK(1:3,i+1)=rotVecToGlob(accelVec(1:3,i+1),qrK(1:4,i+1)) ...
        - [0;0;localg]; %We are using the local ground value of g, but
        % g varies with altitude, and we're not accounting for the change.
    accelLocVecK(1:3,i+1) = rotVecToLoc(accelGlobVecK(1:3,i+1),qrK(1:4,i+1));
    velGlobVecK(1:3,i+1) = velGlobVecK(1:3,i) + sp*accelGlobVecK(1:3,i+1);
    velLocVecK(1:3,i+1) = rotVecToLoc(velGlobVecK(1:3,i+1),qrK(1:4,i+1));
    velLocK(i+1)= norm(velLocVecK(1:3,i+1));
    vel1DLocK(i+1) = vel1DLocK(i) +  sp*accelLocVecK(3,i+1);
    posGlobVecK(1:3,i+1) = posGlobVecK(1:3,i)+ sp*velGlobVecK(1:3,i+1);
end

%% PlotGlobalAcceleration
if(separateFigs) figure(); end
plot(timeArr,accelGlobVec(1,1:n),timeArr,accelGlobVec(2,1:n), ...
    timeArr,accelGlobVec(3,1:n),timeArr,accelGlobVecK(1,1:n),timeArr, ...
    accelGlobVecK(2,1:n),timeArr,accelGlobVecK(3,1:n))
title('Global Acceleration Data')
xlabel('time (s)')
ylabel('Acceleration (m/s^2)')
legend('a_x', 'a_y', 'a_z','a_{xK}', 'a_{yK}', 'a_{zK}')

%% PlotZAcceleration
% Compare raw accelerometer data, raw with g subtracted, Global with
% smoothing, Global with Kalman, Local with smoothing, Local with Kalman

if(separateFigs) figure(); end
plot(timeArr,accelZrs,timeArr,accelZrs-localg,timeArr, ...
    accelGlobVec(3,1:n),timeArr,accelGlobVecK(3,1:n), ...
    timeArr,accelLocVec(3,1:n),timeArr,accelLocVecK(3,1:n))
title('z Acceleration Data')
xlabel('time (s)')
ylabel('Acceleration (m/s^2)')
legend('raw', 'raw-g', 'Global-s','Global-K','Local-s','Local-K')

%% PlotGlobalVelocity
if(separateFigs) figure(); end
plot(timeArr,velGlobVec(1,1:n),timeArr,velGlobVec(2,1:n), ...
    timeArr,velGlobVec(3,1:n),timeArr,velGlobVecK(1,1:n),timeArr,velGlobVecK(2,1:n), ...
    timeArr,velGlobVecK(3,1:n))
title('Global Velocity Data')
xlabel('time (s)')
ylabel('Velocity (m/s)')
legend('v_x', 'v_y', 'v_z','v_{xK}', 'v_{yK}', 'v_{zK}')

%% PlotZVelocity
% Compare Global with smoothing,
% Global with Kalman, Local with smoothing, Local with Kalman, Integral of
% Local z acceleration with smoothing, Locak z acceleration with Kalman
if(separateFigs) figure(); end
plot(timeArr,velGlobVec(3,1:n),timeArr,velGlobVecK(3,1:n), ...
    timeArr,velLocVec(3,1:n),timeArr,velLocVecK(3,1:n), ...
    timeArr,vel1DLoc,timeArr,vel1DLocK,timeArr,vel1DRaw)
title('z Velocity Data')
xlabel('time (s)')
ylabel('Acceleration (m/s^2)')
legend('Global-s', 'Global-K', 'Local-s','Local-K','Local1D-s','Local1D-K','Raw z')

%% PlotGlobalPosition
if(separateFigs) figure(); end
plot(timeArr,posGlobVec(1,1:n),timeArr,posGlobVec(2,1:n), ...
    timeArr,posGlobVec(3,1:n),timeArr,posGlobVecK(1,1:n),timeArr,posGlobVecK(2,1:n), ...
    timeArr,posGlobVecK(3,1:n))
title('Global Position Data')
xlabel('time (s)')
ylabel('Position (m)')
legend('x', 'y', 'z','x_K', 'y_K', 'z_K')

%% PlotLocalAcceleration
if(separateFigs) figure(); end
plot(timeArr,accelLocVec(1,1:n),timeArr,accelLocVec(2,1:n), ...
    timeArr,accelLocVec(3,1:n),timeArr,accelLocVecK(1,1:n),timeArr,accelLocVecK(2,1:n), ...
    timeArr,accelLocVecK(3,1:n))
title('Local Acceleration Data')
xlabel('time (s)')
ylabel('Acceleration (m/s^2)')
legend('a_x', 'a_y', 'a_z','a_{xK}', 'a_{yK}', 'a_{zK}')

%% PlotLocalVelocity
if(separateFigs) figure(); end
plot(timeArr,velLocVec(1,1:n),timeArr,velLocVec(2,1:n), ...
    timeArr,velLocVec(3,1:n),timeArr,velLoc,timeArr,vel1DLoc,timeArr,velLocVecK(1,1:n) ...
    ,timeArr,velLocVecK(2,1:n), ...
    timeArr,velLocVecK(3,1:n),timeArr,velLocK,timeArr,vel1DLocK)
title('Local Velocity Data')
xlabel('time (s)')
ylabel('Velocity (m/s)')
legend('v_x', 'v_y', 'v_z', 'v_{tot}','v_{1D}','v_{xK}', 'v_{yK}', ...
    'v_{zK}', 'v_{totK}','v_{1DK}')
%% PlotLocalVelocityII
if(separateFigs) figure(); end
plot(timeArr,velLocVec(3,1:n),timeArr,velLoc,timeArr,vel1DLoc, ...
    timeArr,velLocVecK(3,1:n),timeArr,velLocK,timeArr,vel1DLocK)
title('Local Velocity Data')
xlabel('time (s)')
ylabel('Velocity (m/s)')
legend('v_z', 'v_{tot}','v_{1D}', ...
    'v_{zK}', 'v_{totK}','v_{1DK}')

%% GetAltimeterdata
timeAlt=transpose(AIM_Data.time_7(isfinite(AIM_Data.time_7)));
Pdata=transpose(AIM_Data.pressure(isfinite(AIM_Data.pressure)));
altMSL=transpose(AIM_Data.PressureMSL(isfinite(AIM_Data.PressureMSL)));
altAGL=transpose(AIM_Data.PressureAGL(isfinite(AIM_Data.PressureAGL)));

if(separateFigs) figure(); end
plot(timeAlt,altMSL, timeAlt,altAGL)
title('Pressure Altimeter Data')
xlabel('time (s)')
ylabel('Altitude MSL (m), Altitude AGL (m)')
legend('MSL', 'AGL')

%% Get Altimeter Parts & Sync w GPS
PBase = Pdata(timeAlt<0);
altMSLBase = altMSL(timeAlt<0);
altAGLBase = altAGL(timeAlt<0);
timeAltBoost = timeAlt(timeAlt>=0 & timeAlt<=apogeeTime);
timeAltDesc = timeAlt(timeAlt>apogeeTime);
altMSLSyncGPS = interp1(timeAlt,altMSL,timeGPSDesc,'pchip');

if(separateFigs) figure(); end
plot(timeGPSDesc,GPSMSLDesc, timeGPSDesc,altMSLSyncGPS)
title('GPS and Altimeter Data')
xlabel('time (s)')
ylabel('GPS Altitude MSL (m), Altimeter Altitude MSL (m)')
legend('GPS', 'ALT')

%% Get GPS vs Alt Slope Intercept
coeffs = polyfit(altMSLSyncGPS, GPSMSLDesc, 1)
altMSLCorr = coeffs(1)*altMSL +coeffs(2);
altAGLCorr = altMSLCorr - mean (altMSLCorr(timeAlt<0));

if(separateFigs) figure(); end
plot(timeAlt,altMSLCorr,'.',timeAlt,altMSL,'.',timeGPS,GPSMSL,'+')
title('GPS and Corrected Altimeter Data')
xlabel('time (s)')
ylabel('Altitude MSL (m)')
legend('corr MSL','raw MSL', 'GPS')

%% Resample Corrected Altitude MSL
altAGLCorrrs = interp1(timeAlt,altAGLCorr,timeArr,'pchip');

if(separateFigs) figure(); end
plot(timeArr,posGlobVec(3,1:n),timeArr,altAGLCorrrs)
title('Corrected Altimeter Data and IMU Altitude')
xlabel('time (s)')
ylabel('Altitude AGL (m)')
legend('IMU','Alt AGL')

%% Get Temperature vs Altitude
TCCrs = interp1(TtimeC,TCC,timeAltDesc,'pchip');
TDCrs = interp1(TtimeD,TDC,timeAltDesc,'pchip');
altAGLDesc = altAGLCorr(timeAlt>apogeeTime);

T_ground = 17.41; %degrees celcius; temp in the shade 
%calculated from mean(TCC(10:196))
standardLapseT = T_ground - 0.0065 * altAGLDesc;

if(separateFigs) figure(); end
plot(altAGLDesc,TCCrs,'.',altAGLDesc,TDCrs,'.', altAGLDesc, standardLapseT, '.')
title('Temperature vs. Altitude')
xlabel('Altitude AGL (m)')
ylabel('Temperature (°C)')
legend('Channel C','Channel D', 'Standard Lapse')

%% PlotPressuredata


%timeAlt=transpose(AIM_Data.time_7(isfinite(AIM_Data.time_7)));
%timeAltDesc = timeAlt(timeAlt>apogeeTime); %recall
Pdata=transpose(AIM_Data.pressure(isfinite(AIM_Data.pressure)));

Pdata_desc = Pdata(timeAlt>apogeeTime);
altMSLDesc = altMSLCorr(timeAlt>apogeeTime);

M = 0.0289644;%kg/m
g = 9.80665;
R = 8.31432; 
D =  -0.0065;
P0 = 101325; 
T0 = T_ground + 273;

%standard lapse rate
P_standard = P0*(1+D*altMSLDesc./T0).^(g*M/(-D*R));

if(separateFigs) figure(); end
plot(altMSLDesc, Pdata_desc,altMSLDesc, P_standard)
title('Pressure versus Altitude')
xlabel('Altitude (MSL)')
ylabel('Pressure (Pa)')
legend('Measured Pressure', 'Standard Model Pressure')


%% Calculate drag coefficient
%code provided by Makoto.

meanTDesc = mean([TCCrs;TDCrs]);
g = 1.4;
R = 8.314;
M = 0.0289644;
c = sqrt(g*R*(meanTDesc+273.15)/M);

[altAGLDesc,I1] = sort(altAGLDesc);
c = c(I1);

figure()
plot(altAGLDesc(altAGLDesc<3500),c(altAGLDesc<3500))
% plot(posGlobVec(3,1:n),c2)

[altAGLDesc, ind] = unique(altAGLDesc(altAGLDesc<3500)); 

c2 = interp1(altAGLDesc,c(ind),posGlobVec(3,1:n));

[SZ,I1] = sort(posGlobVec(3,1:n));

mR = 1.08 + 0.375;
AP = pi*(4.17/200)^2;
rho = 1.225;

Y2 = -2*mR*accelGlobVec(3,I1)./(rho*AP);
Y3 = Y2./(velGlobVec(3,I1).^2);

I1 = I1(I1>1200);
SZ = SZ(I1);

MachN = velGlobVec(3,I1)./c2(I1);

figure()

plot(Y3(I1),MachN)
title('CD vs. Mach Number');
ylabel('Mach Number')
xlabel('Drag Coefficient');

%% Thrust curve
usefulSamples = timeArr < 1.172; %logical indices

timeBoostAccel = timeArr(usefulSamples);
boostAccelLocVecK = accelLocVecK(:, usefulSamples); 
boostVelLocVecK = velLocVecK(:, usefulSamples); 

figure()
plot(timeBoostAccel, boostAccelLocVecK(3,:),timeBoostAccel, boostVelLocVecK(3,:));
title('Acceleration and Velocity Expressed in Local Orientation');
xlabel('Time (s)');
ylabel('Acceleration (m/s^2) or Velocity (m/s)');
legend('Local Acceleration - K', 'Local Velocity - K');

%calculate, plot thrust curve
[T, mr, time] = calcThrust(boostAccelLocVecK(3,:)', boostVelLocVecK(3,:)', sp); %note transpose

%% Calculates thrust and mass curves given acceleration and velocity curves
%The math behind the dynamics is at 
% https://www.overleaf.com/9711582518ykhhfxgxryhb
% INPUTS:
% acc -- acceleration curve (a(t) + g)
% vel -- velocity curve
% dt -- time step for both
% OUTPUTS:
% mr -- mass curve
% T -- mass curve
% time -- time vector


function [T, mr, time] = calcThrust( acc, vel, dt)
    %constants
    CD = 0.275; %total guess. Who knows? Is it even knowable?
    
    %http://www.thrustcurve.org/simfilesearch.jsp?id=2169 motor mass,
    %casing, I_sp.
    %http://pages.hmc.edu/spjut/AdvRoc/FlightData.md.html#flightdatafiles/e190ajprototyperocket
    %rocket mass
    m_roc = 1.0815; %initial mass of rocket in kg
    m_cas = 0.375; %casing mass kg
    m_mot = 0.5; %propellant mass in kg
    m_0 = m_roc + m_mot + m_cas;
    AP = pi*(0.0406908 / 2)^2; %projected area in m^2
    
    rho = 1.2250; %air density in kg/m^3
    I_sp = 974.9 / (0.5 * 9.81); %specific impulse (s)
    g_0 = 9.81; %gravitational constant (m/s^2)
    v_e = I_sp * g_0; %exit velocity (m/s)
    
    
    %initialize mass curve to constant m_0
    mr = m_0 * ones(length(acc), 1);
    
    T = zeros(length(acc), 1);
    
    time = dt * [1:length(acc)];
    
    %iteratively revise mass and thrust curves
    for i = 1:10
        
        T = reviseThrust(acc, vel, mr, CD, AP, rho);
        
        mr = reviseMass(T, dt, m_mot, v_e, m_0);
        
    end
    
    figure()
        clf;
        plot(time, mr, 'k');
        title('Mass Curve');
        ylabel('Mass (kg)');
        xlabel('Time (s)');
        axis([0 1.3 0 2])
        
    figure()
        clf;
        plot(time, T, 'r');
        title('Thrust Curve');
        ylabel('Thrust (N)');
        xlabel('Time (s)');
        axis([0 1.3  0 1000]);
        
        pause(0.1)

end
    

function T = reviseThrust( acc, vel, mr, CD, AP, rho)
    %use rocket equation to find thrust curve given acc and vel curves
    
    T = 0.5 * CD * AP * rho * vel.^2 + mr .* acc;
end

function mr = reviseMass( T, dt, m_mot, v_e, m_0)
    %adjust mass curve so that mass depends on the integral of thrust
    % and scale mass curve so that expelled mass is the mass of the motor
    
    %expended mass is integral of thrust
    expended_mass = 1/v_e * cumtrapz(T) * dt; 
    
    %expended mass must total motor mass
    expended_mass = expended_mass * m_mot/expended_mass(end);
    
    mr = m_0 - expended_mass;
end

%% Conclusion
% The advantage of 3D Kalman filtering compared to Lab 2's approach of 
% encoding simple linear kinematics without sensor fusion is theoretically
% great. In Lab 2, we used the explicit Euler method to approximate the
% position and velocity at each time step -- the common formula 
% x = 1/2 a*t^2 + v_0 *t 
% that we came to know and love from high school physics.
%
% However, I am not convinced that it is worth the effort for the task of
% calculating thrust curves and drag coefficients. We got perfectly
% acceptable thrust curve shapes in Lab 2, and these ones seem off by a
% scalar factor anyway. Perhaps it is an issue of the Kalman filter being
% indisputably more accurate if other parts of the code are implemented
% correctly, i.e. things such as the estimated drag coefficient might screw
% up the output of thrust curve generation.
%
% It would be interesting to compare the performance of trapezoid integration
% to Kalman state estimation, or other schemes such as a particle filter,
% EKF, or other estimator.

