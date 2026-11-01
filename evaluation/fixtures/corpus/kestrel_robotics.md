# Kestrel Robotics — Warehouse Pilot

Kestrel Robotics is a fictional engineering laboratory testing autonomous mobile robots in a small warehouse. Robots localize using lidar and wheel odometry, then generate candidate routes through mapped aisles. A human supervisor authorizes changes to restricted routes. The test fleet consists of six mobile robots and two charging docks.

The control software logs navigation failures and battery temperature. A safety stop is triggered when the front lidar reports an obstacle closer than 0.6 meters. Engineers replay selected incidents in simulation before changing navigation parameters.

The pilot tracks package bins with QR codes but does not process medical samples or scan historical newspapers. The project measures average delivery time and the number of interventions required per shift. All operational experiments occur in a fenced test area.
