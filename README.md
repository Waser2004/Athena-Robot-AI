# Athena (Robot AI)

The AI training and model development layer of the robotic system. Athena handles machine learning, data pipelines, simulation, training, and evaluation to enable the robot's perception and decision-making capabilities.

## Role in the Ecosystem

Athena is part of a three-layer robotic architecture:

- **[Athena](https://github.com/Waser2004/Athena-Robot-AI)** → AI Training & Model Development  
  Handles machine learning, data pipelines, simulation, training, and evaluation
  
- **[Hermes](https://github.com/Waser2004/Hermes-Robot-Control)** → Control & Orchestration Layer  
  Manages API endpoints, command logic, system communication, and high-level task execution
  
- **[Hephaestus](https://github.com/Waser2004/Hephaestus-Robot-Firmware)** → Embedded & Hardware Layer  
  Runs firmware, microcontroller logic, low-level motor control, and sensor management

## What This Repository Does

This repository provides the data generation and model training scripts for:

- **Cube Detection** - Training models to identify the target cube in images
- **Cube Localisation** - Fine-tuning models to determine precise 3D positions of the target cube

## Model Architecture Overview

**Cube Detection Pipeline:**
- Uses a CNN-based object detector to identify and locate the cube's bounding box in the image
- Outputs: Whether a cube is visible and its approximate location

**Cube Localisation Pipeline:**
- Takes detected cube regions and refines them with a pre-trained backbone (ResNet34) + regression head
- Outputs: Precise 3D coordinates (x, y, z) for the cube's position in space
- Trained on spatially-split datasets to ensure robustness across different workplate regions

The two models work in sequence: Detection identifies the cube, then Localisation determines its exact position for the robot to grasp.